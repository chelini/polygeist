#include "mlirGen.h"
#include "mlirAffineExprGen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

namespace teckyl {

// Resursively maps the function `fn` to `tree` and all of its
// descendants in preorder.
static void mapRecursive(const lang::TreeRef &tree,
                         std::function<void(const lang::TreeRef &)> fn) {
  fn(tree);

  for (auto e : tree->trees())
    mapRecursive(e, fn);
}

// Returns a map with one entry per output tensor specifying their
// ranks for the TC definition `def`. If the same tensor is indexed
// with multiple ranks (e.g., C(i, j) = ... and C(i, j, k) = ..., a fatal
// error occurs.
std::unordered_map<std::string, size_t>
MLIRGenImpl::collectOutputRanks(const lang::Def &def) {
  std::set<std::string> outParamNames;
  std::unordered_map<std::string, size_t> ranks;

  for (const lang::Param &outParam : def.returns())
    outParamNames.insert(outParam.ident().name());

  for (const lang::Comprehension &compr : def.statements()) {
    std::string name = compr.ident().name();
    size_t rank = compr.indices().size();

    if (outParamNames.find(name) != outParamNames.end()) {
      auto it = ranks.find(name);

      if (it != ranks.end()) {
        if (it->second != rank) {
          // TODO: add name 'name'
          llvm::report_fatal_error(
              "Multiple ranks found for output tensor for ");
        }
      } else {
        ranks.insert({name, rank});
      }
    }
  }

  return ranks;
}

mlir::FloatType MLIRGenImpl::getFloatType(int kind) {
  switch (kind) {
  case lang::TK_DOUBLE:
    return builder_.getF64Type();
  case lang::TK_FLOAT:
    return builder_.getF32Type();
  case lang::TK_FLOAT16:
    return builder_.getF16Type();
  case lang::TK_FLOAT32:
    return builder_.getF32Type();
  case lang::TK_FLOAT64:
    return builder_.getF64Type();
  default:
    llvm_unreachable("Not a float type");
  }
}

mlir::Type MLIRGenImpl::getScalarType(int kind) {
  switch (kind) {
  case lang::TK_DOUBLE:
  case lang::TK_FLOAT:
  case lang::TK_FLOAT16:
  case lang::TK_FLOAT32:
  case lang::TK_FLOAT64:
    return getFloatType(kind);
  case lang::TK_INT2:
    return builder_.getIntegerType(2);
  case lang::TK_INT4:
    return builder_.getIntegerType(4);
  case lang::TK_INT8:
    return builder_.getIntegerType(8);
  case lang::TK_INT16:
    return builder_.getIntegerType(16);
  case lang::TK_INT32:
    return builder_.getIntegerType(32);
  case lang::TK_INT64:
    return builder_.getIntegerType(64);
  case lang::TK_SIZET:
    return builder_.getIndexType();
  default:
    llvm_unreachable("Unsupported type");
  }
}

mlir::Type MLIRGenImpl::getMemRefType(const lang::TensorType &tensorType,
                                      size_t ndims) {
  mlir::Type scalarType = getScalarType(tensorType.scalarType());
  if (ndims > 0)
    return mlir::MemRefType::get(
        llvm::SmallVector<int64_t>(ndims, mlir::ShapedType::kDynamic),
        scalarType);
  return scalarType;
}

// Collects the set of iterators of a comprehensions by listing all
// identifiers and retaining only those that are not in the symbol
// table `symTab`.
static std::unordered_map<std::string, teckyl::IteratorKind>
collectIterators(const lang::Comprehension &comprehension,
                 const llvm::MapVector<llvm::StringRef, mlir::Value> &symTab) {
  std::unordered_map<std::string, IteratorKind> iterators;

  for (const lang::Ident &lhsIndex : comprehension.indices())
    iterators.emplace(lhsIndex.name(), IteratorKind::LHS);

  mapRecursive(comprehension.rhs(), [&](const lang::TreeRef &t) {
    if (t->kind() == lang::TK_IDENT) {
      std::string name = lang::Ident(t).name();

      if (iterators.find(name) == iterators.end() && symTab.count(name) == 0) {
        iterators.emplace(name, IteratorKind::RHSOnly);
      }
    }
  });
  return iterators;
}

static llvm::SmallVector<std::pair<std::string, lang::ListView<lang::TreeRef>>,
                         8>
collectTensorAccessesSeq(const lang::TreeRef &t) {
  llvm::SmallVector<std::pair<std::string, lang::ListView<lang::TreeRef>>, 8>
      res;

  // Collect all tensor accesses in subexpressions
  mapRecursive(t, [&](const lang::TreeRef &e) {
    if (e->kind() == lang::TK_ACCESS) {
      lang::Access a = lang::Access(e);
      res.push_back(std::make_pair(a.name().name(), a.arguments()));
    }
    // FIXME: run sema analysis, and collect TK_ACCESS
    if (e->kind() == lang::TK_APPLY) {
      lang::Apply a = lang::Apply(e);
      res.push_back(std::make_pair(a.name().name(), a.arguments()));
    }
  });
  return res;
}

mlir::Value MLIRGenImpl::buildLinalgReductionCore(
    const lang::Comprehension &c, mlir::Value tensor,
    const std::unordered_map<std::string, IteratorKind> &iterators,
    const llvm::SmallVectorImpl<std::string> &iteratorSeq,
    mlir::Location location) {

  llvm::SmallVector<mlir::Value> inputOperands;
  llvm::SmallVector<std::string> operandsAsString;
  llvm::SmallVector<mlir::Value> outputOperands;
  llvm::SmallVector<mlir::AffineMap> indexingMaps;
  llvm::SmallVector<std::string> tensorIds;

  llvm::SmallVector<std::pair<std::string, lang::ListView<lang::TreeRef>>, 8>
      tensorAccesses = collectTensorAccessesSeq(c.rhs());

  // Mapping between dimension id and schedule dimension.
  std::unordered_map<std::string, size_t> iteratorDims;
  size_t dim = 0;
  for (std::string it : iteratorSeq)
    iteratorDims.emplace(it, dim++);

  MLIRAffineExprGen affGen(builder_.getContext(), iteratorDims);

  // get codomain dimension for affine map.
  size_t codomainDim = iteratorSeq.size();

  // map tensor name to affine map.
  std::unordered_map<std::string, mlir::AffineMap> tensorToMap;
  // handle inputs.
  for (const std::pair<std::string, lang::ListView<lang::TreeRef>> &access :
       tensorAccesses) {
    llvm::SmallVector<mlir::AffineExpr, 8> affineExprs =
        affGen.buildAffineExpressions(access.second);
    mlir::AffineMap map =
        mlir::AffineMap::get(codomainDim, 0, affineExprs, context_);
    mlir::Value mValue = symbolTable_.lookup(access.first);
    if (isa<mlir::MemRefType>(mValue.getType())) {
      mValue =
          builder_.create<mlir::bufferization::ToTensorOp>(location, mValue);
      symbolTable_[access.first] = mValue;
    }
    inputOperands.push_back(mValue);
    tensorToMap.insert({access.first, map});
    tensorIds.push_back(access.first);
    operandsAsString.push_back(access.first);
  }
  assert(inputOperands.size() == tensorAccesses.size() &&
         "input operands size must equal the number of accesses on the rhs");

  outputOperands.push_back(tensor);
  assert(outputOperands.size() == 1 && "expect a single output");
  llvm::SmallVector<mlir::AffineExpr, 8> affineExprs =
      affGen.buildAffineExpressions(c.indices());
  mlir::AffineMap map =
      mlir::AffineMap::get(codomainDim, 0, affineExprs, context_);
  tensorToMap.insert({c.ident().name(), map});
  tensorIds.push_back(c.ident().name());
  operandsAsString.push_back(c.ident().name());

  // iterator types.
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes;
  for (const std::string &it : iteratorSeq) {
    if (iterators.at(it) == IteratorKind::LHS)
      iteratorTypes.push_back(mlir::utils::IteratorType::parallel);
    else
      iteratorTypes.push_back(mlir::utils::IteratorType::reduction);
  }

  // order the map based on the input.
  for (std::string operand : operandsAsString)
    indexingMaps.push_back(tensorToMap[operand]);

  // TODO: do not push tensor output if dealing with =
  mlir::Operation *genericOp = builder_.create<mlir::linalg::GenericOp>(
      builder_.getUnknownLoc(), tensor.getType(), inputOperands, outputOperands,
      indexingMaps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        assert(tensorIds.size() == args.size() &&
               "tensor ids must be the same as block args");
        // mapping from lang::Tree ids to block arguments.
        std::map<std::string, mlir::Value> valMap;
        int i = 0;
        for (std::string &tensorId : tensorIds) {
          valMap.insert({tensorId, args[i++]});
        }

        MLIRMappedValueExprGen gen(context_, valMap, nestedLoc, nestedBuilder);
        // yeild value.
        mlir::Value result;
        // rhs.
        mlir::Value rhsVal = gen.buildExpr(c.rhs());
        // accumulator for the output tensor is always the last value.
        mlir::Value acc = args[args.size() - 1];

        // build reduction operator.
        switch (c.assignment()->kind()) {
        case lang::TK_PLUS_EQ:
        case lang::TK_PLUS_EQ_B:
          result = gen.buildBinaryExprFromValues<mlir::arith::AddFOp,
                                                 mlir::arith::AddIOp>(
              rhsVal, acc, builder_.getUnknownLoc());
          break;
        case lang::TK_TIMES_EQ:
        case lang::TK_TIMES_EQ_B:
          result = gen.buildBinaryExprFromValues<mlir::arith::MulFOp,
                                                 mlir::arith::MulIOp>(
              rhsVal, acc, builder_.getUnknownLoc());
          break;
        case '=':
          result = rhsVal;
          break;
        default:
          llvm_unreachable("Unsupported operator");
        }

        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, result);
      });
  assert(genericOp->getNumResults() == 1);
  return genericOp->getResults()[0];
}

void MLIRGenImpl::buildTensorInitialization(const std::string tensorName,
                                            mlir::Value tensor,
                                            NeutralElement elem) {
  mlir::Type type = tensor.getType();
  assert(type.isa<mlir::ShapedType>() && "expect a shaped type");
  mlir::ShapedType shapedType = type.dyn_cast<mlir::ShapedType>();
  mlir::Type scalarType = shapedType.getElementType();
  mlir::Value constant;
  if (isMLIRFloatType(scalarType)) {
    if (elem == NeutralElement::Zero)
      constant = builder_.create<mlir::arith::ConstantOp>(
          builder_.getUnknownLoc(), scalarType,
          builder_.getFloatAttr(scalarType, 0.0));
    else
      constant = builder_.create<mlir::arith::ConstantOp>(
          builder_.getUnknownLoc(), scalarType,
          builder_.getFloatAttr(scalarType, 1.0));
  } else if (isMLIRIntType(scalarType)) {
    // TODO: Here we generate only i32. Which is wrong.
    if (elem == NeutralElement::Zero)
      constant = builder_.create<mlir::arith::ConstantOp>(
          builder_.getUnknownLoc(), scalarType, builder_.getI32IntegerAttr(0));
    else
      constant = builder_.create<mlir::arith::ConstantOp>(
          builder_.getUnknownLoc(), scalarType, builder_.getI32IntegerAttr(1));
  }
  auto fillRes = builder_.create<mlir::linalg::FillOp>(builder_.getUnknownLoc(),
                                                       constant, tensor);
  symbolTable_[tensorName] = fillRes.getResult(0);
}

std::pair<mlir::Value, mlir::Value>
MLIRGenImpl::buildComprehension(const lang::Comprehension &c) {
  std::unordered_map<std::string, IteratorKind> iterators =
      collectIterators(c, symbolTable_);
  std::unordered_set<std::string> iteratorSet;
  std::unordered_set<std::string> iteratorSetReduction;

  for (const std::pair<std::string, IteratorKind> it : iterators) {
    iteratorSet.insert(it.first);
    if (it.second == IteratorKind::RHSOnly)
      iteratorSetReduction.insert(it.first);
  }

  // Decide on an arbitrary order of iterators for the loop nest.
  llvm::SmallVector<std::string, 8> iteratorSeq;
  for (std::pair<std::string, IteratorKind> it : iterators)
    iteratorSeq.push_back(it.first);
  std::reverse(iteratorSeq.begin(), iteratorSeq.end());

  const std::string outTensorName = c.ident().name();
  mlir::Value outVal = symbolTable_.lookup(outTensorName);
  assert(outVal && "outMemRefVal not founded in symbolTable");
  if (isa<mlir::MemRefType>(outVal.getType())) {
    outVal = builder_.create<mlir::bufferization::ToTensorOp>(
        builder_.getUnknownLoc(), outVal);
    symbolTable_[outTensorName] = outVal;
  }

  if (c.assignment()->kind() == lang::TK_PLUS_EQ_B)
    buildTensorInitialization(outTensorName, outVal, NeutralElement::Zero);
  else if (c.assignment()->kind() == lang::TK_TIMES_EQ_B)
    buildTensorInitialization(outTensorName, outVal, NeutralElement::One);
  else if (c.assignment()->kind() == lang::TK_MAX_EQ_B ||
           c.assignment()->kind() == lang::TK_MIN_EQ_B) {
    llvm_unreachable("Unsupported reduction");
  }

  outVal = symbolTable_.lookup(outTensorName);
  mlir::Value resultComp = buildLinalgReductionCore(
      c, outVal, iterators, iteratorSeq, builder_.getUnknownLoc());
  symbolTable_[outTensorName] = resultComp;
  return std::make_pair(resultComp, outVal);
}

mlir::func::FuncOp MLIRGenImpl::buildFunction(const std::string name,
                                              const lang::Def &def) {
  llvm::SmallVector<mlir::Type, 8> funcArgsTypes;

  for (lang::Param param : def.params()) {
    lang::TensorType tensorType = param.tensorType();
    mlir::Type mlirMemRefType =
        getMemRefType(tensorType, tensorType.dims().size());
    funcArgsTypes.push_back(mlirMemRefType);
  }

  std::unordered_map<std::string, size_t> outputRanks = collectOutputRanks(def);

  for (lang::Param param : def.returns()) {
    lang::TensorType tcTensorType = param.tensorType();
    std::string name = param.ident().name();
    mlir::Type mlirMemRefType = getMemRefType(tcTensorType, outputRanks[name]);
    funcArgsTypes.push_back(mlirMemRefType);
  }

  // Build function signature.
  mlir::FunctionType funcType =
      builder_.getFunctionType(funcArgsTypes, std::nullopt);
  mlir::func::FuncOp funcOp =
      mlir::func::FuncOp::create(builder_.getUnknownLoc(), name, funcType);
  mlir::SymbolTable::setSymbolVisibility(
      funcOp, mlir::SymbolTable::Visibility::Private);

  // Add block for function body.
  mlir::Block *entryBlock = funcOp.addEntryBlock();
  builder_.setInsertionPointToStart(entryBlock);

  // Add input and output tensor to symbol table.
  int i = 0;
  for (lang::Param param : def.params()) {
    mlir::BlockArgument arg = funcOp.getArgument(i++);
    symbolTable_[param.ident().name()] = arg;
  }
  for (lang::Param param : def.returns()) {
    mlir::BlockArgument arg = funcOp.getArgument(i++);
    symbolTable_[param.ident().name()] = arg;
  }

  // Build function body.
  std::pair<mlir::Value, mlir::Value> resultComp;
  for (const lang::Comprehension &comprehension : def.statements())
    resultComp = buildComprehension(comprehension);
  builder_.create<mlir::bufferization::MaterializeInDestinationOp>(
      builder_.getUnknownLoc(), resultComp.first, resultComp.second);

  builder_.create<mlir::func::ReturnOp>(builder_.getUnknownLoc());
  return funcOp;
}

mlir::func::FuncOp
buildMLIRFunction(mlir::MLIRContext *context, mlir::OpBuilder &builder,
                  llvm::MapVector<llvm::StringRef, mlir::Value> &symbolTable,
                  const std::string name, const lang::Def &tc) {
  MLIRGenImpl generator(context, builder, symbolTable);
  return generator.buildFunction(name, tc);
}

mlir::transform::NamedSequenceOp buildMLIRTactic(mlir::MLIRContext *context,
                                                 mlir::OpBuilder &builder,
                                                 const std::string name,
                                                 const lang::Tac &tac) {
  // void function
  auto fnType = builder.getFunctionType({}, {});
  return builder.create<mlir::transform::NamedSequenceOp>(
      builder.getUnknownLoc(), builder.getStringAttr(name),
      mlir::TypeAttr::get(fnType), /*sym_visibility=*/nullptr,
      /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
}

mlir::Value MLIRMappedValueExprGen::buildConstant(const lang::Const &cst) {
  assert(false && "not implemented");
  return nullptr;
}

mlir::Value MLIRMappedValueExprGen::buildCmpExpression(const lang::TreeRef &t) {
  assert(false && "not implemented");
  return nullptr;
}

mlir::Value MLIRMappedValueExprGen::buildIdent(const lang::Ident &i) {
  assert((valMap_.find(i.name()) != valMap_.end()) && "cannot find ident");
  return valMap_[i.name()];
}

mlir::Value MLIRMappedValueExprGen::buildIndexLoadExpr(const lang::Access &a) {
  return buildExpr(a.name());
}

mlir::Value
MLIRMappedValueExprGen::buildTernaryExpression(const lang::TreeRef &t) {
  assert(false && "not implemented");
  return nullptr;
}

// Builds a binary MLIR expression from a TC expression. Creates an
// operation of type `FOpTy` if the operands are floats or an
// operation of type `IOpTy` if the operands are integers. If the
// operands have different types or if they are neither integers nor
// floats, an error occurs.
template <typename FOpTy, typename IOpTy>
mlir::Value MLIRMappedValueExprGen::buildBinaryExpr(const lang::TreeRef &t) {
  return buildBinaryExprFromValues<FOpTy, IOpTy>(buildExpr(t->trees().at(0)),
                                                 buildExpr(t->trees().at(1)),
                                                 builder_.getUnknownLoc());
}

mlir::Value MLIRMappedValueExprGen::buildExprImpl(const lang::TreeRef &t) {
  switch (t->kind()) {
  case '+':
    return buildBinaryExpr<mlir::arith::AddFOp, mlir::arith::AddIOp>(t);
  case '-':
    return buildBinaryExpr<mlir::arith::SubFOp, mlir::arith::SubIOp>(t);
  case '*':
    return buildBinaryExpr<mlir::arith::MulFOp, mlir::arith::MulIOp>(t);
  case '/':
    return buildBinaryExpr<mlir::arith::DivFOp, mlir::arith::DivUIOp>(t);
  case '?':
    return buildTernaryExpression(t);
  case '<':
  case '>':
  case lang::TK_LE:
  case lang::TK_GE:
  case lang::TK_EQ:
    return buildCmpExpression(t);
  case lang::TK_NUMBER:
  case lang::TK_CONST:
    return buildConstant(lang::Const(t));
  case lang::TK_IDENT:
    return buildIdent(lang::Ident(t));
  case lang::TK_APPLY:
    return buildExpr(lang::Apply(t).name());
  case lang::TK_ACCESS:
    return buildIndexLoadExpr(lang::Access(t));
  default: {
    llvm::errs() << lang::pretty_tree(t) << "\n";
    llvm_unreachable("Unknown tree type\n");
  }
  }
}

mlir::Value MLIRMappedValueExprGen::buildExpr(const lang::TreeRef &t) {
  mlir::Value builtValue = buildExprImpl(t);
  return builtValue;
}

} // end namespace teckyl
