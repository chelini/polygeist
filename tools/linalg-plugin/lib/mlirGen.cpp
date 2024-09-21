#include "mlirGen.h"
#include "mlirAffineExprGen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

#define DEBUG_TYPE "teckyl"

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

static void applyRecursive(const lang::TreeRef &tree,
                           std::function<void(const lang::TreeRef &)> fn) {
  fn(tree);
  for (auto e : tree->trees())
    applyRecursive(e, fn);
}

mlir::FailureOr<mlir::OpResult>
emitMatcherComprehension(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value opToMatch,
                         lang::Comprehension comprehension, bool isFirst) {
  LLVM_DEBUG(llvm::dbgs() << lang::pretty_tree(comprehension) << "\n");

  // Consider only simple comprehension at the moment. With 'simple' meaning
  // single statment and single result.
  if (comprehension.assignment()->kind() != '=')
    return mlir::failure();

  auto rhs = comprehension.rhs();
  llvm::SmallVector<lang::Access> rhsAccesses;
  applyRecursive(rhs, [&](const lang::TreeRef &t) {
    if (t->kind() == lang::TK_ACCESS) {
      rhsAccesses.push_back(lang::Access(t));
    }
  });
  lang::ListView<lang::Ident> lhsIndexes = comprehension.indices();
  int64_t parallelDims = 0;
  int64_t reductionDims = 0;
  llvm::SetVector<llvm::StringRef> uniqueDims;
  for (const auto &lhsIndex : lhsIndexes)
    uniqueDims.insert(lhsIndex.name());

  for (const lang::Access &rhsAccess : rhsAccesses) {
    lang::ListView<lang::TreeRef> rhsDims = rhsAccess.arguments();
    for (const lang::TreeRef &rhsDim : rhsDims) {
      lang::Ident dim = lang::Ident(rhsDim);
      if (uniqueDims.count(dim.name()) == 0)
        reductionDims++;
    }
  }
  parallelDims = uniqueDims.size();

  if (reductionDims != 0)
    return mlir::failure();

  LLVM_DEBUG(llvm::dbgs() << "#parallel dims: " << parallelDims << "\n");
  LLVM_DEBUG(llvm::dbgs() << "#reduction dims: " << reductionDims << "\n");
  LLVM_DEBUG(llvm::dbgs() << "#input operands: " << rhsAccesses.size() << "\n");

  mlir::transform::AnyValueType anyValueTy =
      mlir::transform::AnyValueType::get(builder.getContext());
  mlir::transform::AnyOpType anyOpTy =
      mlir::transform::AnyOpType::get(builder.getContext());

  mlir::Type returnTy = anyValueTy;
  llvm::SmallVector<mlir::Type> returnTys(rhsAccesses.size() + 1, returnTy);
  if (isFirst)
    returnTys.insert(returnTys.begin(), anyOpTy);

  mlir::transform::MatchStructuredOp matcherComprehension =
      builder.create<mlir::transform::MatchStructuredOp>(
          loc, returnTys, opToMatch,
          mlir::transform::FailurePropagationModeAttr::get(
              builder.getContext(),
              mlir::transform::FailurePropagationMode::Propagate));
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Region &region = matcherComprehension.getRegion();
    region.push_back(new mlir::Block());
    mlir::Block *block = matcherComprehension.getBody();
    block->addArguments({anyOpTy}, {loc});
    builder.setInsertionPointToStart(block);

    opToMatch = block->getArguments()[0];

    // Match number of inputs and init.
    mlir::transform::ParamConstantOp cstNumInputs =
        builder.create<mlir::transform::ParamConstantOp>(
            loc,
            mlir::transform::ParamType::get(builder.getContext(),
                                            builder.getI64Type()),
            builder.getI64IntegerAttr(rhsAccesses.size()));
    mlir::transform::MatchStructuredNumInputsOp numInputs =
        builder.create<mlir::transform::MatchStructuredNumInputsOp>(
            loc,
            mlir::transform::ParamType::get(builder.getContext(),
                                            builder.getI64Type()),
            opToMatch);
    builder.create<mlir::transform::MatchParamCmpIOp>(
        loc, numInputs, cstNumInputs, mlir::transform::MatchCmpIPredicate::eq);
    mlir::transform::ParamConstantOp cstNumInits =
        builder.create<mlir::transform::ParamConstantOp>(
            loc,
            mlir::transform::ParamType::get(builder.getContext(),
                                            builder.getI64Type()),
            builder.getI64IntegerAttr(1));
    mlir::transform::MatchStructuredNumInitsOp numInits =
        builder.create<mlir::transform::MatchStructuredNumInitsOp>(
            loc,
            mlir::transform::ParamType::get(builder.getContext(),
                                            builder.getI64Type()),
            opToMatch);
    builder.create<mlir::transform::MatchParamCmpIOp>(
        loc, numInits, cstNumInits, mlir::transform::MatchCmpIPredicate::eq);

    // Match number of iterators (aka rank).
    mlir::transform::ParamConstantOp cstRank =
        builder.create<mlir::transform::ParamConstantOp>(
            loc,
            mlir::transform::ParamType::get(builder.getContext(),
                                            builder.getI64Type()),
            builder.getI64IntegerAttr(parallelDims + reductionDims));
    mlir::transform::MatchStructuredRankOp rank =
        builder.create<mlir::transform::MatchStructuredRankOp>(
            loc,
            mlir::transform::ParamType::get(builder.getContext(),
                                            builder.getI64Type()),
            opToMatch);
    builder.create<mlir::transform::MatchParamCmpIOp>(
        loc, rank, cstRank, mlir::transform::MatchCmpIPredicate::eq);

    // Match the iterator.
    builder.create<mlir::transform::MatchStructuredDimOp>(
        loc, mlir::Type(), opToMatch, builder.getDenseI64ArrayAttr({}),
        /*is_inverted=*/nullptr, /*is_all=*/builder.getUnitAttr(),
        /*parallel=*/builder.getUnitAttr(), /*reduction=*/nullptr);

    // Capture all init/inputs operands. If we are matching the
    // last generic we also push back the operation itself (the one that need to
    // be replaced).
    llvm::SmallVector<mlir::Value> capturedInputs;
    for (size_t idx = 0; idx < rhsAccesses.size(); idx++) {
      mlir::transform::MatchStructuredInputOp matchedInput =
          builder.create<mlir::transform::MatchStructuredInputOp>(
              loc, anyValueTy, opToMatch, idx);
      capturedInputs.push_back(matchedInput.getResult());
    }
    mlir::transform::MatchStructuredInitOp matchedInit =
        builder.create<mlir::transform::MatchStructuredInitOp>(loc, anyValueTy,
                                                               opToMatch, 0);
    capturedInputs.push_back(matchedInit.getResult());
    assert(capturedInputs.size() == rhsAccesses.size() + 1);
    if (isFirst)
      capturedInputs.insert(capturedInputs.begin(), opToMatch);
    builder.create<mlir::transform::MatchStructuredYieldOp>(loc,
                                                            capturedInputs);
  }
  return matcherComprehension.getOutputs()[0];
}

static std::string getOperandName(lang::Comprehension comprehension) {
  auto rhs = comprehension.rhs();
  llvm::SmallVector<lang::Access> rhsAccesses;
  applyRecursive(rhs, [&](const lang::TreeRef &t) {
    if (t->kind() == lang::TK_ACCESS) {
      rhsAccesses.push_back(lang::Access(t));
    }
  });
  assert(rhsAccesses.size() && "multi operand not supported yet");
  lang::Ident variableName = rhsAccesses[0].name();
  return variableName.name();
}

static size_t getNumOperands(lang::Comprehension comprehension) {
  auto rhs = comprehension.rhs();
  size_t numOperands = 1; // account for the single output.
  applyRecursive(rhs, [&](const lang::TreeRef &t) {
    if (t->kind() == lang::TK_ACCESS) {
      numOperands++;
    }
  });
  return numOperands;
}

static mlir::FailureOr<mlir::transform::NamedSequenceOp>
buildMatcher(mlir::MLIRContext *context, mlir::OpBuilder &builder,
             mlir::Location loc, const lang::Tac &tac) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::transform::AnyOpType anyOpTy = mlir::transform::AnyOpType::get(context);
  mlir::transform::AnyValueType anyValueTy =
      mlir::transform::AnyValueType::get(context);
  lang::ListView<lang::Comprehension> replacements = tac.replacement();
  assert(replacements.size() <= 1 && "expect at most 1 replacement");
  size_t numberOfReplOperands = getNumOperands(replacements[0]);
  LLVM_DEBUG(llvm::dbgs() << "operand to replace: " << numberOfReplOperands
                          << "\n");
  // args: op_to_match
  // result: [op_to_replace, arguments...]
  llvm::SmallVector<mlir::Type> resultTyNamedSeq = {numberOfReplOperands,
                                                    anyValueTy};
  resultTyNamedSeq.insert(resultTyNamedSeq.begin(), anyOpTy);
  auto fnType = builder.getFunctionType({anyOpTy}, resultTyNamedSeq);
  mlir::transform::NamedSequenceOp matcherOp =
      builder.create<mlir::transform::NamedSequenceOp>(
          loc, builder.getStringAttr("pattern"), mlir::TypeAttr::get(fnType),
          /*sym_visibility=*/nullptr,
          /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
  matcherOp.setArgAttrs(
      0, mlir::DictionaryAttr::get(
             context,
             {mlir::NamedAttribute(builder.getStringAttr("transform.readonly"),
                                   builder.getUnitAttr())}));

  mlir::Region &region = matcherOp.getBody();
  region.push_back(new mlir::Block());
  mlir::Block &block = matcherOp.getBody().front();
  block.addArguments({anyOpTy}, {loc});

  builder.setInsertionPointToStart(&block);

  llvm::SmallVector<mlir::Value> allMatched;
  mlir::Value opToMatch = matcherOp.getArgument(0);
  mlir::Value rootMatch;
  mlir::Value matched;
  lang::ListView<lang::Comprehension> patterns = tac.pattern();
  llvm::MapVector<std::string, mlir::Value, llvm::StringMap<unsigned>> symTable;
  for (int idx = patterns.size() - 1; idx >= 0; idx--) {
    std::string lhsOperandName = patterns[idx].ident().name();
    auto it = symTable.find(lhsOperandName);
    if (it != symTable.end()) {
      mlir::Operation *op = it->second.getDefiningOp();
      assert(isa<mlir::transform::MatchStructuredOp>(op));
      mlir::transform::MatchStructuredOp prevMatcher =
          cast<mlir::transform::MatchStructuredOp>(op);
      // Note here we assume to have at most 1 operand on the rhs (see
      // getOperandName). this means that the input is either at position 0 or
      // position 1 in the result.
      int64_t resultIdx = (idx == 0) ? 1 : 0;
      opToMatch = builder.create<mlir::transform::GetDefiningOp>(
          loc, anyOpTy, prevMatcher.getResult(resultIdx));
    }
    mlir::FailureOr<mlir::Value> maybeMatch = emitMatcherComprehension(
        builder, loc, opToMatch, patterns[idx], idx == patterns.size() - 1);
    if (mlir::failed(maybeMatch))
      return mlir::failure();
    matched = *maybeMatch;
    allMatched.push_back(matched);
    if (idx == patterns.size() - 1)
      rootMatch = matched;
    std::string rhsOperandName = getOperandName(patterns[idx]);
    symTable.insert(std::make_pair(rhsOperandName, matched));
  }

  llvm::SmallVector<mlir::Value> toReturn = {rootMatch};
  // Get the operand to replace. We match in reverse order.
  for (auto [idx, v] : llvm::enumerate(allMatched)) {
    mlir::Operation *op = v.getDefiningOp();
    assert(op && isa<mlir::transform::MatchStructuredOp>(op));
    mlir::transform::MatchStructuredOp matcher =
        cast<mlir::transform::MatchStructuredOp>(op);
    if (idx == 0) {
      // we are interested only in the init.
      toReturn.push_back(matcher.getResult(matcher.getNumResults() - 1));
    } else if (idx == allMatched.size() - 1) {
      // we are interested only in the input, jump 1 operand since the first
      // is the operation to replace.
      toReturn.push_back(matcher.getResult(1));
    } else {
      // we are interested only in the input
      toReturn.push_back(matcher.getResult(0));
    }
  }
  builder.create<mlir::transform::YieldOp>(loc, toReturn);
  return matcherOp;
}

static mlir::transform::NamedSequenceOp
buildReplacement(mlir::MLIRContext *context, mlir::OpBuilder &builder,
                 mlir::Location loc, const lang::Tac &tac) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::transform::AnyOpType anyOpTy = mlir::transform::AnyOpType::get(context);
  mlir::transform::AnyValueType anyValueTy =
      mlir::transform::AnyValueType::get(context);
  // XXX hardcoded!!;
  auto fnType = builder.getFunctionType({anyOpTy, anyValueTy, anyValueTy}, {});
  mlir::transform::NamedSequenceOp replacementOp =
      builder.create<mlir::transform::NamedSequenceOp>(
          loc, builder.getStringAttr("replacement"),
          mlir::TypeAttr::get(fnType),
          /*sym_visibility=*/nullptr,
          /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

  for (int idx = 0; idx < replacementOp.getNumArguments(); idx++) {
    replacementOp.setArgAttrs(
        idx,
        mlir::DictionaryAttr::get(
            context,
            {mlir::NamedAttribute(builder.getStringAttr("transform.consumed"),
                                  builder.getUnitAttr())}));
  }

  mlir::Region &region = replacementOp.getBody();
  region.push_back(new mlir::Block());
  mlir::Block &block = replacementOp.getBody().front();
  block.addArguments({anyOpTy, anyValueTy, anyValueTy}, {loc, loc, loc});
  builder.setInsertionPointToStart(&block);
  builder.create<mlir::transform::YieldOp>(loc);
  return replacementOp;
}

mlir::FailureOr<BuiltTactic> buildMLIRTactic(mlir::MLIRContext *context,
                                             mlir::OpBuilder &builder,
                                             const std::string name,
                                             const lang::Tac &tac) {

  mlir::transform::AnyOpType anyOpTy = mlir::transform::AnyOpType::get(context);
  mlir::transform::AnyValueType anyValueTy =
      mlir::transform::AnyValueType::get(context);
  mlir::Location loc = builder.getUnknownLoc();

  mlir::FailureOr<mlir::transform::NamedSequenceOp> matcherOp =
      buildMatcher(context, builder, loc, tac);
  if (mlir::failed(matcherOp))
    return mlir::failure();

  mlir::transform::NamedSequenceOp replacementOp =
      buildReplacement(context, builder, loc, tac);

  mlir::transform::SequenceOp rootOp =
      builder.create<mlir::transform::SequenceOp>(
          loc, mlir::TypeRange(),
          mlir::transform::FailurePropagationMode::Suppress, anyOpTy,
          [&](mlir::OpBuilder bodyBuilder, mlir::Location loc,
              mlir::BlockArgument args) {
            mlir::MLIRContext *ctx = bodyBuilder.getContext();
            mlir::ArrayAttr matcher = mlir::ArrayAttr::get(
                ctx, mlir::SymbolRefAttr::get(ctx, "pattern"));
            mlir::ArrayAttr action = mlir::ArrayAttr::get(
                ctx, mlir::SymbolRefAttr::get(ctx, "replacement"));
            bodyBuilder.create<mlir::transform::ForeachMatchOp>(
                loc, anyOpTy, args, matcher, action);
            bodyBuilder.create<mlir::transform::YieldOp>(loc);
          });

  return BuiltTactic{rootOp, *matcherOp, replacementOp};
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
