#ifndef TECKYL_MLIRGEN_H
#define TECKYL_MLIRGEN_H

#include "mlir/IR/Builders.h"
#include "parser.h"
#include "llvm/ADT/MapVector.h"
#include <string>
#include <unordered_map>

namespace mlir {
namespace func {
class FuncOp;
} // end namespace func
namespace transform {
class NamedSequenceOp;
} // end namespace transform
class MLIRContext;
class Value;
class Type;
class FloatType;
} // namespace mlir

namespace teckyl {

enum class IteratorKind {
  // Iterator appears on the LHS (and may also appear on the RHS)
  LHS,
  // Iterator only on the RHS
  RHSOnly
};

enum class NeutralElement { Zero, One };

static inline bool isMLIRFloatType(mlir::Type t) {
  return t.isF16() || t.isF32() || t.isF64();
}

static inline bool isMLIRIntType(mlir::Type t) {
  return t.isInteger(2) || t.isInteger(4) || t.isInteger(8) ||
         t.isInteger(16) || t.isInteger(32) || t.isInteger(64);
}

// Builds MLIR expressions without control flow from tensor
// expressions. The difference with MLIRValueExprGen is that entire
// subtrees of the tensor expression can be mapped to MLIR values
// (e.g., to map sub-expressions to block or function arguments or to
// avoid re-generation of known sub-expressions).
class MLIRMappedValueExprGen {
public:
  MLIRMappedValueExprGen() = delete;
  MLIRMappedValueExprGen(mlir::MLIRContext *context,
                         std::map<std::string, mlir::Value> &mapping,
                         mlir::Location loc, mlir::OpBuilder &builder)
      : valMap_(mapping), loc_(loc), builder_(builder) {};

  mlir::Value buildExpr(const lang::TreeRef &t);

  // Builds a binary operation from `lhs` and `rhs` associated to the
  // specified location. If both values are float values, the newly
  // created operation is `FOpTyp` and if both values are integer
  // values, `IOpTy` is instantiated. If the values have different types
  // or if they are neither floats nor integers, an error occurs
  template <typename FOpTy, typename IOpTy>
  mlir::Value buildBinaryExprFromValues(mlir::Value lhs, mlir::Value rhs,
                                        mlir::Location loc) {
    mlir::Type resType = lhs.getType();

    if (isMLIRFloatType(resType)) {
      return builder_.create<FOpTy>(loc, lhs, rhs);
    } else if (isMLIRIntType(resType)) {
      return builder_.create<IOpTy>(loc, lhs, rhs);
    } else {
      lhs.getType().dump();
      rhs.getType().dump();
      assert(0 && "broken types!");
    }
  }

private:
  std::map<std::string, mlir::Value> &valMap_;
  mlir::Location loc_;
  mlir::OpBuilder &builder_;

  template <typename FOpTy, typename IOpTy>
  mlir::Value buildBinaryExpr(const lang::TreeRef &t);
  mlir::Value buildExprImpl(const lang::TreeRef &t);
  mlir::Value buildIndexLoadExpr(const lang::Access &a);
  mlir::Value buildIdent(const lang::Ident &i);
  mlir::Value buildCmpExpression(const lang::TreeRef &t);
  mlir::Value buildConstant(const lang::Const &cst);
  mlir::Value buildTernaryExpression(const lang::TreeRef &t);
};

class MLIRGenImpl {
public:
  MLIRGenImpl() = delete;
  MLIRGenImpl(mlir::OpBuilder &builder,
              llvm::MapVector<llvm::StringRef, mlir::Value> &symbolTable)
      : builder_(builder), symbolTable_(symbolTable) {}

  // Build a funcOp for a definition 'def'
  mlir::func::FuncOp buildFunction(const std::string name,
                                   const lang::Def &def);

  // Build the MLIR representation of a single comprehension.
  // Result: [result.generic, whereToStore]
  std::pair<mlir::Value, mlir::Value>
  buildComprehension(const lang::Comprehension &c);

private:
  mlir::MLIRContext *context_;
  mlir::OpBuilder &builder_;
  llvm::MapVector<llvm::StringRef, mlir::Value> &symbolTable_;

  // Get MLIR type from 'kind'
  mlir::Type getScalarType(int kind);
  // Get MLIR memref type from 'tensorType' with #n dimensions.
  mlir::Type getMemRefType(const lang::TensorType &tensorType, size_t ndims);
  // Get MLIR float type from 'kind'
  mlir::FloatType getFloatType(int kind);

  std::unordered_map<std::string, size_t>
  collectOutputRanks(const lang::Def &def);

  // Build linalg.fill for a given tensor.
  void buildTensorInitialization(const std::string tensorName,
                                 mlir::Value tensor, NeutralElement elem);

  // Builds the core of a comprehension (e.g., just the actual
  // compitation without the initialization broadcasting the neutral
  // element for default-initialized reductions) with affine
  // accesses. The check for affine accesses must be performed prior
  // to the call.
  mlir::Value buildLinalgReductionCore(
      const lang::Comprehension &c, mlir::Value tensor,
      const std::unordered_map<std::string, IteratorKind> &iterators,
      const llvm::SmallVectorImpl<std::string> &iteratorsSeq,
      mlir::Location location);
};

mlir::func::FuncOp
buildMLIRFunction(mlir::OpBuilder &builder,
                  llvm::MapVector<llvm::StringRef, mlir::Value> &symbolTable,
                  const std::string name, const lang::Def &tc);

mlir::transform::NamedSequenceOp buildMLIRTactic(mlir::OpBuilder &builder,
                                                 const std::string name,
                                                 const lang::Tac &tac);

} // namespace teckyl

#endif
