#ifndef TECKYL_MLIRAFFINEEXPRGEN_H
#define TECKYL_MLIRAFFINEEXPRGEN_H

#include "mlir/IR/AffineExpr.h"
#include "tree.h"
#include "tree_views.h"
#include <unordered_map>

namespace teckyl {

static bool inline isSignedIntType(int kind) {
  switch (kind) {
  case lang::TK_INT2:
  case lang::TK_INT4:
  case lang::TK_INT8:
  case lang::TK_INT16:
  case lang::TK_INT32:
  case lang::TK_INT64:
    return true;

  default:
    return false;
  }
}

static inline bool isUnsignedIntType(int kind) {
  switch (kind) {
  case lang::TK_UINT2:
  case lang::TK_UINT4:
  case lang::TK_UINT8:
  case lang::TK_UINT16:
  case lang::TK_UINT32:
  case lang::TK_UINT64:
    return true;

  default:
    return false;
  }
}

static inline bool isIntType(int kind) {
  return isSignedIntType(kind) || isUnsignedIntType(kind);
}

class MLIRAffineExprGen {
public:
  MLIRAffineExprGen() = delete;
  MLIRAffineExprGen(mlir::MLIRContext *context,
                    const std::unordered_map<std::string, size_t> &iteratorDims)
      : context_(context), iteratorDims_(iteratorDims) {}

  // Builds an AffineExpr for each of the argument of 'access' and
  // returns the result in a vector.
  llvm::SmallVector<mlir::AffineExpr, 8>
  buildAffineExpressions(const lang::ListView<lang::TreeRef> &access);

  // Builds an AffineExpr for each of the identifiers and returns the
  // result in a vector.
  llvm::SmallVector<mlir::AffineExpr, 8>
  buildAffineExpressions(const lang::ListView<lang::Ident> &idents);

private:
  mlir::MLIRContext *context_;
  const std::unordered_map<std::string, size_t> &iteratorDims_;

  mlir::AffineExpr buildAffineExpression(const lang::TreeRef &t);
  mlir::AffineExpr buildAffineBinaryExpression(const lang::TreeRef &t,
                                               mlir::AffineExprKind kind);
  mlir::AffineExpr buildAffineSubtraction(const lang::TreeRef &t);
};

} // end namespace teckyl

#endif
