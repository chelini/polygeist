#include "mlirAffineExprGen.h"

namespace teckyl {

// There re no subtraction expressions for AffineExpr; emulate by
// creating an addition with -1 as a factor for the second operand.
mlir::AffineExpr
MLIRAffineExprGen::buildAffineSubtraction(const lang::TreeRef &t) {
  if (t->trees().size() != 2)
    llvm::report_fatal_error(
        "Subtraction expression with an operator count != 2");

  mlir::AffineExpr lhs = buildAffineExpression(t->tree(0));
  mlir::AffineExpr rhsSub = buildAffineExpression(t->tree(1));
  mlir::AffineExpr minusOne = mlir::getAffineConstantExpr(-1, context_);
  mlir::AffineExpr rhs =
      mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Mul, minusOne, rhsSub);

  return mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Add, lhs, rhs);
}

mlir::AffineExpr
MLIRAffineExprGen::buildAffineBinaryExpression(const lang::TreeRef &t,
                                               mlir::AffineExprKind kind) {
  if (t->trees().size() != 2)
    llvm::report_fatal_error("Binary expression with an operator count != 2");

  mlir::AffineExpr lhs = buildAffineExpression(t->tree(0));
  mlir::AffineExpr rhs = buildAffineExpression(t->tree(1));
  return mlir::getAffineBinaryOpExpr(kind, lhs, rhs);
}

mlir::AffineExpr
MLIRAffineExprGen::buildAffineExpression(const lang::TreeRef &t) {
  switch (t->kind()) {
  case lang::TK_IDENT: {
    lang::Ident ident(t);
    unsigned int iterDimIdx = iteratorDims_.at(ident.name());
    return mlir::getAffineDimExpr(iterDimIdx, context_);
  }
  case lang::TK_CONST: {
    lang::Const cst(t);
    int tKind = cst.type()->kind();

    if (!isIntType(tKind))
      llvm_unreachable("Constant is not an integer");

    // FIXME: AffineExpr uses *signed* 64-bit integers for
    // constants, so the *unsigned* constants from TC cannot
    // necessarily be respresented correctly. Bail out if the TC
    // constant is too big.
    if (tKind == lang::TK_UINT64) {
      uint64_t uintval = cst.value<uint64_t>();

      if (uintval > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        llvm::report_fatal_error("Unsigned integer constant too big");
    }

    return mlir::getAffineConstantExpr(cst.value<int64_t>(), context_);
  }
  case '+':
    return buildAffineBinaryExpression(t, mlir::AffineExprKind::Add);
  case '-':
    return buildAffineSubtraction(t);
  case '*':
    return buildAffineBinaryExpression(t, mlir::AffineExprKind::Mul);
  case '/':
    return buildAffineBinaryExpression(t, mlir::AffineExprKind::FloorDiv);
  default:
    llvm_unreachable("Unsupported operator for affine expression");
  }
}

llvm::SmallVector<mlir::AffineExpr, 8>
MLIRAffineExprGen::buildAffineExpressions(
    const lang::ListView<lang::Ident> &idents) {
  llvm::SmallVector<mlir::AffineExpr, 8> res;
  for (const lang::Ident &ident : idents)
    res.push_back(buildAffineExpression(ident));
  return res;
}

llvm::SmallVector<mlir::AffineExpr, 8>
MLIRAffineExprGen::buildAffineExpressions(
    const lang::ListView<lang::TreeRef> &access) {
  llvm::SmallVector<mlir::AffineExpr, 8> res;
  for (const lang::TreeRef &idxExpr : access)
    res.push_back(buildAffineExpression(idxExpr));
  return res;
}

} // end namespace teckyl
