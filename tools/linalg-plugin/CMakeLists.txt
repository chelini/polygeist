include_directories(
  "${LLVM_SOURCE_DIR}/../clang/include"
  "${CMAKE_BINARY_DIR}/tools/clang/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set(LLVM_LINK_COMPONENTS
  Core
  Support
)

add_mlir_library(LinalgPlugin
  lib/lexer.cc
  lib/mlirAffineExprGen.cpp
  lib/mlirGen.cpp
  lib/parser.cc
  lib/syntaxHandler.cc
  lib/tree.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include

  LINK_LIBS PUBLIC

  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSupport
  MLIRTransformDialect

  clangAST
  clangBasic
  clangParse
  clangLex
)

add_llvm_tool(linalg-plugin-opt
  linalg-plugin-opt.cc
)

llvm_update_compile_flags(linalg-plugin-opt)

target_link_libraries(linalg-plugin-opt
  PRIVATE

  MLIROptLib
  LinalgPlugin
)

add_subdirectory(test)
