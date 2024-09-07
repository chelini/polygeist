#ifndef MLIR_TOOLS_MLIRCLANG_LIB_PRAGMASYNTAXHANDLER_H
#define MLIR_TOOLS_MLIRCLANG_LIB_PRAGMASYNTAXHANDLER_H

#include <string>

namespace linalg_plugin {

struct LinalgPlugin {
  std::string funcName;
  std::string body;
};

} // end namespace linalg_plugin

void addSyntaxHandlers();

#endif
