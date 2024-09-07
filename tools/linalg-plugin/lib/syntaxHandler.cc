#include "syntaxHandler.h"

#include "mlir/Support/FileUtilities.h"
#include "clang/Parse/Parser.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"

using namespace clang;
using namespace linalg_plugin;

using llvm::yaml::MappingTraits;
using llvm::yaml::Output;

namespace llvm {
namespace yaml {

template <> struct MappingTraits<LinalgPlugin> {
  static void mapping(IO &io, LinalgPlugin &info) {
    io.mapRequired("funcName", info.funcName);
    io.mapRequired("body", info.body);
  }
};

} // namespace yaml
} // namespace llvm

namespace {

// XXX: The llvm registry requires registered subclass to have a default
// constructor.
// But we need to export the information parsed here. For now write them on a
// file.
class PrintTokensHandler : public SyntaxHandler {
public:
  PrintTokensHandler() : SyntaxHandler("linalg") {}

  void GetReplacement(Preprocessor &PP, Declarator &D, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS) override {
    std::string funcName = D.getIdentifier()->getName().str();
    std::string content = "";
    for (auto &Tok : Toks)
      content += PP.getSpelling(Tok);
    content += "\n";
    LinalgPlugin p = {funcName, content};

    std::string errorMessage;
    std::unique_ptr<llvm::ToolOutputFile> outPlugin;
    outPlugin = mlir::openOutputFile("linalg.plugin", &errorMessage);
    if (!outPlugin) {
      llvm::errs() << errorMessage << "\n";
      return;
    }
    Output yout(outPlugin->os());
    yout << p;
    if (outPlugin)
      outPlugin->keep();
  }

  void AddToPredefines(llvm::raw_string_ostream &OS) override {}
};

} // end namespace

void addSyntaxHandlers() {
  static SyntaxHandlerRegistry::Add<PrintTokensHandler> X(
      "linalg", "emit linalg dialect");
}
