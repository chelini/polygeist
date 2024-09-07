#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlirGen.h"
#include "parser.h"
#include "sema.h"

#include <fstream>

static llvm::cl::OptionCategory
    toolOptions("clang to mlir plugin for TC - tekyl");

static llvm::cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false), llvm::cl::cat(toolOptions));

static llvm::cl::opt<bool> showAst("show-ast",
                                   llvm::cl::desc("Print the TC AST"),
                                   llvm::cl::init(false),
                                   llvm::cl::cat(toolOptions));

static llvm::cl::opt<bool> showMlir("show-mlir",
                                    llvm::cl::desc("Print the MLIR Module"),
                                    llvm::cl::init(false),
                                    llvm::cl::cat(toolOptions));

static llvm::cl::opt<std::string>
    inputFileName(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                  llvm::cl::init("-"), llvm::cl::value_desc("filename"));

namespace {

// Parses a string with TCs and returns a map with one entry for each
// kernel, composed of the kernel's name and its AST.
std::map<std::string, lang::Def> parse(const std::string tc,
                                       const std::string filename) {
  lang::Parser parser(tc, filename);
  std::map<std::string, lang::Def> parsed;

  while (parser.L.cur().kind != lang::TK_EOF) {
    auto t = parser.parseFunction();
    auto def = lang::Def(t);
    auto name = def.name().name();
    parsed.emplace(std::make_pair(name, def));
  }

  return parsed;
}

// Reads an entire file into a string.
std::string readFile(const std::string filename) {
  std::ifstream ifs(filename);

  if (!ifs.good())
    assert(false && "Could not open file ");

  // look for __start_tc / __end_tc to avoid passing the TC parser garbage.
  std::string line;
  std::string tc;
  bool inScope = false;
  while (getline(ifs, line)) {
    if (line.compare("__end_tc") == 0)
      break;
    if (inScope)
      tc += line + "\n";
    if (line.compare("__start_tc") == 0)
      inScope = true;
  }
  ifs.close();
  return tc;
}

// Dumps the AST for a set of kernels to stdout.
void dumpAST(const std::map<std::string, lang::Def> &tcs) {
  for (const auto &res : tcs)
    std::cout << res.second << std::endl;
}

void dumpMlir(const std::map<std::string, lang::Def> &tcs) {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  // register linalg and tensor.
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  context.disableMultithreading();

  if (showDialects) {
    llvm::outs() << "Registered Dialects:\n";
    for (mlir::Dialect *dialect : context.getLoadedDialects()) {
      llvm::outs() << dialect->getNamespace() << "\n";
    }
    return;
  }

  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(builder.getUnknownLoc()));

  lang::Sema sema;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  for (auto &tc : tcs) {
    lang::TreeRef checked = sema.checkFunction(tc.second);
    mlir::func::FuncOp f = teckyl::buildMLIRFunction(
        &context, builder, symbolTable, tc.first, lang::Def(checked));
    module->push_back(f);
  }

  if (mlir::failed(mlir::verify(*module))) {
    module->dump();
    llvm::report_fatal_error("Module verification failed\n");
  }

  module->print(llvm::outs());
}

} // end anonymous namespace.

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  std::string source = readFile(inputFileName);
  std::map<std::string, lang::Def> tcs;
  tcs = parse(source, inputFileName);

  if (showAst) {
    llvm::outs() << "Printing TC AST\n";
    dumpAST(tcs);
    return 0;
  }

  if (showMlir) {
    dumpMlir(tcs);
    return 0;
  }

  return 0;
}
