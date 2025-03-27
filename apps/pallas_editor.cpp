//
// Created by khatharsis on 17/04/24.
//

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_dbg.h"
#include "pallas/pallas_parameter_handler.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"
#include "pallas/pallas_write.h"
#include "pallas/pallas_log.h"

using namespace pallas;

std::string test() {
  return "test";
}

void usage() {
  std::cout << "Usage: pallas_editor [OPTION] trace_file" << std::endl;
  //  std::cout << "\t-f [name]: Filter out any event matching that name" << std::endl;
  std::cout << "\t-c [compression]: Changes the compression from the trace to the new one." << std::endl;
}

int main(int argc, char** argv) {
  int nb_opts = 0;
  char* trace_name = nullptr;
  CompressionAlgorithm compressionAlgorithm = pallas::CompressionAlgorithm::Invalid;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-v")) {
      pallas_debug_level_set(DebugLevel::Debug);
      nb_opts++;
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?")) {
      usage();
      return EXIT_SUCCESS;
    } else if (!strcmp(argv[i], "-c")) {
      nb_opts += 2;
      compressionAlgorithm = compressionAlgorithmFromString(argv[++i]);
      break;
    } else {
      /* Unknown parameter name. It's probably the program name. We can stop
       * parsing the parameter list.
       */
      break;
    }
  }

  trace_name = argv[nb_opts + 1];
  if (trace_name == nullptr) {
    usage();
    return EXIT_SUCCESS;
  }

  auto& trace = *pallas_open_trace(trace_name);
  if (compressionAlgorithm != pallas::CompressionAlgorithm::Invalid &&
      compressionAlgorithm != parameterHandler->getCompressionAlgorithm()) {
    char* newDirName = new char[strlen(trace.dir_name) + 10];
    sprintf(newDirName, "%s_%s", trace.dir_name, toString(compressionAlgorithm).c_str());
    trace.dir_name = newDirName;
    trace.trace_name[strlen(trace.trace_name) - strlen(".pallas")] = '\0';
    DOFOR(i, trace.nb_archives) {
      trace.archive_list[i]->dir_name = newDirName;
    }
    auto originalCompressionAlgorithm = parameterHandler->compressionAlgorithm;
//    DOFOR(i, trace.nb_threads) {
//      std::cout << "Reading thread " << i << std::endl;
//      parameterHandler->compressionAlgorithm = originalCompressionAlgorithm;
//      trace.threads[i]->loadTimestamps();
//      std::cout << "Compressing thread " << i << std::endl;
//      parameterHandler->compressionAlgorithm = compressionAlgorithm;
//      trace.threads[i]->finalizeThread();
//    }
  }

  DOFOR(i, trace.nb_archives) {
    trace.archive_list[i]->close();
  }
  trace.close();
}