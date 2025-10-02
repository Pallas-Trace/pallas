//
// Created by khatharsis on 17/04/24.
//

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_dbg.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_parameter_handler.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"
#include "pallas/pallas_write.h"

using namespace pallas;
std::vector compressionValues = {
        CompressionAlgorithm::None,
        CompressionAlgorithm::ZSTD,
        CompressionAlgorithm::Histogram,
#ifdef WITH_SZ
        CompressionAlgorithm::SZ,
#endif
#ifdef WITH_ZFP
        CompressionAlgorithm::ZFP,
#endif
        CompressionAlgorithm::ZSTD_Histogram,
};

void usage() {
    std::cout << "Usage: pallas_editor [OPTION] trace_file" << std::endl;
    std::cout << "\t-c, --compression: Changes the compression from the trace to the new one. Possible values:" << std::endl;
    for (auto v : compressionValues) {
        std::cout << "\t\t - " << toString(v) << std::endl;
    }
}

int main(int argc, char** argv) {
    int nb_opts = 0;
    char* trace_name = nullptr;
    auto compressionAlgorithm = CompressionAlgorithm::Invalid;
    auto encodingAlgorithm = EncodingAlgorithm::Invalid;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-v")) {
            pallas_debug_level_set(DebugLevel::Verbose);
            nb_opts++;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?")) {
            usage();
            return EXIT_SUCCESS;
        } else if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "--compression")) {
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

    auto* trace = pallas_open_trace(trace_name);
    if (trace == nullptr) {
        return -1;
    }
    auto* parameter_handler = trace->parameter_handler;
    if (compressionAlgorithm != CompressionAlgorithm::Invalid) {
        ParameterHandler new_parameter_handler = *parameter_handler;
        new_parameter_handler.compressionAlgorithm = compressionAlgorithm;
        auto newDirName = strdup((std::string(trace->dir_name) + "_" + toString(compressionAlgorithm)).c_str());
        for (auto& lg : trace->location_groups) {
            std::cout << "Reading archive " << lg.id << " @ " << trace->dir_name << std::endl;
            auto* a = trace->getArchive(lg.id);
            for (auto& loc : a->locations) {
                std::cout << "\tReading thread " << loc.id << " @ " << a->dir_name << std::endl;
                auto* t = a->getThread(loc.id);
                std::cout << "\tCompressing thread " << t->id << " @ " << a->dir_name << std::endl;
                t->store(newDirName, &new_parameter_handler, true);
                a->freeThread(loc.id);
            }
            std::cout << "Writing archive " << lg.id << " @ " << trace->dir_name << std::endl;
            a->store(newDirName, &new_parameter_handler);
            a->dir_name = nullptr;
            trace->freeArchive(lg.id);
        }
        trace->store(newDirName, &new_parameter_handler);
    }
    delete trace;
    return 0;
}