//
// Created by khatharsis on 11/12/25.
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <sstream>
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_record.h"
#include "pallas/pallas_write.h"
#include "pallas/pallas_log.h"
using namespace pallas;

std::vector<RegionRef> regions;
std::vector<std::string> region_names;
std::vector<StringRef> strings;

static StringRef registerString(GlobalArchive& trace, const std::string& str) {
    static std::atomic<StringRef> next_ref = 0;
    StringRef ref = next_ref++;
    trace.addString(ref, str.c_str());
    return ref;
}

static pallas_timestamp_t getTimestamp() {
    static size_t curTimestamp = 0;
    return curTimestamp++;
}

static void logFunction(ThreadWriter* writer, int functionNumber) {
    pallas_record_enter(writer, nullptr, getTimestamp(), regions[functionNumber]);
    pallas_record_leave(writer, nullptr, getTimestamp(), regions[functionNumber]);
}


int main() {
    pallas_debug_level_set(DebugLevel::Debug);
    GlobalArchive globalArchive("write_pattern_trace", "main");
    size_t processID = 0;
    StringRef processName = registerString(globalArchive, "Main process");
    globalArchive.defineLocationGroup(processID, processName, processID);
    Archive mainProcess(globalArchive, 0);
    mainProcess.global_archive = &globalArchive;

    int nb_functions = 4;
    for (int i = 0; i < nb_functions; i++) {
        std::ostringstream os;
        os << "function_" << i;
        region_names.push_back(os.str());
        os.clear();
        strings.push_back(registerString(globalArchive, region_names.back()));
        regions.push_back(strings.back());
        globalArchive.addRegion(regions.back(), strings.back());
    }
    StringRef threadNameRef = registerString(globalArchive, "main_thread");
    mainProcess.defineLocation(processID, threadNameRef, processID);

    ThreadWriter threadWriter(mainProcess, processID);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            logFunction(&threadWriter, 0);
            for (int loop = 0; loop < 3; loop++) {
                logFunction(&threadWriter, 1);
                logFunction(&threadWriter, 2);
            }
        }
    }
    threadWriter.threadClose();
    mainProcess.store();
    globalArchive.store();
    return 0;
}