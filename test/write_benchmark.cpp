/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <sstream>
#if __cplusplus >= 202002L && (__GNUC__ >= 13 || __clang__ >= 14 || _MSC_VER >= 1929)
#include <format>
#define HAS_FORMAT
#endif

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_record.h"
#include "pallas/pallas_write.h"
#include "pallas/pallas_log.h"

using namespace pallas;
static LocationGroupId processID;
static StringRef processName;

static int nb_iter_default = 2000;
static int nb_functions_default = 2;
static int nb_threads_default = 4;
static int pattern_default = 0;
static int use_logical_clock_default = 0;

static int nb_iter;
static int nb_functions;
static int nb_threads;
static int pattern;
static int use_logical_clock;

std::vector<ThreadWriter> thread_writers;
std::vector<RegionRef> regions;
std::vector<std::string> region_names;
std::vector<StringRef> strings;

static pthread_barrier_t bench_start;
static pthread_barrier_t bench_stop;

#define TIME_DIFF(t1, t2) (((t2).tv_sec - (t1).tv_sec) + ((t2).tv_nsec - (t1).tv_nsec) / 1e9)

static StringRef registerString(GlobalArchive& trace, const std::string& str) {
  static std::atomic<StringRef> next_ref = 0;
  StringRef ref = next_ref++;
  trace.addString(ref, str.c_str());
  return ref;
}

static ThreadId newThread() {
  static std::atomic<ThreadId> next_id = 0;
  ThreadId id = next_id++;
  return id;
}

static pallas_timestamp_t get_timestamp() {
  pallas_timestamp_t res = PALLAS_TIMESTAMP_INVALID;

  if (use_logical_clock) {
    static int next_ts = 1;
    res = next_ts++;
  }
  return res;
}

struct example_data {
    std::string name;
    size_t random_number;
};

size_t write_example_data(example_data* d, FILE* file) {
    return fwrite(d, sizeof(example_data), 1, file);
}


void* worker(void* arg) {
  ThreadId threadID = newThread();
  auto& archive = *static_cast<Archive*>(arg);
  //auto threadWriter = ThreadWriter();

#ifdef HAS_FORMAT
  StringRef threadNameRef = registerString(std::format("thread_{}", threadID));
#else
  std::ostringstream os;
    os << "thread_" << threadID;
  StringRef threadNameRef = registerString(*archive.global_archive, os.str());
#endif
  archive.defineLocation(threadID, threadNameRef, processID);
  example_data data{"ThreadTest", 42};
  AdditionalContent<example_data> content{&data, &write_example_data};
  archive.add_content(&content);

  ThreadWriter threadWriter(archive, threadID);

  pthread_barrier_wait(&bench_start);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < nb_iter; i++) {
    switch (pattern) {
    case 0:
      /* each iteration is a sequence of functions:
       * E_f1 L_f1 E_f2 L_f2 E_f3 L_f3 ...
       */
      for (int j = 0; j < nb_functions; j++) {
        pallas_record_generic(&threadWriter, nullptr, get_timestamp(), strings[j]);
      }
      break;

    case 1:
      /* each iteration contains recursive function calls::
       * E_f1 E_f2 E_f3 ... L_f3 L_f2 L_f1
       */
      for (int j = 0; j < nb_functions; j++) {
        pallas_record_enter(&threadWriter, nullptr, get_timestamp(), regions[j]);
      }
      for (int j = nb_functions - 1; j >= 0; j--) {
        pallas_record_leave(&threadWriter, nullptr, get_timestamp(), regions[j]);
      }
      break;
    default:
      fprintf(stderr, "invalid pattern: %d\n", pattern);
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();

  pthread_barrier_wait(&bench_stop);

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  int nb_event_per_iter = 2 * nb_functions;
  int nb_events = nb_iter * nb_event_per_iter;
  auto duration_per_event = duration / nb_events;

    pthread_mutex_lock(&archive.lock);
 std::cout << "T#" << threadID << ": "<< nb_events << " events in " << duration/1e9<< "s -> "<< duration_per_event << " ns per event" << std::endl;
    pthread_mutex_unlock(&archive.lock);
  threadWriter.threadClose();
  return nullptr;
}

void usage(const char* prog_name) {
  printf("Usage: %s [OPTION] program [arg1 arg2 ...]\n", prog_name);
  printf("\t-n X    Set the number of iterations (default: %d)\n", nb_iter_default);
  printf("\t-f X    Set the number of functions (default: %d)\n", nb_functions_default);
  printf("\t-t X    Set the number of threads (default: %d)\n", nb_threads_default);
  printf("\t-p X    Select the event pattern\n");
  printf("\t-l      Use a per-thread logical clock instead of the default clock (default: %d)\n",
         use_logical_clock_default);

  printf("\t-? -h   Display this help and exit\n");
}

int main(int argc, char** argv) {
  int nb_opts = 0;  // options

  nb_iter = nb_iter_default;
  nb_functions = nb_functions_default;
  nb_threads = nb_threads_default;
  pattern = pattern_default;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-n")) {
      nb_iter = std::atoi(argv[i + 1]);
      nb_opts += 2;
      i++;
    } else if (!strcmp(argv[i], "-f")) {
      nb_functions = std::atoi(argv[i + 1]);
      nb_opts += 2;
      i++;
    } else if (!strcmp(argv[i], "-t")) {
      nb_threads = std::atoi(argv[i + 1]);
      nb_opts += 2;
      i++;
    } else if (!strcmp(argv[i], "-p")) {
      pattern = std::atoi(argv[i + 1]);
      nb_opts += 2;
      i++;
    } else if (!strcmp(argv[i], "-l")) {
      use_logical_clock = 1;
      nb_opts += 1;
    } else if (!strcmp(argv[i], "-?") || !strcmp(argv[i], "-h")) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      fprintf(stderr, "invalid option: %s\n", argv[i]);
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  pthread_barrier_init(&bench_start, nullptr, nb_threads + 1);
  pthread_barrier_init(&bench_stop, nullptr, nb_threads + 1);

  struct separate_thousands : std::numpunct<char> {
    char_type do_thousands_sep() const override { return ' '; }  // separate with commas
    string_type do_grouping() const override { return "\3"; }    // groups of 3 digit
  };
  auto thousands = std::make_unique<separate_thousands>();
  std::cout.imbue(std::locale(std::cout.getloc(), thousands.release()));

  std::cout << "nb_iter = " << nb_iter << std::endl
            << "nb_functions = " << nb_functions << std::endl
            << "nb_threads = " << nb_threads << std::endl
            << "pattern = " << pattern << std::endl
            << "---------------------" << std::endl;

  GlobalArchive globalArchive("write_benchmark_CPP_trace", "main");

  processID = 0;
  processName = registerString(globalArchive, "Main process");
  globalArchive.defineLocationGroup(processID, processName, processID);
  Archive mainProcess(globalArchive, 0);
    mainProcess.global_archive = &globalArchive;

  for (int i = 0; i < nb_functions; i++) {
    std::ostringstream os;
    os << "function_" << i;
    region_names.push_back(os.str());
    os.clear();
    strings.push_back(registerString(globalArchive, region_names.back()));
    regions.push_back(strings.back());
    globalArchive.addRegion(regions.back(), strings.back());
  }

  std::vector<pthread_t> threadID;
  for (int i = 0; i < nb_threads; i++) {
    pthread_t tid;
    pthread_create(&tid, nullptr, worker, &mainProcess);
    threadID.push_back(tid);
  }

  pthread_barrier_wait(&bench_start);

  auto start = std::chrono::high_resolution_clock::now();
  pthread_barrier_wait(&bench_stop);
  auto end = std::chrono::high_resolution_clock ::now();

  for (int i = 0; i < nb_threads; i++)
    pthread_join(threadID[i], nullptr);

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  int nb_event_per_iter = 2 * nb_functions;
  int nb_events = nb_iter * nb_event_per_iter * nb_threads;
  auto events_per_second = nb_events / duration;

  std::cout << "TOTAL: " << nb_events << " events in " << duration << " s -> " << events_per_second / 1e6 << " Me/s"
            << std::endl;

  mainProcess.close();
  globalArchive.close();
  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
