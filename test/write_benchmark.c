/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_write.h"
#include "pallas/pallas_record.h"


static struct GlobalArchive* trace = NULL;
static struct Archive* archive = NULL;
static LocationGroupId process_id;
static StringRef process_name;

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

ThreadWriter** thread_writers;
static RegionRef* regions;
static char** region_names;
static StringRef* strings;

static pthread_barrier_t thread_ready;
static pthread_barrier_t bench_start;
static pthread_barrier_t bench_stop;

#define TIME_DIFF(t1, t2) (((t2).tv_sec - (t1).tv_sec) + ((t2).tv_nsec - (t1).tv_nsec) / 1e9)

static _Atomic StringRef string_ref = 0;
static _Atomic LocationGroupId location_group_id = 0;
static _Atomic ThreadId thread_id = 0;

static StringRef _register_string(char* str) {
  StringRef ref = string_ref++;
  pallas_global_archive_register_string(trace, ref, str);
  return ref;
}

static ThreadId _new_thread(void) {
  ThreadId id = thread_id++;
  return id;
}

static pallas_timestamp_t get_timestamp(void) {
  pallas_timestamp_t res = PALLAS_TIMESTAMP_INVALID;

  if(use_logical_clock) {
      static _Thread_local int next_ts = 1;
      res = next_ts++;
  }
  return res;
}

void* worker(void* arg __attribute__((unused))) {
  ThreadId thread_id = _new_thread();
  char thread_name[20];
  snprintf(thread_name, 20, "thread_%u", thread_id);
  StringRef thread_name_id = _register_string(thread_name);

  pallas_archive_define_location(archive, thread_id, thread_name_id, process_id);

  thread_writers[thread_id] = pallas_thread_writer_new(archive, thread_id);
  struct ThreadWriter* thread_writer = thread_writers[thread_id];
  pthread_barrier_wait(&thread_ready);

  struct timespec t1, t2;
  pthread_barrier_wait(&bench_start);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  for (int i = 0; i < nb_iter; i++) {
    switch (pattern) {
    case 0:
      /* each iteration is a sequence of functions:
       * E_f1 L_f1 E_f2 L_f2 E_f3 L_f3 ...
       */
      for (int j = 0; j < nb_functions; j++) {
        pallas_record_generic(thread_writer, NULL, get_timestamp(), strings[j]);
      }
      break;

    case 1:
      /* each iteration contains recursive function calls::
       * E_f1 E_f2 E_f3 ... L_f3 L_f2 L_f1
       */
      for (int j = 0; j < nb_functions; j++) {
        pallas_record_enter(thread_writer, NULL, get_timestamp(), regions[j]);
      }
      for (int j = nb_functions - 1; j >= 0; j--) {
        pallas_record_leave(thread_writer, NULL, get_timestamp(), regions[j]);
      }
      break;
    default:
      fprintf(stderr, "invalid pattern: %d\n", pattern);
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &t2);

  pthread_barrier_wait(&bench_stop);

  double duration = TIME_DIFF(t1, t2);
  int nb_event_per_iter = 2 * nb_functions;
  int nb_events = nb_iter * nb_event_per_iter;
  double duration_per_event = duration / nb_events;

  printf("T#%u: %d events in %lf s -> %lf ns per event\n",
  thread_id, nb_events, duration, duration_per_event * 1e9);

  pallas_thread_writer_close(thread_writer);
  return NULL;
}

void usage(const char* prog_name) {
  printf("Usage: %s [OPTION] program [arg1 arg2 ...]\n", prog_name);
  printf("\t-n X    Set the number of iterations (default: %d)\n", nb_iter_default);
  printf("\t-f X    Set the number of functions (default: %d)\n", nb_functions_default);
  printf("\t-t X    Set the number of threads (default: %d)\n", nb_threads_default);
  printf("\t-p X    Select the event pattern\n");
  printf("\t-l      Use a per-thread logical clock instead of the default clock (default: %d)\n", use_logical_clock_default);

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
      nb_iter = atoi(argv[i + 1]);
      nb_opts += 2;
      i++;
    } else if (!strcmp(argv[i], "-f")) {
      nb_functions = atoi(argv[i + 1]);
      nb_opts += 2;
      i++;
    } else if (!strcmp(argv[i], "-t")) {
      nb_threads = atoi(argv[i + 1]);
      nb_opts += 2;
      i++;
    } else if (!strcmp(argv[i], "-p")) {
      pattern = atoi(argv[i + 1]);
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

  pthread_t tid[nb_threads];
  pthread_barrier_init(&thread_ready, NULL, 2);
  pthread_barrier_init(&bench_start, NULL, nb_threads + 1);
  pthread_barrier_init(&bench_stop, NULL, nb_threads + 1);
  thread_writers = calloc(nb_threads, sizeof(ThreadWriter*));

  printf("nb_iter = %d\n", nb_iter);
  printf("nb_functions = %d\n", nb_functions);
  printf("nb_threads = %d\n", nb_threads);
  printf("pattern = %d\n", pattern);
  printf("---------------------\n");

  trace = pallas_global_archive_new("write_benchmark_trace", "main");
  archive = pallas_archive_new("write_benchmark_trace", 0);
    archive->global_archive = trace;
  process_id = 0; // main process
  process_name = _register_string("Process");


  pallas_global_archive_define_location_group(trace, process_id, process_name, PALLAS_LOCATION_GROUP_ID_INVALID);

  regions = malloc(sizeof(RegionRef) * nb_functions);
  strings = malloc(sizeof(StringRef) * nb_functions);
  region_names = malloc(sizeof(char*) * nb_functions);
  for (int i = 0; i < nb_functions; i++) {
    region_names[i] = malloc(sizeof(char)*50);
    snprintf(region_names[i], 50, "function_%d", i);
    strings[i] = _register_string(region_names[i]);
    regions[i] = strings[i];
    pallas_global_archive_register_region(trace, regions[i], strings[i]);
  }

  for (int i = 0; i < nb_threads; i++) {
    pthread_create(&tid[i], NULL, worker, NULL);
    /* make sure the thread is fully initialized before creating the next one */
    pthread_barrier_wait(&thread_ready);
  }

  struct timespec t1, t2;
  pthread_barrier_wait(&bench_start);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  pthread_barrier_wait(&bench_stop);
  clock_gettime(CLOCK_MONOTONIC, &t2);

  for (int i = 0; i < nb_threads; i++)
    pthread_join(tid[i], NULL);

  double duration = TIME_DIFF(t1, t2);
  int nb_event_per_iter = 2 * nb_functions;
  int nb_events = nb_iter * nb_event_per_iter * nb_threads;
  double events_per_second = nb_events / duration;

  printf("TOTAL: %d events in %lf s -> %lf Me/s \n",
  nb_events, duration, events_per_second / 1e6);

  pallas_archive_close(archive);
  pallas_global_archive_close(trace);
  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
