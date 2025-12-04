/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <mpi.h>
#include <pallas/pallas_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_record.h"
#include "pallas/pallas_write.h"
#include "pallas/pthread_barrier_wrapper.h"

static GlobalArchive * trace = NULL;
static Archive * archive = NULL;
static LocationGroupId process_id;

static int nb_iter_default = 2000;
static int nb_functions_default = 2;
static int nb_threads_default = 4;
static int pattern_default = 0;

static int nb_iter;
static int nb_functions;
static int nb_threads;
static int pattern;

ThreadWriter** thread_writers = NULL;
static RegionRef* regions = NULL;
static StringRef* strings = NULL;

static pthread_barrier_t bench_start;
static pthread_barrier_t bench_stop;

#define TIME_DIFF(t1, t2) (((t2).tv_sec - (t1).tv_sec) + ((t2).tv_nsec - (t1).tv_nsec) / 1e9)

static RegionRef next_region_ref = 0;
static StringRef next_string_ref = 0;
_Atomic ThreadId next_thread_id = 0;
static ThreadId base_thread_id = 0;

static StringRef _register_string_global(char* str) {
  StringRef ref = next_string_ref++;
  pallas_global_archive_register_string(trace, ref, str);
  return ref;
}

static StringRef _register_string_local(char* str) {
  StringRef ref = next_string_ref++;
  pallas_archive_register_string(archive, ref, str);
  return ref;
}

static RegionRef _register_region(StringRef string_ref) {
  RegionRef id = next_region_ref++;
  pallas_archive_register_region(archive, id, string_ref);
  return id;
}

static ThreadId _new_thread() {
  ThreadId id = next_thread_id++;
  return id;
}

static pallas_timestamp_t get_timestamp(void) {
    pallas_timestamp_t res = PALLAS_TIMESTAMP_INVALID;
    // if(use_logical_clock) {
    //     static _Thread_local int next_ts = 1;
    //     res = next_ts++;
    // }
    return res;
}


static int mpi_rank;
static int mpi_comm_size;

void* worker(void* arg __attribute__((unused))) {
  ThreadId local_thread_id = _new_thread();
  pallas_thread_rank = local_thread_id;
  ThreadId global_thread_id = local_thread_id + base_thread_id;
  ThreadWriter* thread_writer =  pallas_thread_writer_new(archive, global_thread_id);;
  thread_writers[local_thread_id] = thread_writer;

  char thread_name[20];
  snprintf(thread_name, 20, "P#%dT#%d", mpi_rank, global_thread_id);
  StringRef thread_name_id = _register_string_local(thread_name);
  pallas_archive_define_location(archive, global_thread_id, thread_name_id, process_id);

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
        pallas_record_enter(thread_writer, NULL, PALLAS_TIMESTAMP_INVALID, regions[j]);
      }
      for (int j = nb_functions - 1; j >= 0; j--) {
        pallas_record_leave(thread_writer, NULL, PALLAS_TIMESTAMP_INVALID, regions[j]);
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

  pallas_log(Normal, "%d events in %lf s -> %lf ns per event\n", nb_events, duration, duration_per_event * 1e9);

  pallas_thread_writer_close(thread_writer);
  return NULL;
}

void usage(const char* prog_name) {
  printf("Usage: %s [OPTION] program [arg1 arg2 ...]\n", prog_name);
  printf("\t-n X    Set the number of iterations (default: %d)\n", nb_iter_default);
  printf("\t-f X    Set the number of functions (default: %d)\n", nb_functions_default);
  printf("\t-t X    Set the number of threads (default: %d)\n", nb_threads_default);
  printf("\t-p X    Select the event pattern\n");
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
    } else if (!strcmp(argv[i], "-?") || !strcmp(argv[i], "-h")) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      fprintf(stderr, "invalid option: %s\n", argv[i]);
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
  pallas_mpi_rank = mpi_rank;

  if (mpi_rank == 0) {
    trace = pallas_global_archive_new("mpi_benchmark_trace", "main");
  }

  printf("Hello from %d/%d\n", mpi_rank, mpi_comm_size);

  process_id = mpi_rank;

  int chunk_size = mpi_comm_size * 100;
  next_region_ref = mpi_rank * chunk_size;
  next_string_ref = mpi_rank * chunk_size;
  base_thread_id = mpi_rank * chunk_size;

  pthread_t tid[nb_threads];
  pthread_barrier_init(&bench_start, NULL, nb_threads + 1);
  pthread_barrier_init(&bench_stop, NULL, nb_threads + 1);
  thread_writers = malloc(sizeof(ThreadWriter*) * nb_threads);

  printf("nb_iter = %d\n", nb_iter);
  printf("nb_functions = %d\n", nb_functions);
  printf("nb_threads = %d\n", nb_threads);
  printf("pattern = %d\n", pattern);
  printf("---------------------\n");

  archive = pallas_archive_new("mpi_benchmark_trace", mpi_rank);
  if (mpi_rank == 0) {
      archive->global_archive = trace;
    for (int i = 0; i < mpi_comm_size; i++) {
      char rank_name_str[100];
      snprintf(rank_name_str, 100, "Rank#%d", i);
      StringRef rank_name = _register_string_global(rank_name_str);
      pallas_global_archive_define_location_group(trace, i, rank_name, PALLAS_LOCATION_GROUP_ID_INVALID);
    }
  }
  regions = malloc(sizeof(RegionRef) * nb_functions);
  strings = malloc(sizeof(StringRef) * nb_functions);
  for (int i = 0; i < nb_functions; i++) {
    char str[50];
    snprintf(str, 50, "function_%d", i);
    strings[i] = _register_string_local(str);
    regions[i] = _register_region(strings[i]);
  }


  for (int i = 0; i < nb_threads; i++)
    pthread_create(&tid[i], NULL, worker, NULL);

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

  pallas_log(Normal, "TOTAL: %d events in %lf s -> %lf Me/s \n", nb_events, duration, events_per_second / 1e6);

  pallas_archive_close(archive);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  if (mpi_rank == 0) {
    pallas_global_archive_close(trace);
  }
  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
