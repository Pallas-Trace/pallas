/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Debug macros, debug levels, all that stuff.
 */
#pragma once
#ifndef NDEBUG
/** This macro only exists in Debug mode.*/
#define DEBUG
#endif

#ifdef __cplusplus
/** Only exists in C++. */
#define CXX(cxx_name) cxx_name
/** Only exists in C. */
#define C(c_name)
#else
/** Only exists in C++. */
#define CXX(cxx_name)
/** Only exists in C. */
#define C(c_name) c_name
#endif

/** A macro to help naming conventions in C/C++. First argument is only kept in C, second is only kept in C++. */
#define C_CXX(c_name, cxx_name) C(c_name) CXX(cxx_name)
/** Adds pallas:: in front of the variables in C++. */
#define PALLAS(something) CXX(pallas::) something

#ifdef __cplusplus
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#else
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#endif
#ifdef __cplusplus
namespace pallas {
#endif

/**
 * Enum to handle the different level of logging.
 */
enum CXX(class) DebugLevel {
  Error,   /**< Only print errors. */
  Quiet,   /**< Only print important messages. */
  Normal,  /**< Default verbosity level. */
  Verbose, /**< Print additional information. */
  Debug,   /**< Print much information. */
  Help,    /**< Print the different verbosity level and exit. */
  Max,     /**< Flood stdout with debug messages. */
};
#ifdef __cplusplus
extern enum DebugLevel debugLevel;
}; /* namespace pallas */
extern "C" {
#endif
/** Initializes the DebugLevel using Env Variables. */
extern void pallas_debug_level_init(void);
/** Sets the DebugLevel to the given level. */
extern void pallas_debug_level_set(enum PALLAS(DebugLevel) lvl);
/** Returns the DebugLevel. */
extern enum PALLAS(DebugLevel) pallas_debug_level_get(void);
CXX(
};)



 /**
  * Functions for performance analysis
  */  

#ifdef __cplusplus
extern "C" {
#endif
/* Duration structure */
  typedef struct {
    /* Total duration of the program */
	  double total_d;
    /* Minimal duration of the function */
	  double min_d;
    /* Maximal duration of the function */
	  double max_d;
    /* Number of calls */
	  int count;
    /* size of the details structure */
    int size;
    /* Has size blocs and contains all the detailed data */
    double* durations;
    /* Has size blocs and contains the (sub-)vector sizes for each duration*/
    int* sizes;
  } Duration;

  void write_duration_details(const char* filename, const char* output, const Duration* d);

#ifdef __cplusplus
}
#endif


// //Initialises a duration (all to 0 and DBL_MAX for min)
// void duration_init(Duration* d);

//Updates the duration d with times start and and
void update_duration(Duration* d, struct timespec start, struct timespec end);
void update_durations(Duration* d, struct timespec t1, struct timespec t2, int size);


//writes one duration
void duration_write_csv(const char* filename, const Duration* d);

//All tracked function references
enum FunctionIndex {
  PRINT_TIMESTAMP,
  PRINT_TIMESTAMP_PRECISION,
  PRINT_TIMESTAMP_HEADER,
  PRINT_TIMESTAMP_ELSE,
  PRINT_DURATION,
  PRINT_DURATION_HEADER,
  PRINT_EVENT,
  PRINT_EVENT_PRINT_TIMESTAMP,
  PRINT_EVENT_GET_NAME,
  PRINT_EVENT_GET_TOKEN_STRING,
  PRINT_EVENT_GET_EVENT_STRING,
  PRINT_EVENT_GET_PRINT_EV_ATT,
  PRINT_EV_ATT,
  PRINT_EVENT_ENDL,
  PRINT_FLAME,
  PRINT_CSV,
  PRINT_CSV_BULK,
  PRINT_TRACE,
  PRINT_TRACE_GET_THREAD_LIST,
  GET_THREAD_LIST_GET_ARCHIVE,
  GET_THREAD_LIST_GET_THREAD,
  PRINT_TRACE_EMPLACE_BACK,
  PRINT_TRACE_POLLCURTOKEN,
  PRINT_TRACE_PRINT_EVENT,
  PRINT_TRACE_GET_EV_OCC,
  PRINT_TRACE_GET_NEXT_TOKEN,
  STRUCTURE_INDENT,
  STRUCTURE_INDENT_POLLCURTOKEN,
  PRINT_THREAD_STRUCTURE,
  PRINT_STRUCTURE,
  GET_EVENT_OCC,
  GET_TOKEN_IN_CALL_STACK,
  ENTER_BLOCK,
  PALLAS_PRINT,
  OPEN_TRACE,
  ZSTD,
  ZFP,
  SZ, 
  WRITE,
  COMP,
  WRITE_VECTOR,
  WRITE_DUR_VECT,
  WRITE_SUBVEC,
  WRITE_DUR_SUBVEC,
  ZSTD_HISTOGRAM,
  PCW_ENC_ALG,
  PCW,
  PCW_COMP_ALG,
  PCW_WRITE,
  WRITE_DUR_SUBVEC_FTELL,
  WRITE_DUR_SUBVEC_PCW, 
  WRITE_DUR_SUBVEC_DELETE,
  NB_FUNCTIONS
};


//Table of durations
extern Duration durations[NB_FUNCTIONS];

//Writes all the durations of duration[NB_FUNCTIONS] to a file named filename.csv
void duration_write_all_csv(const char* filename);


void write_csv_details(const char* filename, const char* output, const char* info, struct timespec t1, struct timespec t2);





/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */