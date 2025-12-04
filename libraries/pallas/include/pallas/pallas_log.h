/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#pragma once
#include <inttypes.h>
#include <stddef.h>
extern __thread uint64_t pallas_thread_rank;
extern unsigned int pallas_mpi_rank;

/** Stops the execution. */
#define pallas_abort() abort()
/** Logs a formated message to the given filedescriptor if the given debugLevel is high enough. */
#define _pallas_log(fd, _debug_level_, format, ...)                           \
  do {                                                                        \
    if (C_CXX(pallas_debug_level_get(), pallas::debugLevel) >= _debug_level_) \
      fprintf(fd, "[P%dT%" PRIu64 "] " format, pallas_mpi_rank, pallas_thread_rank, ##__VA_ARGS__);        \
  } while (0)
/** Logs a formated message to stdout if the given debugLevel is high enough. */
#define pallas_log(_debug_level_, format, ...) _pallas_log(stdout, _debug_level_, format, ##__VA_ARGS__)
/** Logs a formated message to stderr if the debugLevel is under Normal. */
#define pallas_warn(format, ...)                                                                                      \
  do {                                                                                                                \
    _pallas_log(stderr, C_CXX(Normal, pallas::DebugLevel::Normal), "Pallas warning in %s (%s:%d): " format, __func__, \
                __FILE__, __LINE__, ##__VA_ARGS__);                                                                   \
  } while (0)
/** Logs a formated message to stderr if the given debugLevel is under Error. */
#define pallas_error(format, ...)                                                                                 \
  do {                                                                                                            \
    _pallas_log(stderr, C_CXX(Error, pallas::DebugLevel::Error), "Pallas error in %s (%s:%d): " format, __func__, \
                __FILE__, __LINE__, ##__VA_ARGS__);                                                               \
    pallas_abort();                                                                                               \
  } while (0)

/** Asserts a condition whatever the build mode (ie. Debug or Release). */
#define pallas_assert_always(cond)      \
  do {                                  \
    if (!(cond))                        \
      pallas_error("Assertion failed"); \
  } while (0)

/** Asserts an equality whatever the build mode (ie. Debug or Release). */
#define pallas_assert_equals_always(a, b)                                                 \
    do {                                                                                  \
        if (a != b)                                                                       \
            pallas_error("Equality failed: %s=%lu != %s=%lu\n", #a, (size_t)a, #b, (size_t)b); \
    } while (0)

/** Asserts a < b whatever the build mode (ie. Debug or Release). */
#define pallas_assert_inferior_always(a, b)                                                 \
do {                                                                                  \
if (a >= b)                                                                       \
pallas_error("Inequality failed: %s=%lu >= %s=%lu\n", #a, (size_t)a, #b, (size_t)b); \
} while (0)

/** Asserts a <= b whatever the build mode (ie. Debug or Release). */
#define pallas_assert_inferior_equal_always(a, b)                                                 \
do {                                                                                  \
if (a > b)                                                                       \
pallas_error("Inequality failed: %s=%lu > %s=%lu\n", #a, (size_t)a, #b, (size_t)b); \
} while (0)


#ifdef NDEBUG
/** Asserts a condition only if in Debug mode (if DEBUG is defined). */
#define pallas_assert(cond)
/** Checks if a == b */
#define pallas_assert_equals(a,b)
/** Checks if a < b */
#define pallas_assert_inferior(a, b)
/** Checks if a <= b */
#define pallas_assert_inferior_equal(a, b)
#else
/** Asserts a condition only if in Debug mode (if DEBUG is defined). */
#define pallas_assert(cond) pallas_assert_always(cond)
/** Checks if a == b */
#define pallas_assert_equals(a,b) pallas_assert_equals_always(a,b)
/** Checks if a < b */
#define pallas_assert_inferior(a, b) pallas_assert_inferior_always(a, b)
/** Checks if a <= b */
#define pallas_assert_inferior_equal(a, b) pallas_assert_inferior_equal_always(a, b)
#endif


/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
