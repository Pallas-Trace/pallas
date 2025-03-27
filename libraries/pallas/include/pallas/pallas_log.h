//
// Created by khatharsis on 23/07/24.
//

#pragma once
#include <inttypes.h>
extern __thread size_t thread_rank;
extern unsigned int pallas_mpi_rank;

/** Stops the execution. */
#define pallas_abort() abort()
/** Logs a formated message to the given filedescriptor if the given debugLevel is high enough. */
#define _pallas_log(fd, _debug_level_, format, ...)                           \
  do {                                                                        \
    if (C_CXX(pallas_debug_level_get(), pallas::debugLevel) >= _debug_level_) \
      fprintf(fd, "[P%dT%" PRIu64 "] " format, pallas_mpi_rank, thread_rank, ##__VA_ARGS__);        \
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

#ifdef NDEBUG
/** Asserts a condition only if in Debug mode (if DEBUG is defined). */
#define pallas_assert(cond)
#else
/** Asserts a condition only if in Debug mode (if DEBUG is defined). */
#define pallas_assert(cond) pallas_assert_always(cond)
#endif
