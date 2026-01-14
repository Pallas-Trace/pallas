/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstring>

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"

namespace pallas {
enum DebugLevel debugLevel = DebugLevel::Normal;

}

static void print_help() {
  pallas_log(pallas::DebugLevel::Error, "You can set VERBOSE to different values to control the verbosity of Pallas:\n");
  pallas_log(pallas::DebugLevel::Error, "Error:   only print errors\n");
  pallas_log(pallas::DebugLevel::Error, "Quiet:   only print important messages\n");
  pallas_log(pallas::DebugLevel::Error, "Normal:  default verbosity level\n");
  pallas_log(pallas::DebugLevel::Error, "Verbose: print additional information\n");
  pallas_log(pallas::DebugLevel::Error, "Debug:   print many information\n");
  pallas_log(pallas::DebugLevel::Error, "Help:    print the diffent verbosity level and exit\n");
  pallas_log(pallas::DebugLevel::Error, "Max:     flood stdout with debug messages\n");
  exit(EXIT_SUCCESS);
}

void pallas_debug_level_set(enum pallas::DebugLevel lvl) {
  pallas::debugLevel = lvl;
  switch (pallas::debugLevel) {
  case pallas::DebugLevel::Error:  // only print errors
    pallas_log(pallas::DebugLevel::Normal, "Debug level: error\n");
    break;
  case pallas::DebugLevel::Quiet:
    pallas_log(pallas::DebugLevel::Normal, "Debug level: quiet\n");
    break;
  case pallas::DebugLevel::Normal:
    pallas_log(pallas::DebugLevel::Normal, "Debug level: normal\n");
    break;
  case pallas::DebugLevel::Verbose:
    pallas_log(pallas::DebugLevel::Normal, "Debug level: verbose\n");
    break;
  case pallas::DebugLevel::Debug:
    pallas_log(pallas::DebugLevel::Normal, "Debug level: debug\n");
    break;
  case pallas::DebugLevel::Help:
    pallas_log(pallas::DebugLevel::Normal, "Debug level: help\n");
    print_help();
    break;
  case pallas::DebugLevel::Max:
    pallas_log(pallas::DebugLevel::Normal, "Debug level: max\n");
    break;
  }
}

void pallas_debug_level_init() {
  if (const char* verbose_str = getenv("PALLAS_DEBUG_LVL")) {
    auto lvl = pallas::DebugLevel::Verbose;
    if (strcmp(verbose_str, "error") == 0)
      lvl = pallas::DebugLevel::Error;
    else if (strcmp(verbose_str, "quiet") == 0)
      lvl = pallas::DebugLevel::Quiet;
    else if (strcmp(verbose_str, "normal") == 0)
      lvl = pallas::DebugLevel::Normal;
    else if (strcmp(verbose_str, "verbose") == 0)
      lvl = pallas::DebugLevel::Verbose;
    else if (strcmp(verbose_str, "debug") == 0)
      lvl = pallas::DebugLevel::Debug;
    else if (strcmp(verbose_str, "help") == 0)
      lvl = pallas::DebugLevel::Help;
    else if (strcmp(verbose_str, "max") == 0)
      lvl = pallas::DebugLevel::Max;
    pallas_debug_level_set(lvl);
  }
}

enum pallas::DebugLevel pallas_debug_level_get() {
  return pallas::debugLevel;
};

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
