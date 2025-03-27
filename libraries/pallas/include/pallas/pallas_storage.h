/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Functions related to reading/writing a trace file.
 */
#pragma once

#include "pallas.h"
#include "pallas_archive.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 * Creates the directories for the trace to be written.
 * @param archive Archive to be written to a folder.
 */
void pallas_storage_init(const char * dir_name);
/**
 * Finalize the writing process by writing the thread.
 * @param thread Thread to be written to folder.
 */
void pallas_storage_finalize_thread(PALLAS(Thread) * thread);
/**
 * Finalize the writing process by writing the whole archive.
 * @param archive Archive to be written to a folder.
 */
void pallasStoreArchive(PALLAS(Archive) * archive);
/**
 * Finalize the writing process by writing the global archive.
 * @param archive Archive to be written to a folder.
 */
void pallasStoreGlobalArchive(PALLAS(GlobalArchive) * archive);

   /**
   * Allocate and read an archive from a `main.pallas` file.
   * @param trace_filename Path to a `main.pallas` file.
   * @return Pointer to a GlobalArchive if successful, else nullptr.
   */
PALLAS(GlobalArchive*) pallas_open_trace(const char* trace_filename);
#ifdef __cplusplus
};
#endif

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
