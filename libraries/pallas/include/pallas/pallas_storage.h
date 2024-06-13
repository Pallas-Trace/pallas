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
void pallas_storage_init(PALLAS(Archive) * archive);
/**
 * Finalize the writing process by writing the thread.
 * @param thread Thread to be written to folder.
 */
void pallas_storage_finalize_thread(PALLAS(Thread) * thread);
/**
 * Finalize the writing process by writing the whole archive.
 * @param archive Archive to be written to a folder.
 */
void pallas_storage_finalize(PALLAS(Archive) * archive);
/**
 * Returns the path of the archive's folder.
 * @param dir_name Directory for the archive's storage.
 * @param trace_name Name of the trace.
 * @return Path to the archive's folder.
 */
char* pallas_archive_fullpath(char* dir_name, char* trace_name);
/**
 * Read an archive from a `main.pallas` file.
 * @param archive Pointer to an allocated archive.
 * @param main_filename Path to a `main.pallas` file.
 */
void pallas_read_main_archive(PALLAS(Archive) * archive, char* main_filename);
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
