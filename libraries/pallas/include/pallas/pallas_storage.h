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
 * Stores the thread to the given path.
 * @param path Path to the folder.
 * @param thread Thread to be written to folder.
 */
void pallasStoreThread(const char* path, PALLAS(Thread) * thread);
/**
 * Store the archive.
 * @param archive Archive to be written to a folder.
 * @param path Path to the root folder.
 */
void pallasStoreArchive(PALLAS(Archive) * archive, const char* path);
/**
 * Store the global archive.
 * @param archive Archive to be written to a folder.
 * @param path Path to the root folder.
 */
void pallasStoreGlobalArchive(PALLAS(GlobalArchive) * archive, const char* path);

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
