/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Functions related to reading/writing a trace file.
 */
#pragma once

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
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
 * @param path Path to the root folder.
 * @param thread Thread to be written.
 * @param parameter_handler Handler for the storage parameters.
 * @param load_thread Whether you should load the timestamps before writing.
 */
void pallasStoreThread(const char* path, PALLAS(Thread) * thread, const PALLAS(ParameterHandler)* parameter_handler, bool load_thread);
/**
 * Store the archive.
 * @param archive Archive to be written to a folder.
 * @param path Path to the root folder.
 * @param parameter_handler Handler for the storage parameters.
 */
void pallasStoreArchive(PALLAS(Archive) * archive, const char* path, const PALLAS(ParameterHandler)* parameter_handler);
/**
 * Store the global archive.
 * @param archive Archive to be written to a folder.
 * @param path Path to the root folder.
 * @param parameter_handler Handler for the storage parameters.
 */
void pallasStoreGlobalArchive(PALLAS(GlobalArchive) * archive, const char* path, const PALLAS(ParameterHandler)* parameter_handler);

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
