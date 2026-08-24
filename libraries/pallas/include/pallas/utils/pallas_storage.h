/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Functions related to reading/writing a trace file.
 */
#pragma once

#include <libgen.h>
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

#ifdef __cplusplus
namespace pallas {
File* getFirstOpenFile();
#endif

static size_t numberOpenFiles;
static size_t maxNumberFilesOpen;


typedef struct File {
    
#ifdef __cplusplus
  public:
#endif
    FILE* file CXX({nullptr});
    char* path CXX({nullptr});
    bool isOpen CXX({false});

#ifdef __cplusplus
    bool is_open() const { return isOpen; }

    static void pallasMkdir(const char* directory_name, mode_t mode);

    // TODO Add the file mode to the File class
    void open(const char* mode) {
        if (isOpen) {
            pallas_log(pallas::DebugLevel::Verbose, "Trying to open file that is already open: %s\n", path);
            // store();
            return;
        }
        while (numberOpenFiles >= maxNumberFilesOpen) {
            auto* openedFilePath = getFirstOpenFile();
            if (!openedFilePath) {
                pallas_warn("Could not find any more duration files to store: %lu files opened.\n", numberOpenFiles);
                break;
            }
            openedFilePath->close();
        }

        pallas_log(pallas::DebugLevel::Debug, "Open %s with mode %s\n", path, mode);
        
        // dirname modifies its parameter, so we need to duplicate the string
        char* path_copy = strdup(path);
        pallasMkdir(dirname(path_copy), 0777);
        free(path_copy);

        file = fopen(path, mode);
        if (file == nullptr) {
            pallas_warn("Cannot open %s: %s\n", path, strerror(errno));
        } else {
            numberOpenFiles++;
            isOpen = true;
        }
    };

    void close() {
        // TODO grab the lock
        if (!isOpen) {
            pallas_log(pallas::DebugLevel::Debug, "Trying to store file that is already closed: %s\n", path);
        }
        isOpen = false;
        fclose(file);
        if (numberOpenFiles)
            numberOpenFiles--;
    };
    
    void read(void* ptr, size_t size, size_t n) const {
        if (size > 0) {
            size_t ret = fread(ptr, size, n, file);
            if (ret != (n))
                pallas_error("fread failed: %d %s\n", errno, strerror(errno));
        }
    }

    void write(const void* ptr, size_t size, size_t n) const {
        if (size > 0) {
            size_t ret = fwrite(ptr, size, n, file);
            if (ret != (n))
                pallas_error("fwrite failed: %d %s\n", errno, strerror(errno));
        }
    }

    void writeString(const std::string& str) const {
        auto size = str.size() + 1;
        write(&size, sizeof(size), 1);
        write(str.data(), sizeof(char), size);
    }

    [[nodiscard]] std::string readString() const {
        size_t size = 0;
        read(&size, sizeof(size), 1);
        char* str = new char[size];
        read(str, sizeof(char), size);
        return str;
    }

    [[nodiscard]] size_t offset() const {
        if(isOpen)
            return fseek(file, 0, SEEK_CUR);
        return -1;
    }

    explicit File(const char* path, const char* mode = nullptr) {
        static bool first_time = true;
        if(first_time) {
            numberOpenFiles = 0;
            maxNumberFilesOpen = 32;
            first_time=false;
        }
        this->path = strdup(path);
        if (mode) {
            open(mode);
        }
    }

    ~File() {
        if (isOpen) {
            close();
        }
        // delete file;
        free(path);
    }
#endif

} File;

#ifdef __cplusplus
};
#endif

#if 0
#define _pallas_fread(ptr, size, nmemb, stream)                            \
    do {                                                                   \
        size_t ret = fread(ptr, size, nmemb, stream);                      \
        if (ret != (nmemb))                                                \
            pallas_error("fread failed: %d %s\n", errno, strerror(errno)); \
    } while (0)

#define _pallas_fwrite(ptr, size, nmemb, stream)       \
    do {                                               \
        size_t ret = fwrite(ptr, size, nmemb, stream); \
        if (ret != (nmemb))                            \
            pallas_error("fwrite failed\n");           \
    } while (0)
#endif

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
