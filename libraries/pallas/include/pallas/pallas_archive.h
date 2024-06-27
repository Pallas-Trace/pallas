/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * This file contains the definitions for Locations, Definitions and Archives.
 * An Archive is the closest thing to an actual trace file. There's one per process (approx).
 * Locations are threads, which are grouped by LocationGroups (processes, machines, etc.).
 * Definitions are where we store local data that we need to be able to parse later (String, regions, attributes).
 */
#pragma once

#include "pallas.h"

#ifdef __cplusplus
namespace pallas {
#endif

/**
 * A LocationGroup can be a process, a machine, etc.
 */
struct LocationGroup {
  /** Unique id for that group. */
  LocationGroupId id;
  /** Group name */
  StringRef name;
  /** Parent of that group. */
  LocationGroupId parent;
  /** Main location of that group. */
  ThreadId mainLoc;
};

/**
 * A Location is basically a thread (or GPU stream).
 */
struct Location {
  /** Unique id for that location. */
  ThreadId id;
  /** Location name. */
  StringRef name;
  /** Group containing that location. */
  LocationGroupId parent;
};

/**
 * A Definition stores Strings, Regions and Attributes.
 */
typedef struct Definition {
  /** List of String stored in that Definition. */
#ifdef __cplusplus
  std::map<StringRef, String> strings;
#else
  byte strings[MAP_SIZE];
#endif
  /** List of Region stored in that Definition. */
#ifdef __cplusplus
  std::map<RegionRef, Region> regions;
#else
  byte regions[MAP_SIZE];
#endif
  /** List of Attribute stored in that Definition. */
#ifdef __cplusplus
  std::map<AttributeRef, Attribute> attributes;
#else
  byte attributes[MAP_SIZE];
#endif
#ifdef __cplusplus
  [[nodiscard]] const String* getString(StringRef) const;
  void addString(StringRef, const char*);
  [[nodiscard]] const Region* getRegion(RegionRef) const;
  void addRegion(RegionRef, StringRef);
  [[nodiscard]] const Attribute* getAttribute(AttributeRef) const;
  void addAttribute(AttributeRef, StringRef, StringRef, pallas_type_t);
#endif
} Definition;

/**
 * An Archive represents a program.
 */
typedef struct Archive {
  char* dir_name;       /**< Name of the directory in which the archive is recorded. */
  char* trace_name;     /**< Name of the trace. */
  char* fullpath;       /**< \todo Complete this. */
  pthread_mutex_t lock; /**< Archive-wise lock, used for synchronising some threads. */

  LocationGroupId id CXX({PALLAS_LOCATION_GROUP_ID_INVALID}); /**< ID for the pallas::LocationGroup of that Archive. */
  struct Archive* global_archive; /**< The Global Archive is the archive encompassing the whole execution. NULL if
                                   * this is the global archive. */

  Definition definitions; /**< Definitions. */
  struct Thread** threads
    CXX({nullptr});         /**< Array of Thread.
                             * The memory of each thread is handled by their reader / writer individually. */
  int nb_threads;           /**< Number of Thread in #threads. */
  int nb_allocated_threads; /**< Size of #threads. */

  struct Archive** archive_list CXX({nullptr}); /**< Array of Archive *. */
  int nb_archives;                              /**< Number of Archive in #archive_list. */
  int nb_allocated_archives;                    /**< Size of #archive_list. */

  DEFINE_Vector(Location, locations);            /**< Vector of Location. */
  DEFINE_Vector(LocationGroup, location_groups); /**< Vector of LocationGroup. */

  short store_timestamps; /**< Indicates whether there are timestamps in there.*/
#ifdef __cplusplus
  [[nodiscard]] Thread* getThread(ThreadId) const;
  [[nodiscard]] const struct String* getString(StringRef string_ref);
  [[nodiscard]] const struct Region* getRegion(RegionRef region_ref);
  [[nodiscard]] const struct Attribute* getAttribute(AttributeRef attribute_ref);
  void addString(StringRef, const char*);
  void addRegion(RegionRef, StringRef);
  void addAttribute(AttributeRef, StringRef, StringRef, pallas_type_t);
  /**
   * Open a trace file and loads it it that Archive.
   * @param dirname Path to the file.
   * @param given_trace_name Name of the trace.
   * @param archive_id Id of this Archive.
   */
  void open(const char* dirname, const char* given_trace_name, LocationGroupId archive_id);
  /**
   * Open the Global Archive.
   * @param dirname Path to the file.
   * @param given_trace_name Name of the trace.
   */
  void globalOpen(const char* dirname, const char* given_trace_name) {
    open(dirname, given_trace_name, PALLAS_MAIN_LOCATION_GROUP_ID);
  };
  void defineLocationGroup(LocationGroupId id, StringRef name, LocationGroupId parent);
  void defineLocation(ThreadId id, StringRef name, LocationGroupId parent);
  void finalize() { pallas_error("Not implemented yet !\n"); };
  void close();

  [[nodiscard]] const LocationGroup* getLocationGroup(LocationGroupId) const;
  [[nodiscard]] const Location* getLocation(ThreadId) const;
  ~Archive();
#endif
} Archive;

#ifdef __cplusplus
} /* namespace pallas */
extern "C" {
#endif

/** Constructor for an Archive. In C, always use this to create a new Archive. */
extern PALLAS(Archive) * pallas_archive_new(void);

/**
 * Getter for a Thread from its id.
 * @returns First Thread matching the given pallas::ThreadId, or nullptr if it doesn't have a match.
 */
extern struct PALLAS(Thread) * pallas_archive_get_thread(PALLAS(Archive) * archive, PALLAS(ThreadId) thread_id);

/**
 * Getter for a LocationGroup from its id.
 * @returns First LocationGroup matching the given pallas::LocationGroupId in this archive, or in the global_archive if
 * it doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
extern const struct PALLAS(LocationGroup) *
  pallas_archive_get_location_group(PALLAS(Archive) * archive, PALLAS(LocationGroupId) location_group);

/**
 * Getter for a Location from its id.
 * @returns First Location matching the given pallas::ThreadId in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
extern const struct PALLAS(Location) *
  pallas_archive_get_location(PALLAS(Archive) * archive, PALLAS(ThreadId) thread_id);

/**
 * Creates a new String and adds it to that Archive.
 * Error if the given pallas::StringRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_archive_register_string(PALLAS(Archive) * archive, PALLAS(StringRef) string_ref, const char* string);

/**
 * Creates a new Region and adds it to that Archive.
 * Error if the given pallas::RegionRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_archive_register_region(PALLAS(Archive) * archive,
                                           PALLAS(RegionRef) region_ref,
                                           PALLAS(StringRef) string_ref);

/**
 * Creates a new Attribute and adds it to that Archive.
 * Error if the given pallas::AttributeRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_archive_register_attribute(PALLAS(Archive) * archive,
                                              PALLAS(AttributeRef) attribute_ref,
                                              PALLAS(StringRef) name_ref,
                                              PALLAS(StringRef) description_ref,
                                              PALLAS(pallas_type_t) type);

/**
 * Getter for a String from its id.
 * @returns First String matching the given pallas::StringRef in this archive, or in the global_archive if it doesn't
 * have a match, or nullptr if it doesn't have a match in the global_archive.
 */
extern const struct PALLAS(String) * pallas_archive_get_string(PALLAS(Archive) * archive, PALLAS(StringRef) string_ref);

/**
 * Getter for a Region from its id.
 * @returns First Region matching the given pallas::RegionRef in this archive, or in the global_archive if it doesn't
 * have a match, or nullptr if it doesn't have a match in the global_archive.
 */
extern const struct PALLAS(Region) * pallas_archive_get_region(PALLAS(Archive) * archive, PALLAS(RegionRef) region_ref);

/**
 * Getter for a Attribute from its id.
 * @returns First Attribute matching the given pallas::AttributeRef in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
extern const struct PALLAS(Attribute) *
  pallas_archive_get_attribute(PALLAS(Archive) * archive, PALLAS(AttributeRef) attribute_ref);
#ifdef __cplusplus
};
#endif /* __cplusplus */

/* -*-
  mode: c++;
  c-file-style: "k&r";
  c-basic-offset 2;
  tab-width 2 ;
  indent-tabs-mode nil
  -*- */
