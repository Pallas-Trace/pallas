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
#include "pallas_config.h"
#include "utils/pallas_parameter_handler.h"

#define GLOBAL_ARCHIVE_DEPRECATED_LOCATION CXX([[deprecated("You should record Locations on the Archives")]])

#ifdef __cplusplus
namespace pallas {
class ParameterHandler;
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

    /** List of Group stored in that Definition. */
#ifdef __cplusplus
    std::map<GroupRef, Group> groups;
#else
    byte groups[MAP_SIZE];
#endif

    /** List of Comm stored in that Definition. */
#ifdef __cplusplus
    std::map<CommRef, Comm> comms;
#else
    byte comms[MAP_SIZE];
#endif

#ifdef __cplusplus
    [[nodiscard]] const String* getString(StringRef) const;
    void addString(StringRef, const char*);
    [[nodiscard]] const Region* getRegion(RegionRef) const;
    void addRegion(RegionRef, StringRef);
    [[nodiscard]] const Attribute* getAttribute(AttributeRef) const;
    void addAttribute(AttributeRef, StringRef, StringRef, pallas_type_t);
    [[nodiscard]] const Group* getGroup(GroupRef) const;
    void addGroup(GroupRef, StringRef, GroupType group_type, Paradigm paradigm, uint32_t, const uint64_t*);
    [[nodiscard]] const Comm* getComm(CommRef) const;
    void addComm(CommRef, StringRef, GroupRef, CommRef);
#endif
} Definition;

#ifdef __cplusplus
typedef std::map<std::string, std::string> Metadata;
#else
typedef byte Metadata[MAP_SIZE];
#endif

/**
 * A GlobalArchive represents a program as a whole.
 */
typedef struct GlobalArchive {
    /** Name of the directory in which the archive is recorded. */
    char* dir_name;
    /** Name of the trace. */
    char* trace_name;
    /** Name of the directory + name if the trace. */
    char* fullpath;
    /** This GlobalArchive's ABI version. Mostly used in storage-related stuff. */
    uint8_t abi_version CXX({PALLAS_ABI_VERSION});
    /** Archive-wise lock, used for synchronising some threads. */
    pthread_mutex_t lock;
    /** Definitions. */
    Definition definitions;

    /** Array of pointers to the Archives. Each Archive is uniquely identifier by a LocationGroup.
     * This is only used when reading a trace (no synchronization between MPI Processes).
     */
    struct Archive** archive_list CXX({nullptr});
    /** Number of Archives in #archive_list. This should be equal to LocationGroup.size */
    int nb_archives;
    /**< Size of #archive_list. */
    int nb_allocated_archives;
    /** Vector of Locations. Each location uniquely identifies a Thread. */
    GLOBAL_ARCHIVE_DEPRECATED_LOCATION DEFINE_Vector(Location, locations);
    /** Vector of LocationGroups. Each LocationGroup uniquely identifies an Archive. */
    DEFINE_Vector(LocationGroup, location_groups);

    /** Metadata map we want to store to the trace. */
    Metadata metadata;
    /** List of parameters. Only used when reading the trace. */
    ParameterHandler* parameter_handler;

#ifdef __cplusplus
    /** Adds an additional content node.  */
    void add_metadata(const std::string& key, const std::string& value);
    /**
     * Getter for a String from its id.
     * @returns First String matching the given pallas::StringRef in this GlobalArchive. Nullptr if none was found.
     */
    [[nodiscard]] const String* getString(StringRef string_ref);

    /**
     * Getter for a Region from its id.
     * @returns First Region matching the given pallas::RegionRef in this GlobalArchive. Nullptr if none was found.
     */
    [[nodiscard]] const Region* getRegion(RegionRef region_ref);

    /**
     * Getter for a Attribute from its id.
     * @returns First Attribute matching the given pallas::AttributeRef in this GlobalArchive. Nullptr if none was found.
     */
    [[nodiscard]] const Attribute* getAttribute(AttributeRef attribute_ref);

    /**
     * Getter for a Group from its id.
     * @returns First Group matching the given pallas::GroupRef in this GlobalArchive. Nullptr if none was found.
     */
    [[nodiscard]] const Group* getGroup(GroupRef group_ref);

    /**
     * Getter for a Comm from its id.
     * @returns First Comm matching the given pallas::CommRef in this GlobalArchive. Nullptr if none was found.
     */
    [[nodiscard]] const Comm* getComm(CommRef comm_ref);

    /**
     * Creates a new String and adds it to that GlobalArchive.
     * Error if the given pallas::StringRef is already in use.
     * Locks and unlocks the mutex for that operation.
     */
    void addString(StringRef, const char*);

    /**
     * Creates a new Region and adds it to that GlobalArchive.
     * Error if the given pallas::RegionRef is already in use.
     * Locks and unlocks the mutex for that operation.
     */
    void addRegion(RegionRef, StringRef);

    /**
     * Creates a new Attribute and adds it to that GlobalArchive.
     * Error if the given pallas::AttributeRef is already in use.
     * Locks and unlocks the mutex for that operation.
     */
    void addAttribute(AttributeRef, StringRef, StringRef, pallas_type_t);

    /**
     * Creates a new Group and adds it to that GlobalArchive.
     * Error if the given pallas::GroupRef is already in use.
     * Locks and unlocks the mutex for that operation.
     */
    void addGroup(GroupRef, StringRef, GroupType group_type, Paradigm paradigm, uint32_t, const uint64_t*);

    /**
     * Creates a new Comm and adds it to that GlobalArchive.
     * Error if the given pallas::CommRef is already in use.
     * Locks and unlocks the mutex for that operation.
     */
    void addComm(CommRef, StringRef, GroupRef, CommRef);

    /**
     * Create a GlobalArchive. Does not read or write anything in the given .
     * @param dirname Path to the file.
     * @param given_trace_name Name of the trace.
     */
    GlobalArchive(const char* dirname, const char* given_trace_name);

    /**
     * Creates a new LocationGroup and adds it to that GlobalArchive.
     */
    void defineLocationGroup(LocationGroupId id, StringRef name, LocationGroupId parent);

    /**
     * Creates a new Location and adds it to that GlobalArchive. You should be doing this to an Archive.
     */
    GLOBAL_ARCHIVE_DEPRECATED_LOCATION void defineLocation(ThreadId id, StringRef name, LocationGroupId parent);

    /**
     * Close and save that GlobalArchive.
     * @param path Path to the root directory.
     * @param parameter_handler Handler for the storage options.
     */
    void store(const char* path, const ParameterHandler* parameter_handler);

    /**
     * Store this GlobalArchive to dir_name.
     * @param parameter_handler Handler for the storage options.
     */
    void store(const ParameterHandler* parameter_handler) { store(dir_name, parameter_handler); }
    /**
     * Store this GlobalArchive to dir_name using parameter_handler.
     */
    void store() { store(dir_name, parameter_handler); }
    /**
     * Getter for a LocationGroup from its id.
     * @returns First LocationGroup matching the given pallas::LocationGroupId in this GlobalArchive. Nullptr if none was found.
     */
    [[nodiscard]] const LocationGroup* getLocationGroup(LocationGroupId) const;

    /**
     * Getter for a Location from its id.
     * @returns First Location matching the given pallas::ThreadId in this GlobalArchive's archives. Nullptr if none was found.
     */
    [[nodiscard]] const Location* getLocation(ThreadId);

    /**
     * Aggregates a list of the locations of all the Archives.
     */
    [[nodiscard]] std::vector<Location> getLocationList();

    /**r example, the std::vector template has a default argument for the allocator:
     * Aggregates a list of the Threads of all the Archives.
     */
    [[nodiscard]] std::vector<Thread*> getThreadList();

    [[nodiscard]] Archive* getArchive(LocationGroupId archiveId, bool print_warning = true);

    void freeArchive(LocationGroupId archiveId);

    [[nodiscard]] struct Archive* getArchiveFromLocation(ThreadId) const;
    /** Returns the starting timestamp of this trace. Loads the threads. */
    pallas_timestamp_t get_starting_timestamp();
    /** Returns the final timestamp of this trace. Loads the threads. */
    pallas_timestamp_t get_ending_timestamp();

    ~GlobalArchive();
#endif
} GlobalArchive;

/**
 * An Archive represents a process.
 */
typedef struct Archive {
    /** Name of the directory in which the archive is recorded. */
    char* dir_name;
    /** Archive-wise lock, used for synchronising some threads. */
    pthread_mutex_t lock;

    /** ID for the pallas::LocationGroup of that Archive. */
    LocationGroupId id CXX({PALLAS_LOCATION_GROUP_ID_INVALID});
    /** The Global Archive is the archive encompassing the whole execution. */
    GlobalArchive* global_archive;

    /** Array of Thread.
     * The memory of each thread is handled by their reader / writer individually. */
    struct Thread** threads CXX({nullptr});
    /** Number of Thread in #threads. */
    size_t nb_threads;
    /** Size of #threads. */
    size_t nb_allocated_threads;
    /** Local definitions. */
    Definition definitions;
    /** Vector of Locations. Each location uniquely identifies a Thread. */
    DEFINE_Vector(Location, locations);
    /** Vector of LocationGroups. Each LocationGroup uniquely identifies an Archive. */
    DEFINE_Vector(LocationGroup, location_groups);
    /** Metadata map we want to store to the archive. */
    Metadata metadata;
#ifdef __cplusplus

    /** Adds an entry to the metadata. */
    void add_metadata(const std::string& key, const std::string& value);
    /**
     * Getter for a String from its id.
     * @returns First String matching the given pallas::StringRef in this archive, then global_archive. Nullptr if none was found.
     */
    [[nodiscard]] const String* getString(StringRef string_ref);
    /**
     * Getter for a Region from its id.
     * @returns First Region matching the given pallas::RegionRef in this archive, then global_archive. Nullptr if none was found.
     */
    [[nodiscard]] const Region* getRegion(RegionRef region_ref);
    /**
     * Getter for a Attribute from its id.
     * @returns First Attribute matching the given pallas::AttributeRef in this archive, then global_archive. Nullptr if none was found.
     */
    [[nodiscard]] const Attribute* getAttribute(AttributeRef attribute_ref);
    /**
     * Getter for a Group from its id.
     * @returns First Group matching the given pallas::GroupRef in this archive, then global_archive. Nullptr if none was found.
     */
    [[nodiscard]] const Group* getGroup(GroupRef group_ref);
    /**
     * Getter for a Comm from its id.
     * @returns First Comm matching the given pallas::CommRef in this archive, then global_archive. Nullptr if none was found.
     */
    [[nodiscard]] const Comm* getComm(CommRef comm_ref);
    /**
     * Creates a new String and adds it to that Archive.
     * Error if the given pallas::StringRef is already in use.
     * Locks and unlocks the mutex for that operation.
     */
    void addString(StringRef, const char*);
    /**
     * Creates a new Region and adds it to that Archive.
     * Error if the given pallas::RegionRef is already in use.
     * Locks and unlocks the mutex for that operation.
     */
    void addRegion(RegionRef, StringRef);
    /**
     * Creates a new Attribute and adds it to that Archive.
     * Error if the given pallas::AttributeRef is already in use.
     * Locks and unlocks the mutex for that operation.
     */
    void addAttribute(AttributeRef, StringRef, StringRef, pallas_type_t);
    /**
     * Creates a new Group and adds it to that definition.
     * Error if the given pallas::GroupRef is already in use.
     * Locks and unlocks the mutex for that operation.
     */
    void addGroup(GroupRef, StringRef, uint32_t, const uint64_t*, GroupType group_type, Paradigm paradigm);
    /**
     * Creates a new Comm and adds it to that definition.
     * Error if the given pallas::CommRef is already in use.
     * Locks and unlocks the mutex for that operation.
     */
    void addComm(CommRef, StringRef, GroupRef, CommRef);
    /**
     * Creates a new Location and adds it to that Archive.
     */
    void defineLocation(ThreadId id, StringRef name, LocationGroupId parent);

    /**
     * Creates a new LocationGroup and adds it to that Archive.
     */
    void defineLocationGroup(LocationGroupId id, StringRef name, LocationGroupId parent);
    /**
     * Getter for a LocationGroup from its id.
     * @returns First LocationGroup matching the given pallas::LocationGroupId in this Archive, then global_archive. Nullptr if none was found.
     */
    [[nodiscard]] const LocationGroup* getLocationGroup(LocationGroupId) const;
    /**
     * Getter for a Location from its id.
     * @returns First Location matching the given pallas::ThreadId in this Archive, then global_archive. Nullptr if none was found.
     */
    [[nodiscard]] const Location* getLocation(ThreadId) const;

    [[nodiscard]] Thread* getThread(ThreadId);
    [[nodiscard]] Thread* getThreadAt(size_t index);
    const char* getName();
    void freeThread(ThreadId);
    void freeThreadAt(size_t);

    /**
     * Store this Archive.
     * @param path Path to the root directory.
     * @param parameter_handler Handler for the storage options.
     */
    void store(const char* path, const ParameterHandler* parameter_handler);
    /**
     * Store this GlobalArchive to dir_name.
     * @param parameter_handler Handler for the storage options.
     */
    void store(const ParameterHandler* parameter_handler = nullptr);
    Archive(const char* dirname, LocationGroupId archive_id);
    Archive(GlobalArchive& global_archive, LocationGroupId archive_id);
    ~Archive();
#endif
} Archive;

#ifdef __cplusplus
} /* namespace pallas */
extern "C" {
#endif

/** Constructor for an Archive. In C, always use this to create a new Archive. */
extern PALLAS(Archive) * pallas_archive_new(const char* dir_name, PALLAS(LocationGroupId) location_group);

/** Destructor for an Archive. */
extern void pallas_archive_delete(PALLAS(Archive*));

/** Opens a GlobalArchive. In C, always use this to create a new GlobalArchive.
 * @param dirname Path to the file.
 * @param given_trace_name Name of the trace.
 */
extern PALLAS(GlobalArchive) * pallas_global_archive_new(const char* dirname, const char* given_trace_name);

/** Destructor for a Global Archive. */
extern void pallas_global_archive_delete(PALLAS(GlobalArchive*));
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
extern const struct PALLAS(LocationGroup) * pallas_archive_get_location_group(PALLAS(GlobalArchive) * archive, PALLAS(LocationGroupId) location_group);

/**
 * Getter for a Location from its id.
 * @returns First Location matching the given pallas::ThreadId in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
extern const struct PALLAS(Location) * pallas_archive_get_location(PALLAS(GlobalArchive) * archive, PALLAS(ThreadId) thread_id);

/**
 * @returns The archive that contains a pallas::ThreadId, or nullptr if not found.
 */
extern const PALLAS(Archive) * pallas_archive_get_archive_from_location(PALLAS(GlobalArchive) * archive, PALLAS(ThreadId) thread_id);
/**
 * Creates a new String and adds it to that GlobalArchive.
 * Error if the given pallas::StringRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_global_archive_register_string(PALLAS(GlobalArchive) * archive, PALLAS(StringRef) string_ref, const char* string);

/**
 * Creates a new Region and adds it to that GlobalArchive.
 * Error if the given pallas::RegionRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_global_archive_register_region(PALLAS(GlobalArchive) * archive, PALLAS(RegionRef) region_ref, PALLAS(StringRef) string_ref);

/**
 * Creates a new Attribute and adds it to that GlobalArchive.
 * Error if the given pallas::AttributeRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_global_archive_register_attribute(PALLAS(GlobalArchive) * archive,
                                                     PALLAS(AttributeRef) attribute_ref,
                                                     PALLAS(StringRef) name_ref,
                                                     PALLAS(StringRef) description_ref,
                                                     PALLAS(pallas_type_t) type);

/**
 * Creates a new Group and adds it to that GlobalArchive.
 * Error if the given pallas::GroupRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_global_archive_register_group(PALLAS(GlobalArchive) * archive,
                                                 PALLAS(GroupRef) group_ref,
                                                 PALLAS(StringRef) name_ref,
                                                 enum PALLAS(GroupType) group_type,
                                                 enum PALLAS(Paradigm) paradigm,
                                                 uint32_t number_of_members,
                                                 const uint64_t* members);

/**
 * Creates a new Comm and adds it to that GlobalArchive.
 * Error if the given pallas::CommRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_global_archive_register_comm(PALLAS(GlobalArchive) * archive,
                                                PALLAS(CommRef) comm_ref,
                                                PALLAS(StringRef) name_ref,
                                                PALLAS(GroupRef) group_ref,
                                                PALLAS(CommRef) parent_ref);
/**
 * Creates a new LocationGroup and adds it to that GlobalArchive.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_global_archive_define_location_group(PALLAS(GlobalArchive) * archive, PALLAS(LocationGroupId) id, PALLAS(StringRef) name, PALLAS(LocationGroupId) parent);

/**
 * Creates a new Location and adds it to that GlobalArchive.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_global_archive_define_location(PALLAS(GlobalArchive) * archive, PALLAS(ThreadId) id, PALLAS(StringRef) name, PALLAS(LocationGroupId) parent);

/**
 * Creates a new LocationGroup and adds it to that Archive.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_archive_define_location_group(PALLAS(Archive) * archive, PALLAS(LocationGroupId) id, PALLAS(StringRef) name, PALLAS(LocationGroupId) parent);
/**
 * Creates a new Location and adds it to that Archive.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_archive_define_location(PALLAS(Archive) * archive, PALLAS(ThreadId) id, PALLAS(StringRef) name, PALLAS(LocationGroupId) parent);

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
extern void pallas_archive_register_region(PALLAS(Archive) * archive, PALLAS(RegionRef) region_ref, PALLAS(StringRef) string_ref);

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
 * Creates a new Group and adds it to that Archive.
 * Error if the given pallas::GroupRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_archive_register_group(PALLAS(Archive) * archive,
                                          PALLAS(GroupRef) group_ref,
                                          PALLAS(StringRef) name_ref,
                                          enum PALLAS(GroupType) group_type,
                                          enum PALLAS(Paradigm) paradigm,
                                          uint32_t number_of_members,
                                          const uint64_t* members);

/**
 * Creates a new Comm and adds it to that Archive.
 * Error if the given pallas::CommRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
extern void pallas_archive_register_comm(PALLAS(Archive) * archive, PALLAS(CommRef) comm_ref, PALLAS(StringRef) name_ref, PALLAS(GroupRef) group_ref, PALLAS(CommRef) parent_ref);

/**
 * Getter for a String from its id.
 * @returns First String matching the given pallas::StringRef in this archive, or in the global_archive if it doesn't
 * have a match, or nullptr if it doesn't have a match in the global_archive.
 */
extern const struct PALLAS(String) * pallas_archive_get_string(PALLAS(GlobalArchive) * archive, PALLAS(StringRef) string_ref);

/**
 * Getter for a Region from its id.
 * @returns First Region matching the given pallas::RegionRef in this archive, or in the global_archive if it doesn't
 * have a match, or nullptr if it doesn't have a match in the global_archive.
 */
extern const struct PALLAS(Region) * pallas_archive_get_region(PALLAS(GlobalArchive) * archive, PALLAS(RegionRef) region_ref);

/**
 * Getter for a Attribute from its id.
 * @returns First Attribute matching the given pallas::AttributeRef in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
extern const struct PALLAS(Attribute) * pallas_archive_get_attribute(PALLAS(GlobalArchive) * archive, PALLAS(AttributeRef) attribute_ref);

/**
 * Getter for a Group from its id.
 * @returns First Group matching the given pallas::GroupRef in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
extern const struct PALLAS(Group) * pallas_archive_get_group(PALLAS(GlobalArchive) * archive, PALLAS(GroupRef) group_ref);

/**
 * Getter for a Comm from its id.
 * @returns First Comm matching the given pallas::CommRef in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
extern const struct PALLAS(Comm) * pallas_archive_get_comm(PALLAS(GlobalArchive) * archive, PALLAS(CommRef) comm_ref);

extern void pallas_global_archive_add_metadata(PALLAS(GlobalArchive*) archive, const char* key, const char* value);

extern void pallas_archive_add_metadata(PALLAS(Archive*) archive, const char* key, const char* value);

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
