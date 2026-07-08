/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include <cstddef>
#include "pallas/pallas_record.h"

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"

namespace pallas {
/**
 * Getter for a String from its id.
 * @returns First String matching the given pallas::StringRef, nullptr if it doesn't have a match.
 */
const String* Definition::get_string(StringRef string_ref) const {
  if (strings.count(string_ref) > 0)
    return &strings.at(string_ref);
  else
    return nullptr;
}

/**
 * Creates a new String and adds it to that definition. Error if the given pallas::StringRef is already in use.
 */
void Definition::add_string(StringRef string_ref, const char* string) {
  if (get_string(string_ref)) {
    pallas_error("Given string_ref was already in use.\n");
  }

  auto& s = strings[string_ref];
  s.string_ref = string_ref;
  s.length = strlen(string) + 1;
  s.str = (char*) calloc(s.length, sizeof(char));
  strncpy(s.str, string, s.length);

  pallas_log(DebugLevel::Verbose, "Register string #%zu{.ref=%d, .length=%d, .str='%s'}\n", strings.size() - 1, s.string_ref, s.length, s.str);
}

/**
 * Getter for a Region from its id.
 * @returns First Region matching the given pallas::RegionRef, nullptr if it doesn't have a match.
 */
const Region* Definition::get_region(RegionRef region_ref) const {
  if (regions.count(region_ref) > 0)
    return &regions.at(region_ref);
  else
    return nullptr;
}

/**
 * Creates a new Region and adds it to that definition. Error if the given pallas::RegionRef is already in use.
 */
void Definition::add_region(RegionRef region_ref, StringRef string_ref) {
  if (get_region(region_ref)) {
    pallas_error("Given region_ref was already in use.\n");
  }

  auto& r = regions[region_ref];
  r.region_ref = region_ref;
  r.string_ref = string_ref;

  pallas_log(DebugLevel::Verbose, "Register region #%zu{.ref=%d, .str=%d}\n", regions.size() - 1, r.region_ref, r.string_ref);
}

/**
 * Getter for a Attribute from its id.
 * @returns First Attribute matching the given pallas::AttributeRef, nullptr if it doesn't have a match.
 */
const Attribute* Definition::get_attribute(AttributeRef attribute_ref) const {
  if (attributes.count(attribute_ref) > 0)
    return &attributes.at(attribute_ref);
  return nullptr;
}

/**
 * Creates a new Attribute and adds it to that definition. Error if the given pallas::AttributeRef is already in use.
 */
void Definition::add_attribute(AttributeRef attribute_ref, StringRef name_ref, StringRef description_ref, pallas_type_t type) {
  if (get_attribute(attribute_ref)) {
    pallas_error("Given attribute_ref was already in use.\n");
  }
  auto& a= attributes[attribute_ref];
  a.attribute_ref = attribute_ref;
  a.name = name_ref;
  a.description = description_ref;
  a.type = type;

  pallas_log(DebugLevel::Verbose, "Register attribute #%zu{.ref=%d, .name=%d, .description=%d, .type=%d}\n", attributes.size() - 1, a.attribute_ref, a.name, a.description, a.type);
}

/**
 * Getter for a Group from its id.
 * @returns First Group matching the given pallas::GroupRef, nullptr if it doesn't have a match.
 */
const Group* Definition::get_group(GroupRef group_ref) const {
  if (groups.count(group_ref) > 0)
    return &groups.at(group_ref);
  else
    return nullptr;
}

/**
 * Creates a new Group and adds it to that definition. Error if the given pallas::GroupRef is already in use.
 */
void Definition::add_group(GroupRef group_ref,
                          StringRef name,
                          GroupType group_type,
                          Paradigm paradigm,
                          uint32_t number_of_members,
                          const uint64_t* members) {
    if (get_group(group_ref)) {
        pallas_error("Given group_ref was already in use.\n");
    }

    auto& g = groups[group_ref];
    g.group_ref = group_ref;
    g.name = name;
    g.group_type = group_type;
    g.paradigm = paradigm;
    g.numberOfMembers = number_of_members;
    g.members = new uint32_t[number_of_members];
    for (uint32_t i = 0; i < number_of_members; i++)
        g.members[i] = members[i];

    pallas_log(DebugLevel::Verbose, "Register group #%zu{.ref=%d, .name=%d, .type=%d, .paradigm=%d, .nbMembers=%d}\n",
        groups.size() - 1, g.group_ref, g.name, g.group_type, g.paradigm, g.numberOfMembers);
}

/**
 * Getter for a Comm from its id.
 * @returns First Comm matching the given pallas::CommRef, nullptr if it doesn't have a match.
 */
const Comm* Definition::get_comm(CommRef comm_ref) const {
  if (comms.count(comm_ref) > 0)
    return &comms.at(comm_ref);
  else
    return nullptr;
}

/**
 * Creates a new Comm and adds it to that definition. Error if the given pallas::CommRef is already in use.
 */
void Definition::add_comm(CommRef comm_ref, StringRef name, GroupRef group, CommRef parent) {
  if (get_comm(comm_ref)) {
    pallas_error("Given comm_ref was already in use.\n");
  }

  auto& c = comms[comm_ref];
  c.comm_ref = comm_ref;
  c.name = name;
  c.group = group;
  c.parent = parent;

  pallas_log(DebugLevel::Verbose, "Register comm #%zu{.ref=%d, .str=%d, .group=%d, .parent=%d}\n", comms.size() - 1, c.comm_ref, c.name, c.group, c.parent);
}

char* pallas_global_archive_fullpath(char* dir_name, char* trace_name) {
  int len = strlen(dir_name) + strlen(trace_name) + 2;
  char* fullpath = new char[len];
  snprintf(fullpath, len, "%s/%s", dir_name, trace_name);
  return fullpath;
}

GlobalArchive::GlobalArchive(const char* dirname, const char* given_trace_name) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  pallas_debug_level_init();
  dir_name = strdup(dirname);
  trace_name = strdup(given_trace_name);
  fullpath = pallas_global_archive_fullpath(dir_name, trace_name);
  nb_archives = 0;
  nb_allocated_archives = 0;
  lock = {};

  pthread_mutex_init(&lock, nullptr);

  pallas_recursion_shield--;
}

void GlobalArchive::define_location_group(LocationGroupId lg_id, StringRef name, LocationGroupId parent) {
  pthread_mutex_lock(&lock);
  auto l = LocationGroup();
  l.id = lg_id;
  l.name = name;
  l.parent = parent;
  location_groups.push_back(l);
  pthread_mutex_unlock(&lock);
}

void GlobalArchive::define_location(ThreadId l_id, StringRef name, LocationGroupId parent) {
    static bool has_warned_before = false;
    if (! has_warned_before) {
        pallas_warn("Defining Location %d (%s) in GlobalArchive: You should record in it %d's Archive. This warning will only show once.\n", l_id, get_string(name)->str, parent);
        has_warned_before = true;
    }
    pthread_mutex_lock(&lock);
  Location l = {.id = l_id, .name = name, .parent = parent};
  pallas_assert(l.id != PALLAS_THREAD_ID_INVALID);
  locations.push_back(l);
  pthread_mutex_unlock(&lock);
}

const String* GlobalArchive::get_string(StringRef string_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.get_string(string_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

const Region* GlobalArchive::get_region(RegionRef region_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.get_region(region_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

const Attribute* GlobalArchive::get_attributes(AttributeRef attribute_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.get_attribute(attribute_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

const Group* GlobalArchive::get_group(GroupRef group_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.get_group(group_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

const Comm* GlobalArchive::get_comm(CommRef comm_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.get_comm(comm_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

const LocationGroup* GlobalArchive::getLocationGroup(LocationGroupId location_group_id) const {
  for (auto& lc : location_groups) {
    if (lc.id == location_group_id) {
      return &lc;
    }
  }
  return nullptr;
}
const Location* GlobalArchive::getLocation(ThreadId location_id) {
    for (auto& lg : location_groups) {
        auto a = getArchive(lg.id)->get_location(location_id);
        if (a != nullptr) {
            return a;
        }
    }
  return nullptr;
}

std::vector<Location> GlobalArchive::getLocationList() {
    std::vector<Location> output;
    for (auto& lg : location_groups) {
        auto a = getArchive(lg.id);
        output.insert(output.end(), a->locations.begin(), a->locations.end());
    }
    return output;
}

std::vector<Thread*> GlobalArchive::getThreadList() {
    std::vector<Thread*> output;
    for (auto& lg : location_groups) {
        auto a = getArchive(lg.id);
        for (const auto& l : a->locations) {
            auto* t = a->get_thread(l.id);
            output.push_back(t);
        }
    }
  return output;
}



Archive* GlobalArchive::getArchiveFromLocation(ThreadId location_id) const {
  for (int i = 0; i < nb_archives; i++) {
    if (archive_list[i]->get_thread(location_id))
      return archive_list[i];
  }
  return nullptr;
}

StringRef GlobalArchive::add_string(StringRef string_ref, const char* string) {
  pthread_mutex_lock(&lock);
  for (; definitions.get_string(string_ref) == nullptr; string_ref++) {}
  definitions.add_string(string_ref, string);
  pthread_mutex_unlock(&lock);
  return string_ref;
}

RegionRef GlobalArchive::add_region(RegionRef region_ref, StringRef name_ref) {
  pthread_mutex_lock(&lock);
  for (; definitions.get_string(region_ref) == nullptr; region_ref++) {}
  definitions.add_region(region_ref, name_ref);
  pthread_mutex_unlock(&lock);
  return region_ref;
}

AttributeRef GlobalArchive::get_attributes(AttributeRef attribute_ref, StringRef name_ref, StringRef description_ref, pallas_type_t type) {
  pthread_mutex_lock(&lock);
  for (; definitions.get_string(attribute_ref) == nullptr; attribute_ref++) {}
  definitions.add_attribute(attribute_ref, name_ref, description_ref, type);
  pthread_mutex_unlock(&lock);
  return attribute_ref;
}

GroupRef GlobalArchive::add_group(GroupRef group_ref, StringRef name, GroupType group_type, Paradigm
                             paradigm, uint32_t number_of_members, const uint64_t* members) {
  pthread_mutex_lock(&lock);
  for (; definitions.get_string(group_ref) == nullptr; group_ref++) {}
  definitions.add_group(group_ref, name, group_type, paradigm, number_of_members, members);
  pthread_mutex_unlock(&lock);
  return group_ref;
}

CommRef GlobalArchive::add_comm(CommRef comm_ref, StringRef name, GroupRef group, CommRef parent) {
  pthread_mutex_lock(&lock);
  for (; definitions.get_string(comm_ref) == nullptr; comm_ref++) {}
  definitions.add_comm(comm_ref, name, group, parent);
  pthread_mutex_unlock(&lock);
  return comm_ref;
}

GlobalArchive::~GlobalArchive() {
    pallas_log(DebugLevel::Debug, "Deleting GlobalArchive\n");
  free(dir_name);
  free(trace_name);
  delete[] fullpath;
  for (size_t i = 0; i < nb_archives; i++) {
    delete archive_list[i];
  }
  delete[] archive_list;
};

pallas_timestamp_t GlobalArchive::get_starting_timestamp() {
    pallas_timestamp_t out = -1;
    for (auto& thread: getThreadList()) {
        out = std::min(out, thread->first_timestamp);
    }
    return out;
}

pallas_timestamp_t GlobalArchive::get_ending_timestamp() {
    pallas_timestamp_t out = 0;
    for (auto& thread: getThreadList()) {
        out = std::max(out, thread->getLastTimestamp());
    }
    return out;
}

Archive::~Archive() {
    pallas_log(DebugLevel::Debug, "Deleting Archive %d\n", id);
  free(dir_name);
  for (size_t i = 0; i < nb_threads; i++) {
    delete threads[i];
  }
  delete[] threads;
}

Archive::Archive(GlobalArchive& global_archive, LocationGroupId archive_id) : Archive(global_archive.dir_name, archive_id) {
  this->global_archive = &global_archive;
}

Archive::Archive(const char* dirname, LocationGroupId archive_id) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  pallas_debug_level_init();
  dir_name = strdup(dirname);
  id = archive_id;
  global_archive = nullptr;
  lock = {};
  pthread_mutex_init(&lock, nullptr);

  nb_allocated_threads = NB_THREADS_DEFAULT;
  nb_threads = 0;
  threads = new Thread*[nb_allocated_threads];
  pallas_recursion_shield--;
}

void Archive::add_string(StringRef string_ref, const char* string) {
  pthread_mutex_lock(&lock);
  definitions.add_string(string_ref, string);
  pthread_mutex_unlock(&lock);
}

void Archive::add_region(RegionRef region_ref, StringRef name_ref) {
  pthread_mutex_lock(&lock);
  definitions.add_region(region_ref, name_ref);
  pthread_mutex_unlock(&lock);
}

void Archive::add_attribute(AttributeRef attribute_ref, StringRef name_ref, StringRef description_ref, pallas_type_t type) {
  pthread_mutex_lock(&lock);
  definitions.add_attribute(attribute_ref, name_ref, description_ref, type);
  pthread_mutex_unlock(&lock);
}

void Archive::add_group(GroupRef group_ref, StringRef name, uint32_t number_of_members, const uint64_t* members, GroupType group_type, Paradigm paradigm) {
  pthread_mutex_lock(&lock);
  definitions.add_group(group_ref, name, group_type, paradigm, number_of_members, members);
  pthread_mutex_unlock(&lock);
}

void Archive::add_comm(CommRef comm_ref, StringRef name, GroupRef group, CommRef parent) {
  pthread_mutex_lock(&lock);
  definitions.add_comm(comm_ref, name, group, parent);
  pthread_mutex_unlock(&lock);
}

const String* Archive::get_string(StringRef string_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.get_string(string_ref);
  if (res == nullptr && global_archive)
    res = global_archive->get_string(string_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

const Region* Archive::get_region(RegionRef region_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.get_region(region_ref);
  if (res == nullptr && global_archive)
    res = global_archive->get_region(region_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

const Attribute* Archive::get_attribute(AttributeRef attribute_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.get_attribute(attribute_ref);
  if (res == nullptr && global_archive)
    res = global_archive->get_attributes(attribute_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

const Group* Archive::get_group(GroupRef group_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.get_group(group_ref);
  if (res == nullptr && global_archive)
    res = global_archive->get_group(group_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

const Comm* Archive::get_comm(CommRef comm_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.get_comm(comm_ref);
  if (res == nullptr && global_archive)
    res = global_archive->get_comm(comm_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

void Archive::define_location_group(ThreadId l_id, StringRef name, LocationGroupId parent) {
  pthread_mutex_lock(&lock);
  Location l = {.id = l_id, .name = name, .parent = parent};
  pallas_assert(l.id != PALLAS_THREAD_ID_INVALID);
  locations.push_back(l);
  pthread_mutex_unlock(&lock);
}

void Archive::define_location(ThreadId l_id, StringRef name, LocationGroupId parent) {
  pthread_mutex_lock(&lock);
  Location l = {.id = l_id, .name = name, .parent = parent};
  pallas_assert(l.id != PALLAS_THREAD_ID_INVALID);
  locations.push_back(l);
  pthread_mutex_unlock(&lock);
}

const LocationGroup* Archive::get_location_group(LocationGroupId location_group_id) const {
  if (global_archive)
    return global_archive->getLocationGroup(location_group_id);
  return nullptr;
}

const Location* Archive::get_location(ThreadId location_id) const {
  for (auto& l : locations) {
    if (l.id == location_id) {
      return &l;
    }
  }
  return nullptr;
}

const char* Archive::get_name() {
  return global_archive->get_string(global_archive->getLocationGroup(id)->name)->str;
}

void GlobalArchive::add_metadata(const std::string& key, const std::string& value) {
    pthread_mutex_lock(&lock);
    metadata[key] = value;
    pthread_mutex_unlock(&lock);
};
void Archive::add_metadata(const std::string& key, const std::string& value) {
    pthread_mutex_lock(&lock);
    metadata[key] = value;
    pthread_mutex_unlock(&lock);
};
void Archive::store(const ParameterHandler* parameter_handler) {
    store(dir_name, parameter_handler);
}
} /* namespace pallas*/

/********************** C Bindings **********************/
pallas::Archive* pallas_archive_new(const char* dir_name, pallas::LocationGroupId location_group) {
  return new pallas::Archive(dir_name, location_group);
}
void pallas_archive_delete(pallas::Archive* archive) {
  delete archive;
}

pallas::GlobalArchive* pallas_global_archive_new(const char* dirname, const char* trace_name) {
  return new pallas::GlobalArchive(dirname, trace_name);
}

void pallas_global_archive_delete(pallas::GlobalArchive* archive) {
  delete archive;
}

pallas::Thread* pallas_archive_get_thread(pallas::Archive* archive, pallas::ThreadId thread_id) {
  return archive->get_thread(thread_id);
};

pallas::Archive* pallas_global_archive_get_archive(pallas::GlobalArchive* archive, pallas::LocationGroupId archive_id) {
  return archive->getArchive(archive_id);
};

const pallas::LocationGroup* pallas_archive_get_location_group(pallas::GlobalArchive* archive, pallas::LocationGroupId location_group) {
  return archive->getLocationGroup(location_group);
};

const pallas::Archive* pallas_archive_get_archive_from_location(pallas::GlobalArchive* archive, pallas::ThreadId thread_id) {
  return archive->getArchiveFromLocation(thread_id);
}

const pallas::Location* pallas_archive_get_location(pallas::GlobalArchive* archive, pallas::ThreadId threadId) {
  return archive->getLocation(threadId);
}

void pallas_archive_register_string(pallas::Archive* archive, pallas::StringRef string_ref, const char* string) {
  archive->add_string(string_ref, string);
}
void pallas_archive_register_region(pallas::Archive* archive, pallas::RegionRef region_ref, pallas::StringRef string_ref) {
  archive->add_region(region_ref, string_ref);
}
void pallas_archive_register_attribute(pallas::Archive* archive,
                                       pallas::AttributeRef attribute_ref,
                                       pallas::StringRef name_ref,
                                       pallas::StringRef description_ref,
                                       pallas::pallas_type_t type) {
  archive->add_attribute(attribute_ref, name_ref, description_ref, type);
}
void pallas_archive_register_group(pallas::Archive* archive,
                                   pallas::GroupRef group_ref,
                                   pallas::StringRef name,
                                   pallas::GroupType group_type,
                                   pallas::Paradigm paradigm,
                                   uint32_t numberOfMembers,
                                   const uint64_t* members) {
    archive->add_group(group_ref, name, numberOfMembers, members, group_type, paradigm);
}
void pallas_archive_register_comm(pallas::Archive* archive, pallas::CommRef comm_ref, pallas::StringRef name, pallas::GroupRef group, pallas::CommRef parent) {
  archive->add_comm(comm_ref, name, group, parent);
}

void pallas_global_archive_register_string(pallas::GlobalArchive* archive, pallas::StringRef string_ref, const char* string) {
  archive->add_string(string_ref, string);
}
void pallas_global_archive_register_region(pallas::GlobalArchive* archive, pallas::RegionRef region_ref, pallas::StringRef string_ref) {
  archive->add_region(region_ref, string_ref);
}
void pallas_global_archive_register_attribute(pallas::GlobalArchive* archive,
                                              pallas::AttributeRef attribute_ref,
                                              pallas::StringRef name_ref,
                                              pallas::StringRef description_ref,
                                              pallas::pallas_type_t type) {
  archive->get_attributes(attribute_ref, name_ref, description_ref, type);
}
void pallas_global_archive_register_group(pallas::GlobalArchive* archive,
    pallas::GroupRef group_ref,
    pallas::StringRef name,
    pallas::GroupType group_type,
    pallas::Paradigm paradigm,
    uint32_t numberOfMembers,
    const uint64_t* members) {
  archive->add_group(group_ref, name, group_type, paradigm, numberOfMembers, members);
}
void pallas_global_archive_register_comm(pallas::GlobalArchive* archive, pallas::CommRef comm_ref, pallas::StringRef name, pallas::GroupRef group, pallas::CommRef parent) {
  archive->add_comm(comm_ref, name, group, parent);
}

extern void pallas_global_archive_define_location_group(pallas::GlobalArchive* archive, pallas::LocationGroupId id, pallas::StringRef name, pallas::LocationGroupId parent) {
  archive->define_location_group(id, name, parent);
};

extern void pallas_global_archive_define_location(pallas::GlobalArchive* archive, pallas::ThreadId id, pallas::StringRef name, pallas::LocationGroupId parent) {
  archive->define_location(id, name, parent);
};

extern void pallas_archive_define_location_group(pallas::Archive* archive, pallas::LocationGroupId id, pallas::StringRef name, pallas::LocationGroupId parent) {
  archive->define_location_group(id, name, parent);
};

extern void pallas_archive_define_location(pallas::Archive* archive, pallas::ThreadId id, pallas::StringRef name, pallas::LocationGroupId parent) {
  archive->define_location(id, name, parent);
};

const pallas::String* pallas_archive_get_string(pallas::Archive* archive, pallas::StringRef string_ref) {
  return archive->get_string(string_ref);
}
const pallas::Region* pallas_archive_get_region(pallas::Archive* archive, pallas::RegionRef region_ref) {
  return archive->get_region(region_ref);
}
int pallas_archive_get_nb_regions(PALLAS(Archive) * archive) {
    return archive->definitions.regions.size();
}
const pallas::Attribute* pallas_archive_get_attribute(pallas::Archive* archive, pallas::AttributeRef attribute_ref) {
  return archive->get_attribute(attribute_ref);
}
const pallas::Group* pallas_archive_get_group(pallas::Archive* archive, pallas::GroupRef group_ref) {
  return archive->get_group(group_ref);
}
const pallas::Comm* pallas_archive_get_communicator(pallas::Archive* archive, pallas::CommRef comm_ref) {
  return archive->get_comm(comm_ref);
}

void pallas_global_archive_add_metadata(pallas::GlobalArchive* archive, const char* key, const char * value) {
    archive->add_metadata(key, value);
}

void pallas_archive_add_metadata(pallas::GlobalArchive* archive, const char* key, const char * value) {
    archive->add_metadata(key, value);
}

const char *pallas_global_archive_get_metadata(pallas::GlobalArchive* archive, const char* key) {
    if (archive->metadata.contains(key)) {
        return archive->metadata[key].c_str();
    }
    return NULL;
}

const char *pallas_archive_get_metadata(pallas::Archive* archive, const char* key) {
    if (archive->metadata.contains(key)) {
        return archive->metadata[key].c_str();
    }
    return NULL;
}
/* -*-
  mode: c++;
  c-file-style: "k&r";
  c-basic-offset 2;
  tab-width 2 ;
  indent-tabs-mode nil
  -*- */
