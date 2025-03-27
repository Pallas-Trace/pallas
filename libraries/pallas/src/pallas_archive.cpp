/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_archive.h"
#include "pallas/pallas.h"
#include "pallas/pallas_dbg.h"
#include "pallas/pallas_write.h"
#include "pallas/pallas_log.h"

namespace pallas {
/**
 * Getter for a String from its id.
 * @returns First String matching the given pallas::StringRef, nullptr if it doesn't have a match.
 */
const String* Definition::getString(StringRef string_ref) const {
  if (strings.count(string_ref) > 0)
    return &strings.at(string_ref);
  else return nullptr;
}

/**
 * Creates a new String and adds it to that definition. Error if the given pallas::StringRef is already in use.
 */
void Definition::addString(StringRef string_ref, const char* string) {
  if (getString(string_ref)) {
    pallas_error("Given string_ref was already in use.\n");
  }

  auto s = String();
  s.string_ref = string_ref;
  s.length = strlen(string) + 1;
  s.str = new char[s.length];
  strncpy(s.str, string, s.length);
  strings[string_ref] = s;

  pallas_log(DebugLevel::Verbose, "Register string #%zu{.ref=%d, .length=%d, .str='%s'}\n", strings.size() - 1,
          s.string_ref, s.length, s.str);
}

/**
 * Getter for a Region from its id.
 * @returns First Region matching the given pallas::RegionRef, nullptr if it doesn't have a match.
 */
const Region* Definition::getRegion(RegionRef region_ref) const {
  if (regions.count(region_ref) > 0)
    return &regions.at(region_ref);
  else return nullptr;
}

/**
 * Creates a new Region and adds it to that definition. Error if the given pallas::RegionRef is already in use.
 */
void Definition::addRegion(RegionRef region_ref, StringRef string_ref) {
  if (getRegion(region_ref)) {
    pallas_error("Given region_ref was already in use.\n");
  }

  auto r = Region();
  r.region_ref = region_ref;
  r.string_ref = string_ref;
  regions[region_ref] = r;

  pallas_log(DebugLevel::Verbose, "Register region #%zu{.ref=%d, .str=%d}\n", regions.size() - 1, r.region_ref,
          r.string_ref);
}

/**
 * Getter for a Attribute from its id.
 * @returns First Attribute matching the given pallas::AttributeRef, nullptr if it doesn't have a match.
 */
const Attribute* Definition::getAttribute(AttributeRef attribute_ref) const {
  if (attributes.count(attribute_ref) > 0)
    return &attributes.at(attribute_ref);
  return nullptr;
}

/**
 * Creates a new Attribute and adds it to that definition. Error if the given pallas::AttributeRef is already in use.
 */
void Definition::addAttribute(AttributeRef attribute_ref,
                              StringRef name_ref,
                              StringRef description_ref,
                              pallas_type_t type) {
  if (getAttribute(attribute_ref)) {
    pallas_error("Given attribute_ref was already in use.\n");
  }
  auto a = Attribute();
  a.attribute_ref = attribute_ref;
  a.name = name_ref;
  a.description = description_ref;
  a.type = type;
  attributes[attribute_ref] = (a);

  pallas_log(DebugLevel::Verbose, "Register attribute #%zu{.ref=%d, .name=%d, .description=%d, .type=%d}\n",
          attributes.size() - 1, a.attribute_ref, a.name, a.description, a.type);
}

  /**
   * Getter for a Group from its id.
   * @returns First Group matching the given pallas::GroupRef, nullptr if it doesn't have a match.
   */
  const Group* Definition::getGroup(GroupRef group_ref) const {
    if (groups.count(group_ref) > 0)
      return &groups.at(group_ref);
    else return nullptr;
  }

  /**
   * Creates a new Group and adds it to that definition. Error if the given pallas::GroupRef is already in use.
   */
  void Definition::addGroup(GroupRef group_ref, StringRef name, uint32_t number_of_members, const uint64_t* members) {
    if (getGroup(group_ref)) {
      pallas_error("Given group_ref was already in use.\n");
    }

    auto g = Group();
    g.group_ref = group_ref;
    g.name = name;
    g.numberOfMembers = number_of_members;
    g.members = new uint64_t[number_of_members];
    for(uint32_t i = 0; i<number_of_members; i++)
	  g.members[i] = members[i];
    
    groups[group_ref] = g;

    pallas_log(DebugLevel::Verbose, "Register group #%zu{.ref=%d, .str=%d, .nbMembers=%d}\n", groups.size() - 1, g.group_ref, g.name, g.numberOfMembers);
  }


  /**
   * Getter for a Comm from its id.
   * @returns First Comm matching the given pallas::CommRef, nullptr if it doesn't have a match.
   */
  const Comm* Definition::getComm(CommRef comm_ref) const {
    if (comms.count(comm_ref) > 0)
      return &comms.at(comm_ref);
    else return nullptr;
  }

  /**
   * Creates a new Comm and adds it to that definition. Error if the given pallas::CommRef is already in use.
   */
  void Definition::addComm(CommRef comm_ref, StringRef name, GroupRef group, CommRef parent) {
    if (getComm(comm_ref)) {
      pallas_error("Given comm_ref was already in use.\n");
    }

    auto c = Comm();
    c.comm_ref = comm_ref;
    c.name = name;
    c.group = group;
    c.parent = parent;

    comms[comm_ref] = c;

    pallas_log(DebugLevel::Verbose, "Register comm #%zu{.ref=%d, .str=%d, .group=%d, .parent=%d}\n", comms.size() - 1, c.comm_ref, c.name, c.group, c.parent);
  }

/**
 * Getter for a String from its id.
 * @returns First String matching the given pallas::StringRef in this archive, or in the global_archive if it doesn't
 * have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const String* GlobalArchive::getString(StringRef string_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.getString(string_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

/**
 * Getter for a Region from its id.
 * @returns First Region matching the given pallas::RegionRef in this archive, or in the global_archive if it doesn't
 * have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const Region* GlobalArchive::getRegion(RegionRef region_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.getRegion(region_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

/**
 * Getter for a Attribute from its id.
 * @returns First Attribute matching the given pallas::AttributeRef in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const Attribute* GlobalArchive::getAttribute(AttributeRef attribute_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.getAttribute(attribute_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

/**
 * Getter for a Group from its id.
 * @returns First Group matching the given pallas::GroupRef in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const Group* GlobalArchive::getGroup(GroupRef group_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.getGroup(group_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

/**
 * Getter for a Comm from its id.
 * @returns First Comm matching the given pallas::CommRef in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const Comm* GlobalArchive::getComm(CommRef comm_ref) {
  pthread_mutex_lock(&lock);
  auto res = definitions.getComm(comm_ref);
  pthread_mutex_unlock(&lock);
  return res;
}

/**
 * Getter for a LocationGroup from its id.
 * @returns First LocationGroup matching the given pallas::LocationGroupId in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const LocationGroup* GlobalArchive::getLocationGroup(LocationGroupId location_group_id) const {
  for (auto& lc : location_groups) {
    if (lc.id == location_group_id) {
      return &lc;
    }
  }
  return nullptr;
  // The global_archive is the only one for which the global_archive field is nullptr
}

/**
 * Getter for a Location from its id.
 * @returns First Location matching the given pallas::ThreadId in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const Location* GlobalArchive::getLocation(ThreadId location_id) const {
  for (auto& l : locations) {
    if (l.id == location_id) {
      return &l;
    }
  }
  return nullptr;
  // The global_archive is the only one for which the global_archive field is nullptr
}

struct Archive* GlobalArchive::getArchiveFromLocation(ThreadId location_id) const {
  for(int i =0; i<nb_archives; i++) {
    if(archive_list[i]->getThread(location_id))
      return archive_list[i];
  }
  return nullptr;
}

const Archive*
pallas_archive_get_archive_from_location(PALLAS(GlobalArchive) * archive, PALLAS(ThreadId) thread_id) {
  return archive->getArchiveFromLocation(thread_id);
}

/**
 * Creates a new String and adds it to that Archive.
 * Error if the given pallas::StringRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
void GlobalArchive::addString(StringRef string_ref, const char* string) {
  pthread_mutex_lock(&lock);
  definitions.addString(string_ref, string);
  pthread_mutex_unlock(&lock);
}

/**
 * Creates a new Region and adds it to that Archive.
 * Error if the given pallas::RegionRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
void GlobalArchive::addRegion(RegionRef region_ref, StringRef name_ref) {
  pthread_mutex_lock(&lock);
  definitions.addRegion(region_ref, name_ref);
  pthread_mutex_unlock(&lock);
}

/**
 * Creates a new Attribute and adds it to that Archive.
 * Error if the given pallas::AttributeRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
void GlobalArchive::addAttribute(AttributeRef attribute_ref, StringRef name_ref, StringRef description_ref, pallas_type_t type) {
  pthread_mutex_lock(&lock);
  definitions.addAttribute(attribute_ref, name_ref, description_ref, type);
  pthread_mutex_unlock(&lock);
}

/**
 * Creates a new Group and adds it to that definition. Error if the
 * given pallas::GroupRef is already in use.
 */
void GlobalArchive::addGroup(GroupRef group_ref, StringRef name, uint32_t number_of_members, const uint64_t* members) {
  pthread_mutex_lock(&lock);
  definitions.addGroup(group_ref, name, number_of_members, members);
  pthread_mutex_unlock(&lock);
}

/**
 * Creates a new Comm and adds it to that definition. Error if the
 * given pallas::CommRef is already in use.
 */
void GlobalArchive::addComm(CommRef comm_ref, StringRef name, GroupRef group, CommRef parent) {
  pthread_mutex_lock(&lock);
  definitions.addComm(comm_ref, name, group, parent);
  pthread_mutex_unlock(&lock);
}

GlobalArchive::~GlobalArchive() {
  delete[] dir_name;
  delete[] trace_name;
  delete[] fullpath;
  for (size_t i = 0; i < nb_archives; i ++) {
    delete archive_list[i];
  }
  delete[] archive_list;
};

Archive::~Archive() {
  free(dir_name);
  delete[] trace_name;
  for (size_t i = 0; i < nb_threads; i ++) {
    delete threads[i];
  }
  delete[] threads;
}

const char* Archive::getName() const {
  return global_archive->getString(global_archive->getLocation(id)->name)->str;
}

} /* namespace pallas*/

/********************** C Bindings **********************/
pallas::Archive* pallas_archive_new(const char* dir_name, const char* trace_name, pallas::LocationGroupId location_group) {
  return new pallas::Archive(dir_name, trace_name, location_group);
}

pallas::GlobalArchive * pallas_global_archive_new(const char* dirname, const char* trace_name) {
  return new pallas::GlobalArchive (dirname, trace_name);
}

pallas::Thread* pallas_archive_get_thread(pallas::Archive * archive, pallas::ThreadId thread_id) {
  return archive->getThread(thread_id);
};

const pallas::LocationGroup* pallas_archive_get_location_group(pallas::GlobalArchive* archive, pallas::LocationGroupId location_group) {
  return archive->getLocationGroup(location_group);
};

const pallas::Location* pallas_archive_get_location(pallas::GlobalArchive* archive, pallas::ThreadId threadId) {
  return archive->getLocation(threadId);
}
void pallas_archive_register_string(pallas::GlobalArchive* archive, pallas::StringRef string_ref, const char* string) {
  archive->addString(string_ref, string);
}
void pallas_archive_register_region(pallas::GlobalArchive* archive, pallas::RegionRef region_ref, pallas::StringRef string_ref) {
  archive->addRegion(region_ref, string_ref);
}
void pallas_archive_register_attribute(pallas::GlobalArchive* archive,
                                    pallas::AttributeRef attribute_ref,
                                    pallas::StringRef name_ref,
                                    pallas::StringRef description_ref,
                                    pallas::pallas_type_t type) {
  archive->addAttribute(attribute_ref, name_ref, description_ref, type);
}
void pallas_archive_register_group(pallas::GlobalArchive* archive,
				   pallas::GroupRef group_ref,
				   pallas::StringRef name,
				   uint32_t numberOfMembers,
				   const uint64_t* members) {
  archive->addGroup(group_ref, name, numberOfMembers, members);
}
void pallas_archive_register_comm(pallas::GlobalArchive* archive,
				  pallas::CommRef comm_ref,
				  pallas::StringRef name,
				  pallas::GroupRef group,
				  pallas::CommRef parent) {
  archive->addComm(comm_ref, name, group, parent);
}

const pallas::String* pallas_archive_get_string(pallas::GlobalArchive* archive, pallas::StringRef string_ref) {
  return archive->getString(string_ref);
}
const pallas::Region* pallas_archive_get_region(pallas::GlobalArchive* archive, pallas::RegionRef region_ref) {
  return archive->getRegion(region_ref);
}
const pallas::Attribute* pallas_archive_get_attribute(pallas::GlobalArchive* archive, pallas::AttributeRef attribute_ref) {
  return archive->getAttribute(attribute_ref);
}
const pallas::Group* pallas_archive_get_group(pallas::GlobalArchive* archive,
					      pallas::GroupRef group_ref) {
  return archive->getGroup(group_ref);
}
const pallas::Comm* pallas_archive_get_communicator(pallas::GlobalArchive* archive,
							    pallas::CommRef comm_ref) {
  return archive->getComm(comm_ref);
}
/* -*-
  mode: c++;
  c-file-style: "k&r";
  c-basic-offset 2;
  tab-width 2 ;
  indent-tabs-mode nil
  -*- */
