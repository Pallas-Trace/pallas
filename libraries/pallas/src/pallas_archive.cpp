/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_archive.h"
#include "pallas/pallas.h"
#include "pallas/pallas_dbg.h"
#include "pallas/pallas_write.h"

namespace pallas {

/**
 * Getter for a String from its id.
 * @returns First String matching the given pallas::StringRef, nullptr if it doesn't have a match.
 */
const String* Definition::getString(StringRef string_ref) const {
  /* TODO: race condition here ? when adding a region, there may be a realloc so we may have to hold the mutex */
  for (auto& s : strings) {
    if (s.string_ref == string_ref) {
      return &s;
    }
  }
  return nullptr;
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
  strings.push_back(s);

  pallas_log(DebugLevel::Verbose, "Register string #%zu{.ref=%x, .length=%d, .str='%s'}\n", strings.size() - 1,
          s.string_ref, s.length, s.str);
}

/**
 * Getter for a Region from its id.
 * @returns First Region matching the given pallas::RegionRef, nullptr if it doesn't have a match.
 */
const Region* Definition::getRegion(RegionRef region_ref) const {
  /* TODO: race condition here ? when adding a region, there may be a realloc so we may have to hold the mutex */
  for (auto& r : regions) {
    if (r.region_ref == region_ref) {
      return &r;
    }
  }
  return nullptr;
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
  regions.push_back(r);

  pallas_log(DebugLevel::Verbose, "Register region #%zu{.ref=%x, .str=%d}\n", regions.size() - 1, r.region_ref,
          r.string_ref);
}

/**
 * Getter for a Attribute from its id.
 * @returns First Attribute matching the given pallas::AttributeRef, nullptr if it doesn't have a match.
 */
const Attribute* Definition::getAttribute(AttributeRef attribute_ref) const {
  /* TODO: race condition here ? when adding a region, there may be a realloc so we may have to hold the mutex */
  for (auto& a : attributes) {
    if (a.attribute_ref == attribute_ref) {
      return &a;
    }
  }
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
  attributes.push_back(a);

  pallas_log(DebugLevel::Verbose, "Register attribute #%zu{.ref=%x, .name=%d, .description=%d, .type=%d}\n",
          attributes.size() - 1, a.attribute_ref, a.name, a.description, a.type);
}

/**
 * Getter for a String from its id.
 * @returns First String matching the given pallas::StringRef in this archive, or in the global_archive if it doesn't
 * have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const String* Archive::getString(StringRef string_ref) const {
  auto res = definitions.getString(string_ref);
  return (res) ? res : (global_archive) ? global_archive->getString(string_ref) : nullptr;
}

/**
 * Getter for a Region from its id.
 * @returns First Region matching the given pallas::RegionRef in this archive, or in the global_archive if it doesn't
 * have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const Region* Archive::getRegion(RegionRef region_ref) const {
  auto res = definitions.getRegion(region_ref);
  return (res) ? res : (global_archive) ? global_archive->getRegion(region_ref) : nullptr;
}

/**
 * Getter for a Attribute from its id.
 * @returns First Attribute matching the given pallas::AttributeRef in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const Attribute* Archive::getAttribute(AttributeRef attribute_ref) const {
  auto res = definitions.getAttribute(attribute_ref);
  return (res) ? res : (global_archive) ? global_archive->getAttribute(attribute_ref) : nullptr;
}

/**
 * Getter for a Thread from its id.
 * @returns First Thread matching the given pallas::ThreadId, or nullptr if it doesn't have a match.
 */
Thread* Archive::getThread(ThreadId thread_id) const {
  for (int i = 0; i < nb_threads; i++) {
    if (threads[i]->id == thread_id)
      return threads[i];
  }
  return nullptr;
}

/**
 * Getter for a LocationGroup from its id.
 * @returns First LocationGroup matching the given pallas::LocationGroupId in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const LocationGroup* Archive::getLocationGroup(LocationGroupId location_group_id) const {
  for (auto& lc : location_groups) {
    if (lc.id == location_group_id) {
      return &lc;
    }
  }
  return (global_archive) ? global_archive->getLocationGroup(location_group_id) : nullptr;
  // The global_archive is the only one for which the global_archive field is nullptr
}

/**
 * Getter for a Location from its id.
 * @returns First Location matching the given pallas::ThreadId in this archive, or in the global_archive if it
 * doesn't have a match, or nullptr if it doesn't have a match in the global_archive.
 */
const Location* Archive::getLocation(ThreadId location_id) const {
  for (auto& l : locations) {
    if (l.id == location_id) {
      return &l;
    }
  }
  return (global_archive) ? global_archive->getLocation(location_id) : nullptr;
  // The global_archive is the only one for which the global_archive field is nullptr
}

/**
 * Creates a new String and adds it to that Archive.
 * Error if the given pallas::StringRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
void Archive::addString(StringRef string_ref, const char* string) {
  pthread_mutex_lock(&lock);
  definitions.addString(string_ref, string);
  pthread_mutex_unlock(&lock);
}

/**
 * Creates a new Region and adds it to that Archive.
 * Error if the given pallas::RegionRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
void Archive::addRegion(RegionRef region_ref, StringRef name_ref) {
  pthread_mutex_lock(&lock);
  definitions.addRegion(region_ref, name_ref);
  pthread_mutex_unlock(&lock);
}

/**
 * Creates a new Attribute and adds it to that Archive.
 * Error if the given pallas::AttributeRef is already in use.
 * Locks and unlocks the mutex for that operation.
 */
void Archive::addAttribute(AttributeRef attribute_ref, StringRef name_ref, StringRef description_ref, pallas_type_t type) {
  pthread_mutex_lock(&lock);
  definitions.addAttribute(attribute_ref, name_ref, description_ref, type);
  pthread_mutex_unlock(&lock);
}
} /* namespace pallas*/

/********************** C Bindings **********************/
pallas::Archive* pallas_archive_new() {
  return new pallas::Archive();
};

pallas::Thread* pallas_archive_get_thread(pallas::Archive* archive, pallas::ThreadId thread_id) {
  return archive->getThread(thread_id);
};

const pallas::LocationGroup* pallas_archive_get_location_group(pallas::Archive* archive, pallas::LocationGroupId location_group) {
  return archive->getLocationGroup(location_group);
};

const pallas::Location* pallas_archive_get_location(pallas::Archive* archive, pallas::ThreadId threadId) {
  return archive->getLocation(threadId);
}
void pallas_archive_register_string(pallas::Archive* archive, pallas::StringRef string_ref, const char* string) {
  archive->addString(string_ref, string);
}
void pallas_archive_register_region(pallas::Archive* archive, pallas::RegionRef region_ref, pallas::StringRef string_ref) {
  archive->addRegion(region_ref, string_ref);
}
void pallas_archive_register_attribute(pallas::Archive* archive,
                                    pallas::AttributeRef attribute_ref,
                                    pallas::StringRef name_ref,
                                    pallas::StringRef description_ref,
                                    pallas::pallas_type_t type) {
  archive->addAttribute(attribute_ref, name_ref, description_ref, type);
}
const pallas::String* pallas_archive_get_string(pallas::Archive* archive, pallas::StringRef string_ref) {
  return archive->getString(string_ref);
}
const pallas::Region* pallas_archive_get_region(pallas::Archive* archive, pallas::RegionRef region_ref) {
  return archive->getRegion(region_ref);
}
const pallas::Attribute* pallas_archive_get_attribute(pallas::Archive* archive, pallas::AttributeRef attribute_ref) {
  return archive->getAttribute(attribute_ref);
}

/* -*-
  mode: c++;
  c-file-style: "k&r";
  c-basic-offset 2;
  tab-width 2 ;
  indent-tabs-mode nil
  -*- */