//
// Created by Aaron Bushnell on 23.09.2025
//

#include <sys/types.h>
#include <unistd.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
// #include "pallas/pallas_dbg.h"
// #include "pallas/pallas_log.h"
// #include "pallas/pallas_parameter_handler.h"
// #include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"
// #include "pallas/pallas_write.h"

#define DEBUG_LEVEL 0
#define ENABLE_WRITE

bool event_cmp(pallas::Event *e_1, pallas::Event *e_2) {

  if (e_1->record != e_2->record) {
    return false;
  }

  if (memcmp(e_1->event_data, 
             e_2->event_data, 
             e_1->event_size) != 0) {
    return false;
  }

  return true;
}

void event_insert(pallas::EventSummary *es, 
                  pallas::Thread *t, uint32_t id) {
  
  size_t allocated_events = t->nb_allocated_events;
  if (id > allocated_events) {
    doubleMemorySpaceConstructor(t->events, t->nb_allocated_events);
    t->nb_events++;
  }

  t->events[id] = *es;
  t->events[id].id = id;
}

void event_insert_invalid(pallas::Thread *t, uint32_t id) {

  pallas::EventSummary inv_event = t->events[id];
  inv_event.event = pallas::Event {
    pallas::PALLAS_EVENT_MAX_ID,
    0,
    NULL
  };
}

int main(int argc, char** argv) {

  std::map<uint32_t, pallas::String> synced_strings;
  std::map<uint32_t, uint32_t> string_ref_lookup;
  uint32_t next_free_string_ref = 0;

  std::map<uint32_t, pallas::Region> synced_regions;
  std::map<uint32_t, uint32_t> region_ref_lookup;
  uint32_t next_free_region_ref = 0;

  char* trace_name = nullptr;

  if (argc < 2) {
    std::cout << "ERROR: Missing trace file" << std::endl;
    return EXIT_FAILURE;
  }

  trace_name = argv[1];
  pallas::GlobalArchive* trace = pallas_open_trace(trace_name);
  
  std::cout << "Pallas: Trace File Opened" << std::endl;

  // loop over StringRef -> String map in GlobalArchive Definition
  for (auto const& [string_ref, string]
    : trace->definitions.strings) {
    if (DEBUG_LEVEL > 0) {
      std::cout << "Pallas: string #" << string_ref;
      std::cout << " = '" << string.str << "'" << std::endl;
    }

    // loop over local synchronized StringRef -> String map
    bool contains_string = false;
    for (auto const& [synced_string_ref, synced_string]
      : synced_strings) {
      if (std::strcmp(string.str, synced_string.str) == 0) {
        contains_string = true;

        // record updated StringRef
        string_ref_lookup[string_ref] = (uint32_t) synced_string_ref;
      }
    }

    if (!contains_string) {
      // add new String to synchronized map
      auto& s = synced_strings[next_free_string_ref];
      s.string_ref = next_free_string_ref;
      s.length = string.length;
      s.str = (char*) std::calloc(sizeof(char), s.length);
      std::strncpy(s.str, string.str, s.length);

      // record new StringRef and increment id
      string_ref_lookup[string_ref] = (uint32_t) next_free_string_ref;
      next_free_string_ref++;
    }
  }
  
  // loop over RegionRef -> Region map in GlobalArchive Definition
  for (auto const& [region_ref, region]
    : trace->definitions.regions) {
    if (DEBUG_LEVEL > 0) {
      std::cout << "Pallas: region #" << region_ref;
      std::cout << " = (" << region.string_ref << ")'";
      std::cout << trace->definitions.strings[region.string_ref].str;
      std::cout << "'" << std::endl;
    }

    // loop over local synchronized RegionRef -> Region map
    bool contains_region = false;
    for (auto const& [synced_region_ref, synced_region]
      : synced_regions) {
      if (string_ref_lookup[region.string_ref] == synced_region.string_ref) {
        contains_region = true;

        // record updated RegionRef
        region_ref_lookup[region_ref] = (uint32_t) synced_region_ref;
      }
    }

    if (!contains_region) {
      // add new Region to synchronized map
      auto& r = synced_regions[next_free_region_ref];
      r.region_ref = next_free_region_ref;
      r.string_ref = string_ref_lookup[region.string_ref];

      // record new RegionRef and increment id
      region_ref_lookup[region_ref] = (uint32_t) next_free_region_ref;
      next_free_region_ref++;
    }

  }

  // add synchronized string + region maps to Definition
  trace->definitions.strings = std::move(synced_strings);
  trace->definitions.regions = std::move(synced_regions);

  // update other Definition elements' StringRefs
  // attributes
  for (auto& [attribute_ref, attribute]
    : trace->definitions.attributes) {
    attribute.name = string_ref_lookup[attribute.name];
    attribute.description = string_ref_lookup[attribute.description];
  }

  // groups
  for (auto& [group_ref, group]
    : trace->definitions.groups) {
    group.name = string_ref_lookup[group.name];
  }

  // comms
  for (auto& [comm_ref, comm]
    : trace->definitions.comms) {
    comm.name = string_ref_lookup[comm.name];
  }

  // update StringRefs in location_groups, locations and events
  for (auto& lg : trace->location_groups) {
    lg.name = string_ref_lookup[lg.name];

    auto* a = trace->getArchive(lg.id);
    for (auto& loc : a->locations) {
      loc.name = string_ref_lookup[loc.name];

      auto* t = a->getThread(loc.id);

      size_t num_of_events = t->nb_events;
      for (size_t i = 0; i < num_of_events; i++) {
        pallas::EventSummary summary = t->events[i];
        pallas::Event event = summary.event;
        pallas::Record record = event.record;
        if (record == pallas::PALLAS_EVENT_ENTER || record == pallas::PALLAS_EVENT_LEAVE) {

          pallas::RegionRef ref;
          memcpy(&ref, event.event_data, sizeof(pallas::RegionRef));
          pallas::RegionRef new_ref = region_ref_lookup[ref];
          memcpy(t->events[i].event.event_data, &new_ref, sizeof(pallas::RegionRef));
        }
      }
      a->freeThread(loc.id);
    }
    trace->freeArchive(lg.id);
  }

  std::vector<std::map<uint32_t, uint32_t>> thread_token_lookup;

  for (auto& lg : trace->location_groups) {

    auto* a = trace->getArchive(lg.id);

    uint32_t n_events_verified = 0;

    for (auto& loc : a->locations) {
      auto* t = a->getThread(loc.id);
      uint32_t max_events_t = t->nb_events;

      if (max_events_t > n_events_verified) {
        for (uint32_t event_idx = n_events_verified; event_idx < max_events_t; event_idx++) {
          pallas::EventSummary src_event = t->events[event_idx];

          for (auto& loc2 : a->locations) {
            if (loc2.id != loc.id) {
              auto* t2 = a->getThread(loc2.id);
              bool found_match = false;
              pallas::EventSummary candidate_event = t2->events[event_idx];
              if (event_cmp(&src_event.event,
                            &candidate_event.event)) {
                continue;
              } else {

                uint32_t max_events_t2 = t2->nb_events;
                for (uint32_t event2_idx = event_idx; event2_idx < max_events_t2; event2_idx++) {
                  candidate_event = t2->events[event2_idx];
                  if (event_cmp(&src_event.event,
                                &candidate_event.event)) {
                    pallas::EventSummary swapped_event = t2->events[event_idx];
                    event_insert(&candidate_event, t2, event_idx);
                    event_insert(&swapped_event, t2, event2_idx);
                    found_match = true;
                    break;
                  }
                }

                pallas::EventSummary swapped_event = t2->events[event_idx];
                event_insert_invalid(t2, event_idx);
                event_insert(&swapped_event, t2, t2->nb_events);
              }
              a->freeThread(loc.id);

            }
          }
        }
      }

      n_events_verified = max_events_t;

      a->freeThread(loc.id);
    }
    trace->freeArchive(lg.id);
  }


  // write updated trace
  auto newDirName = strdup((
      std::string(trace->dir_name) + "_synchronized"
  ).c_str());

  for (auto& lg : trace->location_groups) {
    auto* a = trace->getArchive(lg.id);
    for (auto& loc : a->locations) {
      auto* t = a->getThread(loc.id);
      // doubleMemorySpaceConstructor

      #ifdef ENABLE_WRITE
      t->store(newDirName, trace->parameter_handler, true);
      #endif

      a->freeThread(loc.id);
    }
    #ifdef ENABLE_WRITE
    a->store(newDirName, trace->parameter_handler);
    #endif

    a->dir_name = nullptr;
    trace->freeArchive(lg.id);
  }

  // write updated GlobalArchive
  #ifdef ENABLE_WRITE
  trace->store(newDirName, trace->parameter_handler);
  #endif


  // ~~~~~~~~~~~
  //
  // DEBUG Block
  //
  // ~~~~~~~~~~~

  if (DEBUG_LEVEL > 1) {
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Synced string list" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    for (auto const& [string_ref, string] 
      : synced_strings) {
      std::cout << "Pallas: string #" << string.string_ref;
      std::cout << " = '" << string.str << "'" << std::endl;
    }
  }

  if (DEBUG_LEVEL > 1) {
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Definition string list" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    for (auto const& [string_ref, string] 
      : trace->definitions.strings) {
      std::cout << "Pallas: string #" << string.string_ref;
      std::cout << " = '" << string.str << "'" << std::endl;
    }
  }

  if (DEBUG_LEVEL > 1) {
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Synced string ref lookup " << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    for (auto const& [old_ref, new_ref] 
      : string_ref_lookup) {
      std::cout << "Mapped ref #" << old_ref << " -> ";
      std::cout << new_ref << std::endl;
    }
  }

  if (DEBUG_LEVEL > 1) {
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Synced region list " << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    for (auto const& [region_ref, region]
      : synced_regions) {
      std::cout << "Pallas: region #" << region_ref;
      std::cout << " = (" << region.string_ref << ")'";
      std::cout << synced_strings[region.string_ref].str;
      std::cout << "'" << std::endl;
    }
  }

  if (DEBUG_LEVEL > 1) {
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Definition region list " << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    for (auto const& [region_ref, region]
      : trace->definitions.regions) {
      std::cout << "Pallas: region #" << region_ref;
      std::cout << " = (" << region.string_ref << ")'";
      std::cout << trace->definitions.strings[region.string_ref].str;
      std::cout << "'" << std::endl;
    }
  }

  if (DEBUG_LEVEL > 1) {
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Synced region ref lookup " << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    for (auto const& [old_ref, new_ref] 
      : region_ref_lookup) {
      std::cout << "Mapped ref #" << old_ref << " -> ";
      std::cout << new_ref << std::endl;
    }
  }

  return EXIT_SUCCESS;
}
