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
    }
  }

  #if 0
  save_thread_copy(trace, archives, threads,
    strdup((
      std::string(base_dir_name) + "_dev1"
  ).c_str()));
  #endif

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // |         Synchronize Events           |
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  thread_token_map thread_event_lookup;

  uint32_t n_events_verified = 0;

  for (auto* t : threads) {
    uint32_t thread_n_events = t->nb_events;

    if (thread_n_events <= n_events_verified) {
      continue;
    }

    // thread has events that need to be synchronized
    for (uint32_t event_id = n_events_verified; event_id < thread_n_events; event_id++) {
      assert(t->events[event_id].event.record != pallas::PALLAS_EVENT_MAX_ID);
      thread_event_lookup[t->id][event_id] = event_id;
    }

    sync_events(threads, t, n_events_verified, thread_n_events, thread_event_lookup);

    // track updated event index
    n_events_verified = thread_n_events;
  }

  #if 0
  save_thread_copy(trace, archives, threads,
    strdup((
      std::string(base_dir_name) + "_dev2"
  ).c_str()));
  #endif

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // |   Synchronize Sequences and Loops    |
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  thread_token_map thread_seq_lookup;
  thread_token_map thread_loop_lookup;

  bool update_events = true;
  bool update_seqs = false;
  bool update_loops = false;

  int nb_cycles = 0;

  while (true) {

    std::cout << ">>Starting update loop #" << nb_cycles << std::endl;

    int number_of_swaps = 0;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |       Update Loop Definitions        |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    update_loop_tokens(threads,
                       thread_event_lookup, update_events,
                       thread_seq_lookup, update_seqs,
                       thread_loop_lookup, update_loops);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |      Synchronize Updated Loops       |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    uint32_t n_loops_verified = 0;

    for (auto* t : threads) {
      uint32_t thread_n_loops = t->nb_loops;

      if (thread_n_loops <= n_loops_verified) {
        continue;
      }

      for (uint32_t loop_id = n_loops_verified; loop_id < thread_n_loops; loop_id++) {
        // TODO: check necessity of this condition
        if (t->loops[loop_id].repeated_token.type == pallas::TypeInvalid) {
          continue;
        }
        thread_loop_lookup[t->id][loop_id] = loop_id;
      }

      number_of_swaps += sync_loops(threads, t, n_loops_verified, thread_n_loops, thread_loop_lookup);

      n_loops_verified = thread_n_loops;
    }

    update_loops = true;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |     Update Sequence Definitions      |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    update_sequence_tokens(threads,
                           thread_event_lookup, update_events,
                           thread_seq_lookup, update_seqs,
                           thread_loop_lookup, update_loops);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |    Synchronize Updated Sequences     |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    uint32_t n_seqs_verified = 0;

    for (auto* t : threads) {
      uint32_t thread_n_seqs = t->nb_sequences;

      if (thread_n_seqs <= n_seqs_verified) {
        continue;
      }

      for (uint32_t seq_id = n_seqs_verified; seq_id < thread_n_seqs; seq_id++) {
        // TODO: check necessity of this condition
        if (t->sequences[seq_id]->id == PALLAS_TOKEN_ID_INVALID) {
          continue;
        }
        thread_seq_lookup[t->id][seq_id] = seq_id;
      }

      number_of_swaps += sync_sequences(threads, t, n_seqs_verified, thread_n_seqs, thread_seq_lookup);

      n_seqs_verified = thread_n_seqs;
    }

    update_seqs = true;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |     Verify Further Updates Needed    |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    std::cout << "Number of sequence swaps performed = " << number_of_swaps << std::endl;

    if (number_of_swaps == 0) {
      break;
    }

    nb_cycles++;
  }

  for (auto* thread : threads) {
    thread->sequence_root = thread_seq_lookup[thread->id][0];
  }

  auto save_name = strdup((
    std::string(base_dir_name) + "_fin"
  ).c_str());

  save_thread_copy(trace, archives, threads, save_name);

  return EXIT_SUCCESS;
}
