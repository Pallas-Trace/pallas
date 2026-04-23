//
// Created by Aaron Bushnell on 23.09.2025
//

#include <sys/types.h>
#include <unistd.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/utils/pallas_hash.h"
#include "pallas/utils/pallas_storage.h"

#define DEBUG_LEVEL 0
#define ENABLE_WRITE

typedef std::map<uint32_t, uint32_t> token_map;
typedef std::map<uint32_t, token_map> thread_token_map;

void map_set(thread_token_map& fwd, thread_token_map& rev, uint32_t thread_id, uint32_t from, uint32_t to) {
  fwd[thread_id][from] = to;
  rev[thread_id][to] = from;
}

void map_swap(thread_token_map& fwd, thread_token_map& rev, uint32_t thread_id, uint32_t id1, uint32_t id2) {
  bool count1 = rev[thread_id].count(id1);
  bool count2 = rev[thread_id].count(id2);
  uint32_t owner1 = count1 ? rev[thread_id][id1] : id1;
  uint32_t owner2 = count2 ? rev[thread_id][id2] : id2;

  if (count1) {
    fwd[thread_id][owner1] = id2;
    rev[thread_id][id2] = owner1;
  }
  else {
    rev[thread_id].erase(id2);
  }

  if (count2) {
    fwd[thread_id][owner2] = id1;
    rev[thread_id][id1] = owner2;
  }
  else {
    rev[thread_id].erase(id1);
  }
}

uint32_t map_eval(thread_token_map& map, uint32_t thread_id, uint32_t in_id) {
  auto thread_it = map.find(thread_id);
  if (thread_it == map.end()) return in_id;

  uint32_t out_id = in_id;
  auto& token_map = thread_it->second;
  uint32_t current = in_id;
  while (true) {
    auto it = token_map.find(current);
    if (it == token_map.end() || it->second == current) return current;
    current = it->second;
  }
}

bool event_cmp(pallas::Event& e1, pallas::Event& e2) {
  if (e1.data.record != e2.data.record) {
    return false;
  }
  size_t data_size = e1.data.event_size - sizeof(e1.data.record) - sizeof(e1.data.event_size);
  if (memcmp(e1.data.event_data,e2.data.event_data,data_size) != 0) {
    return false;
  }
  return true;
}

void event_insert(pallas::Event& e, pallas::Thread *t, uint32_t id) {
  while (id >= t->nb_allocated_events) {
    doubleMemorySpaceConstructor(t->events, t->nb_allocated_events);
  }
  t->events[id] = std::move(e);
  if (id >= t->nb_events) {
    t->nb_events++;
    t->events[id].id = id;
  }
}

void event_override_invalid(pallas::Thread *t, uint32_t id) {
  t->events[id].id = PALLAS_TOKEN_ID_INVALID;
  t->events[id].data = pallas::EventData {
    pallas::PALLAS_EVENT_MAX_ID,
    0,
    {}
  };
  t->events[id].timestamps = NULL;
  t->events[id].nb_occurrences = 0;
  t->events[id].attribute_buffer = NULL;
  t->events[id].attribute_buffer_size = 0;
  t->events[id].attribute_pos = 0;
}

void event_swap(pallas::Thread *t, uint32_t src_id, uint32_t swap_id) {
  std::swap(t->events[src_id], t->events[swap_id]);
  t->events[src_id].id = src_id;
  t->events[swap_id].id = swap_id;
}

uint32_t find_matching_event(pallas::Event& src_event, pallas::Thread *t) {
  uint32_t t_n_events = t->nb_events;
  for (uint32_t e2_idx = src_event.id; e2_idx < t_n_events; e2_idx++) {
    pallas::Event& cand_event = t->events[e2_idx];
    if (event_cmp(src_event, cand_event)) {
      return cand_event.id;
    }
  }
  // if no match found return src_event id
  return src_event.id;
}

int sync_events(std::vector<pallas::Thread*>& threads,
                pallas::Thread *t,
                uint32_t start_id,
                uint32_t end_id,
                thread_token_map& event_map,
                thread_token_map& event_rev) {
  for (auto* t2 : threads) {
    // ignore identical threads
    if (t2->id == t->id) {
      continue;
    }

    for (uint32_t event_id = start_id; event_id < end_id; event_id++) {
      pallas::Event& src_event  = t->events[event_id];
      pallas::Event& cand_event = t2->events[event_id];
      bool found_match = false;

      assert(src_event.data.record != pallas::PALLAS_EVENT_MAX_ID);

      // check if already synchronized
      if (event_cmp(src_event,cand_event)) {
        found_match = true;

      // try to find other match somewhere
      } else {
        uint32_t match_id = find_matching_event(src_event, t2);
        if (match_id != event_id) {
          event_swap(t2, event_id, match_id);
          found_match = true;
          map_swap(event_map, event_rev, t2->id, event_id, match_id);
        }
      }

      // if no match found insert placeholder
      if (!found_match && cand_event.data.record != pallas::PALLAS_EVENT_MAX_ID) {
        uint32_t swap_id = t2->nb_events;
        pallas::Event swap_event = std::move(t2->events[event_id]);
        event_insert(swap_event, t2, swap_id);
        event_override_invalid(t2, event_id);

        uint32_t prev_owner = event_rev[t2->id].count(event_id) ? event_rev[t2->id][event_id] : event_id;
        map_set(event_map, event_rev, t2->id, prev_owner, swap_id);
        event_rev[t2->id].erase(event_id);
      }
    }
  }
  return 0;
}

bool loop_cmp(pallas::Loop& l1, pallas::Loop& l2) {
  if (l1.repeated_token.type != l2.repeated_token.type) {
    return false;
  }
  if (l1.repeated_token.id != l2.repeated_token.id) {
    return false;
  }
  if (l1.nb_iterations != l2.nb_iterations) {
    return false;
  }
  return true;
}

void loop_insert(pallas::Loop& l,
                 pallas::Thread *t, uint32_t id) {
  while (id >= t->nb_allocated_loops) {
    doubleMemorySpaceConstructor(t->loops, t->nb_allocated_loops);
  }
  t->loops[id] = std::move(l);
  if (id >= t->nb_loops) {
    t->nb_loops++;
    t->loops[id].self_id.id = id;
  }
}

void loop_override_invalid(pallas::Thread *t, uint32_t id) {
  t->loops[id].repeated_token.type = pallas::TypeInvalid;
  t->loops[id].repeated_token.id = PALLAS_TOKEN_ID_INVALID;
  t->loops[id].self_id.type = pallas::TypeInvalid;
  t->loops[id].self_id.id = PALLAS_TOKEN_ID_INVALID;
  t->loops[id].nb_iterations = 0;
  t->loops[id].nb_occurrences = 0;
}

void loop_swap(pallas::Thread *t,
               uint32_t src_id, uint32_t swap_id) {
  std::swap(t->loops[src_id], t->loops[swap_id]);
  t->loops[src_id].self_id.id = src_id;
  t->loops[swap_id].self_id.id = swap_id;
}

uint32_t find_matching_loop(pallas::Loop& src_loop,
                            pallas::Thread *t) {
  uint32_t t_n_loops = t->nb_loops;
  for (uint32_t l2_idx = src_loop.self_id.id; l2_idx < t_n_loops; l2_idx++) {
    pallas::Loop& cand_loop = t->loops[l2_idx];
    if (loop_cmp(src_loop, cand_loop)) {
      // std::cout << "found swap match!" << std::endl;
      return cand_loop.self_id.id;
    }
  }
  // if no match found return src_event id
  return src_loop.self_id.id;
}

void update_loop_tokens(std::vector<pallas::Thread*>& threads,
                        thread_token_map& event_map,
                        bool update_events,
                        thread_token_map& seq_map,
                        bool update_seqs,
                        thread_token_map& loop_map,
                        bool update_loops) {
  for (auto* t : threads) {
    for (uint32_t loop_id = 0; loop_id < t->nb_loops; loop_id++) {
      pallas::Loop& loop = t->loops[loop_id];

      if (loop.self_id.type == pallas::TypeInvalid) {
        continue;
      }

      auto& token = loop.repeated_token;
      if (token.type == pallas::TypeEvent && update_events) {
        token.id = map_eval(event_map, t->id, token.id);
      }
      if (token.type == pallas::TypeSequence && update_seqs) {
        token.id = map_eval(seq_map, t->id, token.id);
      }
      if (token.type == pallas::TypeLoop && update_loops) {
        token.id = map_eval(loop_map, t->id, token.id);
      }
    }
  }
}

int sync_loops(std::vector<pallas::Thread*>& threads,
               pallas::Thread *t,
               uint32_t start_id,
               uint32_t end_id,
               thread_token_map& loop_map,
               thread_token_map& loop_rev) {
  int number_of_swaps = 0;
  for (auto* t2 : threads) {
    // ignore identical threads
    if (t2->id == t->id) {
      continue;
    }

    for (uint32_t loop_id = start_id; loop_id < end_id; loop_id++) {
      // make sure t2 has enough allocated loops
      if (t2->nb_allocated_loops <= loop_id) {
        doubleMemorySpaceConstructor(t2->loops, t2->nb_allocated_loops);
      }
      pallas::Loop& src_loop = t->loops[loop_id];
      pallas::Loop& cand_loop = t2->loops[loop_id];
      bool found_match = false;

      // check if src invalid
      if (src_loop.self_id.type == pallas::TypeInvalid) {
        continue;
      }

      bool cand_is_invalid = (cand_loop.self_id.type == pallas::TypeInvalid);

      // check if already synchronized
      if (!cand_is_invalid && loop_cmp(src_loop, cand_loop)) {
        found_match = true;

      // try to find other match somewhere
      } else {
        uint32_t match_id = find_matching_loop(src_loop, t2);
        if (match_id != loop_id) {
          loop_swap(t2, loop_id, match_id);
          found_match = true;
          map_swap(loop_map, loop_rev, t2->id, loop_id, match_id);
          number_of_swaps++;
        }
      }

      // if no match found insert placeholder
      if (!found_match) {
        if (!cand_is_invalid) {
          uint32_t swap_id = t2->nb_loops;
          pallas::Loop disp_loop = std::move(t2->loops[loop_id]);
          loop_insert(disp_loop, t2, swap_id);
          loop_override_invalid(t2, loop_id);

          uint32_t prev_owner = loop_rev[t2->id].count(loop_id) ? loop_rev[t2->id][loop_id] : loop_id;
          map_set(loop_map, loop_rev, t2->id, prev_owner, swap_id);
          loop_rev[t2->id].erase(loop_id);

          number_of_swaps++;
        } else {
          if (t2->nb_loops <= loop_id) {
            t2->nb_loops = loop_id + 1;
          }
        }
      }
    }
  }
  return number_of_swaps;
}

bool seq_cmp(pallas::Sequence& seq1, pallas::Sequence& seq2) {
  if (seq1.hash != seq2.hash) {
    // NOTE: this was causing issues
    // return false;
  }
  if (seq1.tokens.size() != seq2.tokens.size()) {
    return false;
  }
  for (int i = 0; i < seq1.tokens.size(); i++) {
    if (seq1.tokens[i].id != seq2.tokens[i].id) {
      return false;
    }
  }
  return true;
}

void seq_insert(pallas::Sequence& seq, pallas::Thread *t, uint32_t id) {
  while (id >= t->nb_allocated_sequences) {
    doubleMemorySpaceConstructor(t->sequences, t->nb_allocated_sequences);
  }
  t->sequences[id] = std::move(seq);
  if (id >= t->nb_sequences) {
    t->nb_sequences++;
    t->sequences[id].id.id = id;
  }
}

void seq_override_invalid(pallas::Thread *t, uint32_t id) {
  // NOTE: consider re-init sequence
  // i.e. t->sequences[id] = pallas::Sequence{};
  t->sequences[id].id = pallas::Token();
  t->sequences[id].durations = NULL;
  t->sequences[id].exclusive_durations = NULL;
  t->sequences[id].timestamps = NULL;
  t->sequences[id].hash = 0;
  t->sequences[id].tokens.clear();
  t->sequences[id].type = pallas::SEQUENCE_BLOCK;
}

void seq_swap(pallas::Thread *t, uint32_t src_id, uint32_t swap_id) {
  std::swap(t->sequences[src_id], t->sequences[swap_id]);
  t->sequences[src_id].id.id = src_id;
  t->sequences[swap_id].id.id = swap_id;
}

uint32_t find_matching_seq(pallas::Sequence& src_seq, pallas::Thread *t) {
  uint32_t t_n_seqs = t->nb_sequences;
  for (uint32_t s2_idx = src_seq.id.id; s2_idx < t_n_seqs; s2_idx++) {
    pallas::Sequence& cand_seq = t->sequences[s2_idx];
    if (seq_cmp(src_seq, cand_seq)) {
      // std::cout << "found swap match!" << std::endl;
      return cand_seq.id.id;
    }
  }
  // if no match found return src_event id
  return src_seq.id.id;
}

void update_sequence_tokens(std::vector<pallas::Thread*>& threads,
                            thread_token_map& event_map,
                            bool update_events,
                            thread_token_map& seq_map,
                            bool update_seqs,
                            thread_token_map& loop_map,
                            bool update_loops) {
  for (auto* t : threads) {
    for (uint32_t seq_id = 0; seq_id < t->nb_sequences; seq_id++) {
      pallas::Sequence& seq = t->sequences[seq_id];

      if (seq.id.type == pallas::TypeInvalid) {
        continue;
      }

      for (auto& token : seq.tokens) {
        if (token.type == pallas::TypeEvent && update_events) {
          token.id = map_eval(event_map, t->id, token.id);
        }
        if (token.type == pallas::TypeSequence && update_seqs) {
          token.id = map_eval(seq_map, t->id, token.id);
        }
        if (token.type == pallas::TypeLoop && update_loops) {
          token.id = map_eval(loop_map, t->id, token.id);
        }
      }
      seq.hash = pallas::hash32(reinterpret_cast<const byte*>(seq.tokens.data()),
                                  seq.tokens.size() * sizeof(pallas::Token), SEED);
    }
  }
}

int sync_sequences(std::vector<pallas::Thread*>& threads,
                   pallas::Thread *t,
                   uint32_t start_id,
                   uint32_t end_id,
                   thread_token_map& seq_map,
                   thread_token_map& seq_rev) {
  int number_of_swaps = 0;
  for (auto* t2 : threads) {
    // ignore identical threads
    if (t2->id == t->id) {
      continue;
    }

    for (uint32_t seq_id = start_id; seq_id < end_id; seq_id++) {
      // make sure t2 has enough allocated sequences
      if (t2->nb_allocated_sequences <= seq_id) {
          doubleMemorySpaceConstructor(t2->sequences, t2->nb_allocated_sequences);
      }
      pallas::Sequence& src_seq = t->sequences[seq_id];
      pallas::Sequence& cand_seq = t2->sequences[seq_id];
      bool found_match = false;

      // check if src invalid
      // NOTE: possibly update handling?
      if (src_seq.id.type == pallas::TypeInvalid) {
        continue;
      }

      bool cand_is_invalid = (cand_seq.id.type == pallas::TypeInvalid);

      // check if already synchronized
      if (!cand_is_invalid && seq_cmp(src_seq, cand_seq)) {
        found_match = true;

      // try to find other match somewhere
      } else {
        uint32_t match_id = find_matching_seq(src_seq, t2);
        if (match_id != seq_id) {
          seq_swap(t2, seq_id, match_id);
          found_match = true;
          map_swap(seq_map, seq_rev, t2->id, seq_id, match_id);
          number_of_swaps++;
        }
      }

      // if no match found insert placeholder
      if (!found_match) {
        if (!cand_is_invalid) {
          uint32_t swap_id = t2->nb_sequences;
          pallas::Sequence disp_seq = std::move(t2->sequences[seq_id]);
          seq_insert(disp_seq, t2, swap_id);
          seq_override_invalid(t2, seq_id);

          uint32_t prev_owner = seq_rev[t2->id].count(seq_id) ? seq_rev[t2->id][seq_id] : seq_id;
          map_set(seq_map, seq_rev, t2->id, prev_owner, swap_id);
          seq_rev[t2->id].erase(seq_id);

          number_of_swaps++;
        } else {
          if (t2->nb_sequences <= seq_id) {
            t2->nb_sequences = seq_id + 1;
          }
        }
      }
    }
  }
  return number_of_swaps;
}

void save_thread_copy(pallas::GlobalArchive *trace,
                      std::vector<pallas::Archive*>& archives,
                      std::vector<pallas::Thread*>& threads,
                      char *save_dir_name) {
  for (auto* t : threads) {
    t->store(save_dir_name, trace->parameter_handler, true);
  }

  for (auto* a: archives) {
    a->store(save_dir_name, trace->parameter_handler);
  }

  trace->store(save_dir_name, trace->parameter_handler);
}

void verify_event_sync(std::vector<pallas::Thread*>& threads, uint32_t max_event_id) {
  bool all_ok = true;

  for (uint32_t event_id = 0; event_id < max_event_id; event_id++) {
    pallas::Event* reference = nullptr;
    pallas::Thread* reference_thread = nullptr;
    bool slot_has_valid = false;

    for (auto* t : threads) {
      if (event_id >= t->nb_events) continue;

      pallas::Event& e = t->events[event_id];
      bool is_invalid = (e.data.record == pallas::PALLAS_EVENT_MAX_ID);

      if (!is_invalid) {
        slot_has_valid = true;
        if (reference == nullptr) {
          reference = &e;
          reference_thread = t;
        } else {
          if (!event_cmp(*reference, e)) {
            fprintf(stderr,
              "[VERIFY] MISMATCH at event_id=%u: "
              "thread %u (record=%u) != thread %u (record=%u)\n",
              event_id,
              reference_thread->id, reference->data.record,
              t->id, e.data.record);
            all_ok = false;
          }
        }
      }
    }

    if (!slot_has_valid) {
      bool any_thread_covers = false;
      for (auto* t : threads) {
        if (event_id < t->nb_events) { any_thread_covers = true; break; }
      }
      if (any_thread_covers) {
        fprintf(stderr,
          "[VERIFY] ALL-INVALID slot at event_id=%u "
          "(every thread covering this id has an invalid)\n",
          event_id);
        all_ok = false;
      }
    }
  }

  for (auto* t : threads) {
    for (uint32_t event_id = 0; event_id < t->nb_events; event_id++) {
      pallas::Event& e = t->events[event_id];
      bool is_invalid = (e.data.record == pallas::PALLAS_EVENT_MAX_ID);
      if (!is_invalid && e.id != event_id) {
        fprintf(stderr,
          "[VERIFY] ID FIELD MISMATCH: thread %u events[%u].id = %u (expected %u)\n",
          t->id, event_id, e.id, event_id);
        all_ok = false;
      }
    }
  }

  if (all_ok)
    fprintf(stderr, "[VERIFY] event sync OK — %u ids checked across %zu threads\n",
            max_event_id, threads.size());
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

  auto base_dir_name = strdup((
      std::string(trace->dir_name)
  ).c_str());

  auto base_trace_name = strdup((
    std::string(base_dir_name) + "/" + std::string(trace->trace_name)
  ).c_str());

  auto temp_dir_name = strdup((
    std::string(base_dir_name) + "_temp"
  ).c_str());

  auto temp_trace_name = strdup((
    std::string(temp_dir_name) + "/" + std::string(trace->trace_name)
  ).c_str());
  
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
      s.str = (char*) std::calloc(sizeof(char), s.length + 1);
      std::memcpy(s.str, string.str, s.length);
      s.str[s.length] = '\0';

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

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // |      Update GlobalArchive StringRefs       |
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // | Update LocationGroups Locations and Events |
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  std::vector<pallas::Archive*> archives;
  std::map<uint32_t, uint32_t> archive_id_lookup;

  std::vector<pallas::Thread*> threads;
  std::map<uint32_t, uint32_t> thread_id_lookup;

  for (auto& lg : trace->location_groups) {
    lg.name = string_ref_lookup[lg.name];
    auto* a = trace->getArchive(lg.id);
    archive_id_lookup[lg.id] = archives.size();
    archives.push_back(a);
    for (auto& loc : a->locations) {
      loc.name = string_ref_lookup[loc.name];
      auto* t = a->getThread(loc.id);
      thread_id_lookup[loc.id] = threads.size();
      threads.push_back(t);

      size_t num_of_events = t->nb_events;
      for (size_t i = 0; i < num_of_events; i++) {
        pallas::Event event = t->events[i];
        pallas::EventData data = event.data;
        pallas::Record record = data.record;

        if (record == pallas::PALLAS_EVENT_ENTER || record == pallas::PALLAS_EVENT_LEAVE) {
          pallas::RegionRef ref;
          memcpy(&ref, data.event_data, sizeof(pallas::RegionRef));
          pallas::RegionRef new_ref = region_ref_lookup[ref];
          memcpy(t->events[i].data.event_data, &new_ref, sizeof(pallas::RegionRef));
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

  thread_token_map thread_event_map;
  thread_token_map thread_event_rev;

  uint32_t n_events_verified = 0;

  // initialize event_map to identity for original events
  for (auto* t : threads) {
    for (uint32_t i = 0; i < t->nb_events; i++) {
      if (t->events[i].data.record != pallas::PALLAS_EVENT_MAX_ID) {
        map_set(thread_event_map, thread_event_rev, t->id, i, i);
      }
    }
  }

  for (auto* t : threads) {
    uint32_t thread_n_events = t->nb_events;
    if (thread_n_events <= n_events_verified) {
      continue;
    }
    // else: -> thread has events that need to be synchronized

    // pre-fill other threads with invalids to match nb_events
    for (auto* t2 : threads) {
      if (t2->id == t->id) continue;
      for (uint32_t event_id = t2->nb_events; event_id < thread_n_events; event_id++) {
        while (event_id >= t2->nb_allocated_events)
          doubleMemorySpaceConstructor(t2->events, t2->nb_allocated_events);
        event_override_invalid(t2, event_id);
        t2->nb_events = event_id + 1;
      }
    }

    sync_events(threads, t, n_events_verified, thread_n_events, thread_event_map, thread_event_rev);
    // track updated event index
    n_events_verified = thread_n_events;

    // compact extra danging invalids
    for (auto* t2 : threads) {
      if (t2->id == t->id) continue;
      uint32_t new_nb = t2->nb_events;
      while (new_nb > n_events_verified && t2->events[new_nb - 1].data.record == pallas::PALLAS_EVENT_MAX_ID) {
        new_nb--;
      }
      t2->nb_events = new_nb;
    }
  }

  uint32_t max_id = 0;
  for (auto* t : threads) {
    if (t->nb_events > max_id) {
      max_id = t->nb_events;
    }
    verify_event_sync(threads, max_id);
  }

  uint32_t expected_nb = threads[0]->nb_events;
  for (auto* t : threads) {
    assert(t->nb_events == expected_nb && "nb_events mismatch after sync");
  }

  for (auto* t : threads) {
    for (uint32_t i = t->nb_events; i < t->nb_allocated_events; i++) {
      pallas::Event& e = t->events[i];
      bool is_unoccupied = (e.data.record == pallas::PALLAS_EVENT_MAX_ID ||
                            (e.nb_occurrences == 0 && e.timestamps == nullptr));
      assert(is_unoccupied && "real event stranded beyond nb_events");
    }
  }

  for (auto* t : threads) {
    std::unordered_set<uint32_t> seen_targets;
    for (auto& [src, dst] : thread_event_map[t->id]) {
      if (dst >= t->nb_events) continue; // skip stale dangling entries
      if (seen_targets.count(dst)) {
        fprintf(stderr, "[VERIFY] duplicate map target %u in thread %u\n", dst, t->id);
      }
      seen_targets.insert(dst);
    }
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

  thread_token_map thread_loop_map;
  thread_token_map thread_loop_rev;

  thread_token_map thread_seq_map;
  thread_token_map thread_seq_rev;

  bool update_events = true;
  bool update_seqs   = false;
  bool update_loops  = false;

  // initialize loop_map and seq_map to identity for original tokens
  for (auto* t : threads) {
    for (uint32_t i = 0; i < t->nb_loops; i++) {
      if (t->loops[i].self_id.type != pallas::TypeInvalid) {
        map_set(thread_loop_map, thread_loop_rev, t->id, i, i);
      }
    }
    for (uint32_t i = 0; i < t->nb_sequences; i++) {
      if (t->sequences[i].id.type != pallas::TypeInvalid) {
        map_set(thread_seq_map, thread_seq_rev, t->id, i, i);
      }
    }
  }

  int nb_cycles = 0;

  while (true) {
    std::cout << ">>Starting update loop #" << nb_cycles << std::endl;

    int number_of_swaps = 0;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |       Update Loop Definitions        |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    update_loop_tokens(
      threads,
      thread_event_map, update_events,
      thread_seq_map, update_seqs,
      thread_loop_map, update_loops
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |      Synchronize Updated Loops       |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    uint32_t n_loops_verified = 0;

    for (auto* t : threads) {
      uint32_t thread_n_loops = t->nb_loops;
      if (thread_n_loops <= n_loops_verified) {
        continue;
      }
      // else: -> thread has loops that need to be synchronized

      // pre-fill other threads with invalids to match nb_loops
      for (auto* t2 : threads) {
        if (t2->id == t->id) continue;
        for (uint32_t loop_id = t2->nb_loops; loop_id < thread_n_loops; loop_id++) {
          while (loop_id >= t2->nb_allocated_loops)
            doubleMemorySpaceConstructor(t2->loops, t2->nb_allocated_loops);
          loop_override_invalid(t2, loop_id);
          t2->nb_loops = loop_id + 1;
        }
      }

      number_of_swaps += sync_loops(threads, t, n_loops_verified, thread_n_loops, thread_loop_map, thread_loop_rev);
      n_loops_verified = thread_n_loops;

      // compact extra danging invalids
      for (auto* t2 : threads) {
        if (t2->id == t->id) continue;
        uint32_t new_nb = t2->nb_loops;
        while (new_nb > n_loops_verified &&
               t2->loops[new_nb - 1].self_id.type == pallas::TypeInvalid) {
          new_nb--;
        }
        t2->nb_loops = new_nb;
      }
    }

    update_loops = true;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |     Update Sequence Definitions      |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    update_sequence_tokens(
      threads,
      thread_event_map, update_events,
      thread_seq_map, update_seqs,
      thread_loop_map, update_loops
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |    Synchronize Updated Sequences     |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    uint32_t n_seqs_verified = 0;

    for (auto* t : threads) {
      uint32_t thread_n_seqs = t->nb_sequences;
      if (thread_n_seqs <= n_seqs_verified) {
        continue;
      }
      // else: -> thread has sequences that need to be synchronized

      // pre-fill other threads with invalids to match nb_sequences
      for (auto* t2 : threads) {
        if (t2->id == t->id) continue;
        for (uint32_t seq_id = t2->nb_sequences; seq_id < thread_n_seqs; seq_id++) {
          while (seq_id >= t2->nb_allocated_sequences)
            doubleMemorySpaceConstructor(t2->sequences, t2->nb_allocated_sequences);
          seq_override_invalid(t2, seq_id);
          t2->nb_sequences = seq_id + 1;
        }
      }

      number_of_swaps += sync_sequences(threads, t, n_seqs_verified, thread_n_seqs, thread_seq_map, thread_seq_rev);
      n_seqs_verified = thread_n_seqs;

      // compact extra danging invalids
      for (auto* t2 : threads) {
        if (t2->id == t->id) continue;
        uint32_t new_nb = t2->nb_sequences;
        while (new_nb > n_seqs_verified &&
               t2->sequences[new_nb - 1].id.type == pallas::TypeInvalid) {
          new_nb--;
        }
        t2->nb_sequences = new_nb;
      }
    }

    update_seqs = true;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |    Verify Further Updates Needed     |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    std::cout << "Number of sequence swaps performed = " << number_of_swaps << std::endl;

    if (number_of_swaps == 0) {
      break;
    }

    nb_cycles++;
  }

  for (auto* thread : threads) {
    thread->sequence_root = map_eval(thread_seq_map, thread->id, 0);
  }

  auto save_name = strdup((
    std::string(base_dir_name) + "_fin"
  ).c_str());

  save_thread_copy(trace, archives, threads, save_name);

  return EXIT_SUCCESS;
}
