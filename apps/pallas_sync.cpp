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
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/utils/pallas_hash.h"
#include "pallas/utils/pallas_storage.h"

#define DEBUG_LEVEL 0
#define ENABLE_WRITE

typedef std::map<uint32_t, uint32_t> token_map;
typedef std::map<uint32_t, token_map> thread_token_map;

typedef std::map<uint32_t, pallas::Token> token_lookup;
typedef std::map<uint32_t, std::vector<pallas::Token>> token_vector_lookup;

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
  auto it = thread_it->second.find(in_id);
  return (it != thread_it->second.end()) ? it->second : in_id;
}

uint32_t map_owner(const thread_token_map& rev, uint32_t thread_id, uint32_t current_id) {
  auto t_it = rev.find(thread_id);
  if (t_it == rev.end()) return current_id;
  auto id_it = t_it->second.find(current_id);
  return (id_it != t_it->second.end()) ? id_it->second : current_id;
}

uint32_t get_event_phys_id(pallas::Thread* t, uint32_t logi_id) {
  if (logi_id >= t->event_id_map.size()) return PALLAS_INDEX_INVALID;
  return t->event_id_map[logi_id];
}

void event_ensure_map_size(pallas::Thread* t, uint32_t logi_id) {
  if (logi_id >= t->event_id_map.size()) {
    t->event_id_map.resize(logi_id + 1, PALLAS_INDEX_INVALID);
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

void event_swap(pallas::Thread *t, uint32_t src_logi_id, uint32_t swap_logi_id) {
  event_ensure_map_size(t, src_logi_id);
  event_ensure_map_size(t, swap_logi_id);

  uint32_t src_phys_id = t->event_id_map[src_logi_id];
  uint32_t swap_phys_id = t->event_id_map[swap_logi_id];
  
  std::swap(t->event_id_map[src_logi_id], t->event_id_map[swap_logi_id]);

  if (src_phys_id != PALLAS_INDEX_INVALID) {
        t->events[src_phys_id].id = swap_logi_id;
    }
  if (swap_phys_id != PALLAS_INDEX_INVALID) {
      t->events[swap_phys_id].id = src_logi_id;
  }
}

void event_displace(pallas::Thread* t, uint32_t logi_id) {
  uint32_t phys_id = get_event_phys_id(t, logi_id);
  if (phys_id == PALLAS_INDEX_INVALID) return;
  uint32_t new_logi_id = t->event_id_map.size();
  event_ensure_map_size(t, new_logi_id);
  t->event_id_map[new_logi_id] = phys_id;
  t->events[phys_id].id = new_logi_id;
  t->event_id_map[logi_id] = PALLAS_INDEX_INVALID;
}

uint32_t find_matching_event(pallas::Event& src_event, pallas::Thread *t) {
  for (uint32_t logi_id = src_event.id; logi_id < t->event_id_map.size(); logi_id++) {
    uint32_t phys_id = t->event_id_map[logi_id];
    if (phys_id != PALLAS_INDEX_INVALID && event_cmp(src_event, t->events[phys_id])) {
      return logi_id;
    }
  }
  return PALLAS_INDEX_INVALID;
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

    if (t2->event_id_map.size() < end_id) {
      t2->event_id_map.resize(end_id, PALLAS_INDEX_INVALID);
    }

    for (uint32_t event_id = start_id; event_id < end_id; event_id++) {
      uint32_t src_phys_id = get_event_phys_id(t, event_id);
      assert(src_phys_id != PALLAS_INDEX_INVALID);
      pallas::Event& src_event  = t->events[src_phys_id];

      uint32_t cand_phys_id = get_event_phys_id(t2, event_id);
      bool found_match = false;

      // check if already synchronized
      if (cand_phys_id != PALLAS_INDEX_INVALID && event_cmp(src_event, t2->events[cand_phys_id])) {
        found_match = true;

      // try to find other match somewhere
      } else {
        uint32_t match_id = find_matching_event(src_event, t2);
        if (match_id != PALLAS_INDEX_INVALID) {
          event_swap(t2, event_id, match_id);
          found_match = true;
          map_swap(event_map, event_rev, t2->id, event_id, match_id);
        }
      }

      // if no match found insert placeholder
      if (!found_match && cand_phys_id != PALLAS_INDEX_INVALID) {
        event_displace(t2, event_id);
        uint32_t prev_owner = event_rev[t2->id].count(event_id) ? event_rev[t2->id][event_id] : event_id;
        uint32_t new_logi_id = t2->event_id_map.size();
        map_set(event_map, event_rev, t2->id, prev_owner, new_logi_id);
        event_rev[t2->id].erase(event_id);
      }
    }
  }
  return 0;
}

static uint32_t get_loop_phys_id(pallas::Thread* t, uint32_t logi_id) {
  if (logi_id >= t->loop_id_map.size()) return PALLAS_INDEX_INVALID;
  return t->loop_id_map[logi_id];
}

static void loop_ensure_map_size(pallas::Thread* t, uint32_t logi_id) {
  if (logi_id >= t->loop_id_map.size()) {
    t->loop_id_map.resize(logi_id + 1, PALLAS_INDEX_INVALID);
  }
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

void loop_swap(pallas::Thread *t, uint32_t src_logi_id, uint32_t swap_logi_id) {
  loop_ensure_map_size(t, src_logi_id);
  loop_ensure_map_size(t, swap_logi_id);

  uint32_t src_phys_id = t->loop_id_map[src_logi_id];
  uint32_t swap_phys_id = t->loop_id_map[swap_logi_id];

  std::swap(t->loop_id_map[src_logi_id], t->loop_id_map[swap_logi_id]);

  if (src_phys_id != PALLAS_INDEX_INVALID) {
    t->loops[src_phys_id].self_id = PALLAS_LOOP_ID(swap_logi_id);
  }
  if (swap_phys_id != PALLAS_INDEX_INVALID) {
    t->loops[swap_phys_id].self_id = PALLAS_LOOP_ID(src_logi_id);
  }
}

void loop_displace(pallas::Thread* t, uint32_t logi_id) {
  uint32_t phys_id = get_loop_phys_id(t, logi_id);
  if (phys_id == PALLAS_INDEX_INVALID) return;
  uint32_t new_logi_id = t->loop_id_map.size();
  loop_ensure_map_size(t, new_logi_id);
  t->loop_id_map[new_logi_id] = phys_id;
  t->loops[phys_id].self_id = PALLAS_LOOP_ID(new_logi_id);
  t->loop_id_map[logi_id] = PALLAS_INDEX_INVALID;
}

uint32_t find_matching_loop(pallas::Loop& src_loop, pallas::Thread *t) {
  for (uint32_t logi_id = src_loop.self_id.id; logi_id < t->loop_id_map.size(); logi_id++) {
    uint32_t phys_id = t->loop_id_map[logi_id];
    if (phys_id != PALLAS_INDEX_INVALID && loop_cmp(src_loop, t->loops[phys_id]))
      return logi_id;
  }
  return PALLAS_INDEX_INVALID;
}

void update_loop_tokens(std::vector<pallas::Thread*>& threads,
                        thread_token_map& event_map,
                        bool update_events,
                        thread_token_map& seq_map,
                        bool update_seqs,
                        thread_token_map& loop_map,
                        bool update_loops,
                        thread_token_map& loop_rev,
                        const std::map<uint32_t, token_lookup>& loop_base_tokens) {
  for (auto* t : threads) {
    for (uint32_t logi_id = 0; logi_id < t->loop_id_map.size(); logi_id++) {
      uint32_t phys_id = t->loop_id_map[logi_id];
      if (phys_id == PALLAS_INDEX_INVALID) {
        continue;
      }
      pallas::Loop& loop = t->loops[phys_id];

      uint32_t owner = map_owner(loop_rev, t->id, logi_id);
      auto thread_it = loop_base_tokens.find(t->id);
      if (thread_it == loop_base_tokens.end()) continue;
      auto owner_it = thread_it->second.find(owner);
      if (owner_it == thread_it->second.end()) continue;
      pallas::Token token = loop_base_tokens.at(t->id).at(owner);

      if (token.type == pallas::TypeEvent && update_events) {
        token.id = map_eval(event_map, t->id, token.id);
      }
      if (token.type == pallas::TypeSequence && update_seqs) {
        token.id = map_eval(seq_map, t->id, token.id);
      }
      if (token.type == pallas::TypeLoop && update_loops) {
        token.id = map_eval(loop_map, t->id, token.id);
      }

      loop.repeated_token = token;
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
      uint32_t src_phys_id = get_loop_phys_id(t, loop_id);
      // check if src invalid
      if (src_phys_id == PALLAS_INDEX_INVALID) {
        continue;
      }
      pallas::Loop& src_loop = t->loops[src_phys_id];

      uint32_t cand_phys_id = get_loop_phys_id(t2, loop_id);
      bool found_match = false;

      // check if already synchronized
      if (cand_phys_id != PALLAS_INDEX_INVALID && loop_cmp(src_loop, t2->loops[cand_phys_id])) {
        found_match = true;

      // try to find other match somewhere
      } else {
        uint32_t match_id = find_matching_loop(src_loop, t2);
        if (match_id != PALLAS_INDEX_INVALID) {
          loop_swap(t2, loop_id, match_id);
          found_match = true;
          map_swap(loop_map, loop_rev, t2->id, loop_id, match_id);
          number_of_swaps++;
        }
      }

      // if no match found insert placeholder
      if (!found_match && cand_phys_id != PALLAS_INDEX_INVALID) {
        loop_displace(t2, loop_id);
        uint32_t prev_owner = loop_rev[t2->id].count(loop_id) ? loop_rev[t2->id][loop_id] : loop_id;
        uint32_t new_logi_id = t2->loop_id_map.size();
        map_set(loop_map, loop_rev, t2->id, prev_owner, new_logi_id);
        loop_rev[t2->id].erase(loop_id);
        number_of_swaps++;
      }
    }
  }
  return number_of_swaps;
}

static uint32_t get_seq_phys_id(pallas::Thread* t, uint32_t logi_id) {
  if (logi_id >= t->sequence_id_map.size()) return PALLAS_INDEX_INVALID;
  return t->sequence_id_map[logi_id];
}

static void seq_ensure_map_size(pallas::Thread* t, uint32_t logi_id) {
  if (logi_id >= t->sequence_id_map.size()) {
    t->sequence_id_map.resize(logi_id + 1, PALLAS_INDEX_INVALID);
  }
}

bool seq_cmp(pallas::Sequence& seq1, pallas::Sequence& seq2) {
  if (seq1.hash != seq2.hash) {
    // NOTE: this was causing issues
    // return false;
  }
  if (seq1.tokens.size() != seq2.tokens.size()) {
    return false;
  }
  for (size_t i = 0; i < seq1.tokens.size(); i++) {
    if (seq1.tokens[i].type != seq2.tokens[i].type) {
      return false;
    }
    if (seq1.tokens[i].id != seq2.tokens[i].id) {
      return false;
    }
  }
  return true;
}

void seq_swap(pallas::Thread *t, uint32_t src_logi_id, uint32_t swap_logi_id) {
  seq_ensure_map_size(t, src_logi_id);
  seq_ensure_map_size(t, swap_logi_id);

  uint32_t src_phys_id = get_seq_phys_id(t, src_logi_id);
  uint32_t swap_phys_id = get_seq_phys_id(t, swap_logi_id);

  std::swap(t->sequence_id_map[src_logi_id], t->sequence_id_map[swap_logi_id]);

  if (src_phys_id != PALLAS_INDEX_INVALID) {
    t->sequences[src_phys_id].id = PALLAS_SEQUENCE_ID(swap_logi_id);
  }
  if (swap_phys_id != PALLAS_INDEX_INVALID) {
    t->sequences[swap_phys_id].id = PALLAS_SEQUENCE_ID(src_logi_id);
  }
}

void seq_displace(pallas::Thread* t, uint32_t logi_id) {
    uint32_t phys_id = get_seq_phys_id(t, logi_id);
    if (phys_id == PALLAS_INDEX_INVALID) return;
    uint32_t new_logi_id = t->sequence_id_map.size();
    seq_ensure_map_size(t, new_logi_id);
    t->sequence_id_map[new_logi_id] = phys_id;
    t->sequences[phys_id].id = PALLAS_SEQUENCE_ID(new_logi_id);
    t->sequence_id_map[logi_id] = PALLAS_INDEX_INVALID;
}

uint32_t find_matching_seq(pallas::Sequence& src_seq, pallas::Thread *t) {
  for (uint32_t logi_id = src_seq.id.id; logi_id < t->sequence_id_map.size(); logi_id++) {
    uint32_t phys_id = t->sequence_id_map[logi_id];
    if (phys_id != PALLAS_INDEX_INVALID && seq_cmp(src_seq, t->sequences[phys_id])) {
      return logi_id;
    }
  }
  return PALLAS_INDEX_INVALID;
}

void update_sequence_tokens(std::vector<pallas::Thread*>& threads,
                            thread_token_map& event_map,
                            bool update_events,
                            thread_token_map& seq_map,
                            bool update_seqs,
                            thread_token_map& loop_map,
                            bool update_loops,
                            thread_token_map& seq_rev,
                            const std::map<uint32_t, token_vector_lookup>& seq_base_tokens) {
  for (auto* t : threads) {
    for (uint32_t logi_id = 0; logi_id < t->sequence_id_map.size(); logi_id++) {
      uint32_t phys_id = t->sequence_id_map[logi_id];
      if (phys_id == PALLAS_INDEX_INVALID) {
        continue;
      }
      pallas::Sequence& seq = t->sequences[phys_id];

      uint32_t owner = map_owner(seq_rev, t->id, logi_id);
      auto thread_it = seq_base_tokens.find(t->id);
      if (thread_it == seq_base_tokens.end()) continue;
      auto owner_it = thread_it->second.find(owner);
      if (owner_it == thread_it->second.end()) continue;
      const auto& base_tokens = seq_base_tokens.at(t->id).at(owner);

      seq.tokens.clear();
      seq.tokens.reserve(base_tokens.size());

      for (const auto& base_token : base_tokens) {
        pallas::Token token = base_token;

        if (token.type == pallas::TypeEvent && update_events) {
          token.id = map_eval(event_map, t->id, token.id);
        }
        if (token.type == pallas::TypeSequence && update_seqs) {
          token.id = map_eval(seq_map, t->id, token.id);
        }
        if (token.type == pallas::TypeLoop && update_loops) {
          token.id = map_eval(loop_map, t->id, token.id);
        }

        seq.tokens.push_back(token);
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
      uint32_t src_phys_id = get_seq_phys_id(t, seq_id);
      // check if src invalid
      if (src_phys_id == PALLAS_INDEX_INVALID) {
        continue;
      }
      pallas::Sequence& src_seq = t->sequences[src_phys_id];

      uint32_t cand_phys_id = get_seq_phys_id(t2, seq_id);
      bool found_match = false;

      // check if already synchronized
      if (cand_phys_id != PALLAS_INDEX_INVALID && seq_cmp(src_seq, t2->sequences[cand_phys_id])) {
        found_match = true;

      // try to find other match somewhere
      } else {
        uint32_t match_id = find_matching_seq(src_seq, t2);
        if (match_id != PALLAS_INDEX_INVALID) {
          seq_swap(t2, seq_id, match_id);
          found_match = true;
          map_swap(seq_map, seq_rev, t2->id, seq_id, match_id);
          number_of_swaps++;
        }
      }

      // if no match found insert placeholder
      if (!found_match && cand_phys_id != PALLAS_INDEX_INVALID) {
        seq_displace(t2, seq_id);
        uint32_t prev_owner = seq_rev[t2->id].count(seq_id) ? seq_rev[t2->id][seq_id] : seq_id;
        uint32_t new_logi_id = t2->sequence_id_map.size();
        map_set(seq_map, seq_rev, t2->id, prev_owner, new_logi_id);
        seq_rev[t2->id].erase(seq_id);
        number_of_swaps++;
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

  trace->locations.clear();
  trace->store(save_dir_name, trace->parameter_handler);
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
      s.str = (char*) std::calloc(s.length + 1, sizeof(char));
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

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // |         Synchronize Events           |
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  thread_token_map thread_event_map;
  thread_token_map thread_event_rev;

  uint32_t n_events_verified = 0;

  // initialize event_map to identity for original events
  for (auto* t : threads) {
    for (uint32_t i = 0; i < t->event_id_map.size(); i++) {
      if (t->event_id_map[i] != PALLAS_INDEX_INVALID) {
        map_set(thread_event_map, thread_event_rev, t->id, i, i);
      }
    }
  }

  for (auto* t : threads) {
    uint32_t thread_n_events = t->event_id_map.size();
    if (thread_n_events <= n_events_verified) {
      continue;
    }
    // else: -> thread has events that need to be synchronized

    // pre-fill other threads with invalids to match nb_events
    for (auto* t2 : threads) {
      if (t2->id == t->id) continue;
      if (t2->event_id_map.size() < thread_n_events) {
        t2->event_id_map.resize(thread_n_events, PALLAS_INDEX_INVALID);
      }
    }

    sync_events(threads, t, n_events_verified, thread_n_events, thread_event_map, thread_event_rev);
    // track updated event index
    n_events_verified = thread_n_events;

    // compact extra danging invalids
    for (auto* t2 : threads) {
      if (t2->id == t->id) continue;
      while (t2->event_id_map.size() > n_events_verified && t2->event_id_map.back() == PALLAS_INDEX_INVALID) {
        t2->event_id_map.pop_back();
      }
    }
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // |   Synchronize Sequences and Loops    |
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::map<uint32_t, token_lookup> loop_base_tokens;        // thread_id -> original_loop_id -> base repeated_token
  std::map<uint32_t, token_vector_lookup> seq_base_tokens;  // thread_id -> original_seq_id  -> base tokens

  for (auto* t : threads) {
    for (uint32_t i = 0; i < t->loop_id_map.size(); i++) {
      uint32_t phys_id = t->loop_id_map[i];
      if (phys_id != PALLAS_INDEX_INVALID) {
        loop_base_tokens[t->id][i] = t->loops[phys_id].repeated_token;
      }
    }
    for (uint32_t i = 0; i < t->sequence_id_map.size(); i++) {
      uint32_t phys_id = t->sequence_id_map[i];
      if (phys_id != PALLAS_INDEX_INVALID) {
        seq_base_tokens[t->id][i] = t->sequences[phys_id].tokens;
      }
    }
  }

  thread_token_map thread_loop_map;
  thread_token_map thread_loop_rev;

  thread_token_map thread_seq_map;
  thread_token_map thread_seq_rev;

  bool update_events = true;
  bool update_seqs   = false;
  bool update_loops  = false;

  // initialize loop_map and seq_map to identity for original tokens
  for (auto* t : threads) {
    for (uint32_t i = 0; i < t->loop_id_map.size(); i++) {
      if (t->loop_id_map[i] != PALLAS_INDEX_INVALID) {
        map_set(thread_loop_map, thread_loop_rev, t->id, i, i);
      }
    }
    for (uint32_t i = 0; i < t->sequence_id_map.size(); i++) {
      if (t->sequence_id_map[i] != PALLAS_INDEX_INVALID) {
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
      thread_loop_map, update_loops,
      thread_loop_rev, loop_base_tokens
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |      Synchronize Updated Loops       |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    uint32_t n_loops_verified = 0;

    for (auto* t : threads) {
      uint32_t thread_n_loops = t->loop_id_map.size();
      if (thread_n_loops <= n_loops_verified) {
        continue;
      }
      // else: -> thread has loops that need to be synchronized

      // pre-fill other threads with invalids to match nb_loops
      for (auto* t2 : threads) {
        if (t2->id == t->id) continue;
        if (t2->loop_id_map.size() < thread_n_loops) {
          t2->loop_id_map.resize(thread_n_loops, PALLAS_INDEX_INVALID);
        }
      }

      number_of_swaps += sync_loops(threads, t, n_loops_verified, thread_n_loops, thread_loop_map, thread_loop_rev);
      n_loops_verified = thread_n_loops;

      // compact extra danging invalids
      for (auto* t2 : threads) {
        if (t2->id == t->id) continue;
        while (t2->loop_id_map.size() > n_loops_verified && t2->loop_id_map.back() == PALLAS_INDEX_INVALID) {
          t2->loop_id_map.pop_back();
        }
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
      thread_loop_map, update_loops,
      thread_seq_rev, seq_base_tokens
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // |    Synchronize Updated Sequences     |
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    uint32_t n_seqs_verified = 0;

    for (auto* t : threads) {
      uint32_t thread_n_seqs = t->sequence_id_map.size();
      if (thread_n_seqs <= n_seqs_verified) {
        continue;
      }
      // else: -> thread has sequences that need to be synchronized

      // pre-fill other threads with invalids to match nb_sequences
      for (auto* t2 : threads) {
        if (t2->id == t->id) continue;
        if (t2->sequence_id_map.size() < thread_n_seqs) {
          t2->sequence_id_map.resize(thread_n_seqs, PALLAS_INDEX_INVALID);
        }
      }

      number_of_swaps += sync_sequences(threads, t, n_seqs_verified, thread_n_seqs, thread_seq_map, thread_seq_rev);
      n_seqs_verified = thread_n_seqs;

      // compact extra danging invalids
      for (auto* t2 : threads) {
        if (t2->id == t->id) continue;
        while (t2->sequence_id_map.size() > n_seqs_verified && t2->sequence_id_map.back() == PALLAS_INDEX_INVALID) {
          t2->sequence_id_map.pop_back();
        }
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
