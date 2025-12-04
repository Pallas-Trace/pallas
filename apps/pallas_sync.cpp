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
#include "pallas/pallas_hash.h"
#include "pallas/pallas_storage.h"
// #include "pallas/pallas_write.h"

#define DEBUG_LEVEL 0
#define ENABLE_WRITE

typedef std::map<uint32_t, uint32_t> token_map;
typedef std::map<uint32_t, token_map> thread_token_map;

uint32_t eval_token_map(thread_token_map& map, uint32_t thread_id, uint32_t in_id) {
  uint32_t out_id;
  while((out_id = map[thread_id][in_id]) != in_id) {
    in_id = out_id;
  }
  return out_id;
}

bool event_cmp(pallas::EventSummary &e1, pallas::EventSummary &e2) {

  if (e1.event.record != e2.event.record) {
    return false;
  }

  if (memcmp(e1.event.event_data,
             e2.event.event_data,
             e1.event.event_size) != 0) {
    return false;
  }

  return true;
}

void event_insert(pallas::EventSummary &es,
                  pallas::Thread *t, uint32_t id) {
  while (id >= t->nb_allocated_events) {
    // std::cout << "allocating event space" << std::endl;
    doubleMemorySpaceConstructor(t->events, t->nb_allocated_events);
  }

  t->events[id] = std::move(es);

  if (id >= t->nb_events) {
    t->nb_events++;
    t->events[id].id = id;
  }
}

void event_override_invalid(pallas::Thread *t, uint32_t id) {
  t->events[id].id = PALLAS_TOKEN_ID_INVALID;
  t->events[id].event = pallas::Event {
    pallas::PALLAS_EVENT_MAX_ID,
    0,
    {}
  };
  t->events[id].timestamps = NULL;
  t->events[id].nb_occurences = 0;
  t->events[id].attribute_buffer = NULL;
  t->events[id].attribute_buffer_size = 0;
  t->events[id].attribute_pos = 0;
}

void event_swap(pallas::Thread *t,
                uint32_t src_id, uint32_t swap_id) {
  std::swap(t->events[src_id], t->events[swap_id]);
  t->events[src_id].id = src_id;
  t->events[swap_id].id = swap_id;
}

uint32_t find_matching_event(pallas::EventSummary &src_event,
                             pallas::Thread *t) {

  uint32_t t_n_events = t->nb_events;
  for (uint32_t e2_idx = src_event.id; e2_idx < t_n_events; e2_idx++) {
    pallas::EventSummary& cand_event = t->events[e2_idx];
    if (event_cmp(src_event, cand_event)) {
      // std::cout << "found swap match!" << std::endl;
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
                thread_token_map& event_map) {

  for (auto* t2 : threads) {

    // ignore identical threads
    if (t2->id == t->id) {
      continue;
    }

    for (uint32_t event_id = start_id; event_id < end_id; event_id++) {
      assert(t->events[event_id].event.record != pallas::PALLAS_EVENT_MAX_ID);

      pallas::EventSummary& src_event = t->events[event_id];
      pallas::EventSummary& cand_event = t2->events[event_id];

      bool found_match = false;

      // check if already synchronized
      if (event_cmp(src_event,cand_event)) {
        found_match = true;
        event_map[t2->id][event_id] = event_id;

      // try to find other match somewhere
      } else {
        uint32_t match_id = find_matching_event(src_event, t2);
        if (match_id != event_id) {
          event_swap(t2, event_id, match_id);
          found_match = true;
          event_map[t2->id][event_id] = match_id;
          event_map[t2->id][match_id] = event_id;
        }
      }

      if (!found_match) {
        // if no match found insert placeholder
        pallas::EventSummary swap_event = std::move(t2->events[event_id]);
        uint32_t swap_id = t2->nb_events;
        event_insert(swap_event, t2, swap_id);
        event_override_invalid(t2, event_id);
        event_map[t2->id][event_id] = swap_id;
      }
    }
  }
  return 0;
}

bool seq_cmp(pallas::Sequence *seq1, pallas::Sequence *seq2) {

  if (seq1 == NULL || seq2 == NULL) {
    return false;
  }

  if (seq1->hash != seq2->hash) {
    // return false;
  }

  if (seq1->tokens.size() != seq2->tokens.size()) {
    return false;
  }

  for (int i = 0; i < seq1->tokens.size(); i++) {
    if (seq1->tokens[i].id != seq2->tokens[i].id) {
      return false;
    }
  }

  return true;
}

void seq_insert(pallas::Sequence* seq, pallas::Thread *t, uint32_t id) {
  while (id >= t->nb_allocated_sequences) {
    // std::cout << "allocating sequence space" << std::endl;
    doubleMemorySpaceConstructor(t->sequences, t->nb_allocated_sequences);
  }

  t->sequences[id] = seq;

  if (id >= t->nb_sequences) {
    t->nb_sequences++;
    t->sequences[id]->id = id;
  }
}

void seq_override_invalid(pallas::Thread *t, uint32_t id) {
  if (t->sequences[id] == NULL) {
    t->sequences[id] = new pallas::Sequence;
  }
  t->sequences[id]->id = PALLAS_TOKEN_ID_INVALID;
  t->sequences[id]->durations = NULL;
  t->sequences[id]->exclusive_durations = NULL;
  t->sequences[id]->timestamps = NULL;
  t->sequences[id]->hash = 0;
  t->sequences[id]->tokens.clear();
}

void seq_swap(pallas::Thread *t, uint32_t src_id, uint32_t swap_id) {
  pallas::Sequence *temp = t->sequences[src_id];
  t->sequences[src_id] = t->sequences[swap_id];
  t->sequences[src_id]->id = src_id;
  t->sequences[swap_id] = temp;
  t->sequences[swap_id]->id = swap_id;
}

uint32_t find_matching_seq(pallas::Sequence *src_seq, pallas::Thread *t) {
  
  uint32_t t_n_seqs = t->nb_sequences;
  for (uint32_t s2_idx = src_seq->id; s2_idx < t_n_seqs; s2_idx++) {
    pallas::Sequence *cand_seq = t->sequences[s2_idx];
    if (seq_cmp(src_seq, cand_seq)) {
      // std::cout << "found swap match!" << std::endl;
      return cand_seq->id;
    }
  }

  // if no match found return src_event id
  return src_seq->id;
}

void update_sequence_tokens(std::vector<pallas::Thread*> threads,
                            thread_token_map event_map,
                            bool update_events,
                            thread_token_map seq_map,
                            bool update_seqs,
                            thread_token_map loop_map,
                            bool update_loops) {

  for (auto* t : threads) {
    for (uint32_t seq_id = 0; seq_id < t->nb_sequences; seq_id++) {
      pallas::Sequence* seq = t->sequences[seq_id];

      for (auto& token : seq->tokens) {
        if (token.type == 1 && update_events) {
          token.id = eval_token_map(event_map, t->id, token.id);
        }
        if (token.type == 2 && update_seqs) {
          token.id = eval_token_map(seq_map, t->id, token.id);
        }
        if (token.type == 3 && update_loops) {
          token.id = eval_token_map(loop_map, t->id, token.id);
        }
      }

      seq->hash = pallas::hash32(reinterpret_cast<uint8_t*>(seq->tokens.data()), 
                                  seq->tokens.size() * sizeof(pallas::Token), SEED);
    }
  }
}

int sync_sequences(std::vector<pallas::Thread*> threads,
                  pallas::Thread *t,
                  uint32_t start_id,
                  uint32_t end_id,
                  thread_token_map& seq_map) {
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

      pallas::Sequence* src_seq = t->sequences[seq_id];
      pallas::Sequence* cand_seq = t2->sequences[seq_id];

      bool found_match = false;

      // check if src invalid
      if (src_seq->id == PALLAS_TOKEN_ID_INVALID || src_seq->tokens.size() == 0) {
        continue;
      }

      // check if cand aleady invalid
      if (cand_seq != NULL && cand_seq->id == PALLAS_TOKEN_ID_INVALID) {
        if (t2->nb_sequences <= seq_id) {
          t2->nb_sequences = seq_id + 1;
          seq_map[t2->id][seq_id] = seq_id;
        }
        continue;
      }

      // check if already synchronized
      if (seq_cmp(src_seq, cand_seq)) {
        found_match = true;
        seq_map[t2->id][seq_id] = seq_id;

      // try to find other match somewhere
      } else {
        uint32_t match_id = find_matching_seq(src_seq, t2);
        if (match_id != seq_id) {
          seq_swap(t2, seq_id, match_id);
          found_match = true;
          seq_map[t2->id][seq_id] = match_id;
          seq_map[t2->id][match_id] = seq_id;
          number_of_swaps++;
        }
      }

      // if no match found insert placeholder
      if (!found_match) {
        pallas::Sequence* swap_seq = t2->sequences[seq_id];
        uint32_t swap_id = t2->nb_sequences;
        if (swap_seq != NULL && swap_seq->id != PALLAS_TOKEN_ID_INVALID) {
          seq_insert(swap_seq, t2, swap_id);
          t2->sequences[seq_id] = NULL;
          seq_override_invalid(t2, seq_id);
          seq_map[t2->id][seq_id] = swap_id;
          number_of_swaps++;
        } else if (swap_seq == NULL) {
          seq_insert(new pallas::Sequence, t2, swap_id);
          seq_override_invalid(t2, seq_id);
          seq_map[t2->id][swap_id] = swap_id;
        }
      }
    }
  }
  return number_of_swaps;
}

bool loop_cmp(pallas::Loop &l1, pallas::Loop& l2) {

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

void loop_insert(pallas::Loop &l, 
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
  t->loops[id].self_id.id = PALLAS_TOKEN_ID_INVALID;
  t->loops[id].nb_iterations = 0;
}

void loop_swap(pallas::Thread *t,
               uint32_t src_id, uint32_t swap_id) {
  std::swap(t->loops[src_id], t->loops[swap_id]);
  t->loops[src_id].self_id.id = src_id;
  t->loops[swap_id].self_id.id = swap_id;
}

uint32_t find_matching_loop(pallas::Loop &src_loop,
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

void update_loop_tokens(std::vector<pallas::Thread*> threads,
                        thread_token_map event_map,
                        bool update_events,
                        thread_token_map seq_map,
                        bool update_seqs,
                        thread_token_map loop_map,
                        bool update_loops) {

  for (auto* t : threads) {
    for (uint32_t loop_id = 0; loop_id < t->nb_loops; loop_id++) {
      pallas::Loop& loop = t->loops[loop_id];

      auto& token = loop.repeated_token;
      if (token.type == 1 && update_events) {
        token.id = eval_token_map(event_map, t->id, token.id);
      }
      if (token.type == 2 && update_seqs) {
        token.id = eval_token_map(seq_map, t->id, token.id);
      }
      if (token.type == 3 && update_loops) {
        token.id = eval_token_map(loop_map, t->id, token.id);
      }
    }
  }
}

int sync_loops(std::vector<pallas::Thread*> threads,
              pallas::Thread *t,
              uint32_t start_id,
              uint32_t end_id,
              thread_token_map& loop_map) {
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
      if (src_loop.repeated_token.type == pallas::TypeInvalid) {
        continue;
      }

      // check if cand already invalid
      if (cand_loop.repeated_token.type == pallas::TypeInvalid) {
        if (t2->nb_loops <= loop_id) {
          t2->nb_loops = loop_id + 1;
          loop_map[t2->id][loop_id] = loop_id;
        }
        continue;
      }

      // check if already synchronized
      if (loop_cmp(src_loop, cand_loop)) {
        found_match = true;
        loop_map[t2->id][loop_id] = loop_id;

      // try to find other match somewhere
      } else {
        uint32_t match_id = find_matching_loop(src_loop, t2);
        if (match_id != loop_id) {
          loop_swap(t2, loop_id, match_id);
          found_match = true;
          loop_map[t2->id][loop_id] = match_id;
          loop_map[t2->id][match_id] = loop_id;
          number_of_swaps++;
        }
      }

      if (!found_match) {
        // if no match found insert placeholder
        pallas::Loop swap_loop = std::move(t2->loops[loop_id]);
        uint32_t swap_id = t2->nb_loops;
        loop_insert(swap_loop, t2, swap_id);
        loop_override_invalid(t2, loop_id);
        loop_map[t2->id][loop_id] = swap_id;
        number_of_swaps++;
      }
    }
  }
  return number_of_swaps;
}


void save_thread_copy(pallas::GlobalArchive *trace,
                      std::vector<pallas::Archive*> archives,
                      std::vector<pallas::Thread*> threads,
                      char *save_dir_name) {
  for (auto* t : threads) {
    t->store(save_dir_name, trace->parameter_handler, true);
  }

  for (auto* a: archives) {
    a->store(save_dir_name, trace->parameter_handler);
  }

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
