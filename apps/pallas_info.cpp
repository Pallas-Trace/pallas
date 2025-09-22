/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <cstdlib>
#include <cstring>
#include <iomanip>
#if __GNUC__ >= 13 || __clang__ >= 14 || _MSC_VER >= 1929
#include <format>
#define HAS_FORMAT
#endif
#include <pallas/pallas_parameter_handler.h>

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"

#define DURATION_WIDTH 15

using namespace pallas;

int thread_to_print = -1;
int archive_to_print = -1;

enum command {
  none = 0,
  show_thread_content = 1 << 0,
  show_definitions = 1 << 1,
  show_sequence_content = 1 << 2,
  show_sequence_durations = 1 << 3,
  show_sequence_timestamps = 1 << 4,
  list_archives = 1 << 5,
  show_archive_details = 1 << 6,
  list_threads = 1 << 7,
};

int cmd = none;

static bool _should_print_thread(int thread_id) {
  return thread_to_print < 0 || thread_to_print == thread_id;
}

static bool _should_print_archive(int archive_id) {
  return archive_to_print < 0 || archive_to_print == archive_id;
}

static double ns2s(uint64_t ns) {
  return ns * 1.0 / 1e9;
}

void info_archive_header();
void info_archive(Archive* archive);
void info_thread_header();
void info_thread_summary(Thread* thread);

void print_sequence(const Sequence* s, const Thread* t) {
  std::cout << "{";
  for (unsigned i = 0; i < s->size(); i++) {
    const Token token = s->tokens[i];
    std::cout << t->getTokenString(token);
    if (i < s->size() - 1)
      printf(", ");
  }
  std::cout << "}" << std::endl;
}

std::string getTokenString(Thread* thread, Token t) {
  switch (t.type) {
  case TypeEvent: {
    Event* e = thread->getEvent(t);
    return thread->getEventString(e);
    break;
  }
  case TypeSequence: {
    Sequence* s = thread->getSequence(t);
    return s->guessName(thread);
    break;
  }
  case TypeLoop: {
    Loop* l = thread->getLoop(t);
    return l->guessName(thread);
    break;
  }
  default:
    return "Unknown token";
  }
}

void info_event_header() {
  std::cout << std::left << "#";
  std::cout << std::setw(14) << std::left << "Event_id";
  std::cout << std::setw(35) << std::left << "Event_name";
  std::cout << std::setw(20) << std::right << "Nb_occurence";
  std::cout << std::setw(20) << std::right << "Min_duration(ns)";
  std::cout << std::setw(20) << std::right << "Max_duration(ns)";
  std::cout << std::setw(20) << std::right << "Mean_duration(ns)";
  std::cout << std::endl;
}

void info_event(Thread* t, int index) {
  EventSummary* e = &t->events[index];

  std::cout << std::left << "E" << std::setw(14) << std::left << index;
  std::cout << std::setw(35) << std::left << t->getEventString(&e->event);
  std::cout << std::setw(20) << std::right << e->timestamps->size;
  std::cout << std::endl;
}

void info_sequence_header() {
  std::cout << std::left << "#";
  std::cout << std::setw(14) << std::left << "Sequence_id";
  std::cout << std::setw(35) << std::left << "Sequence_name";
  std::cout << std::setw(18) << std::right << "Nb_occurence";
  std::cout << std::setw(18) << std::right << "Min_duration(s)";
  std::cout << std::setw(18) << std::right << "Max_duration(s)";
  std::cout << std::setw(18) << std::right << "Mean_duration(s)";
  std::cout << std::setw(18) << std::right << "Total_duration(s)";
  std::cout << std::setw(18) << std::right << "Nb_token";
  //  std::cout << std::setw(18) << std::right << "Event_count";
  std::cout << std::setw(18) << std::right << "Contention_score";
  std::cout << std::endl;
}

float contention_score(Thread* t, Sequence* s) {
  pallas_duration_t delta_duration = s->durations->size * (s->durations->mean - s->durations->min);
  pallas_duration_t thread_duration = t->getDuration();
  if (delta_duration > thread_duration)
    return -1;
  return (float)delta_duration / thread_duration;
}

void info_sequence(Thread* t, int index, bool details = false) {
  Sequence* s = t->sequences[index];

  if (details) {
    info_sequence_header();
  }

  std::string sequence_name = s->guessName(t);

  std::cout << std::left << "S" << std::setw(14) << std::left << index;
  std::cout << std::setw(35) << std::left << sequence_name;
  std::cout << std::setw(18) << std::right << s->durations->size;
  std::cout << std::setw(18) << std::right << ns2s(s->durations->min == UINT64_MAX ? 0 : s->durations->min);
  std::cout << std::setw(18) << std::right << ns2s(s->durations->max == UINT64_MAX ? 0 : s->durations->max);
  std::cout << std::setw(18) << std::right << ns2s(s->durations->mean == UINT64_MAX ? 0 : s->durations->mean);
  std::cout << std::setw(18) << std::right << ns2s(s->durations->mean == UINT64_MAX ? 0 : s->durations->mean * s->durations->size);
  std::cout << std::setw(18) << std::right << s->size();

  std::cout << std::setw(18) << std::right << contention_score(t, s);
  // std::cout << std::setw(18) << std::right << s->getEventCount(t);
  std::cout << std::endl;

  if (details) {
    if (cmd & show_sequence_content) {
      std::cout << std::endl << "------------------- Sequence" << s->id << " contains:" << std::endl;
      for (auto token : s->tokens) {
        std::cout << "\t" << std::left << getTokenString(t, token) << std::endl;
      }
        // TODO show exclusive durations
      std::cout << "------------------- End of sequence" << s->id << std::endl;
      std::cout << std::endl;
    }

    if (cmd & show_sequence_durations) {
      std::cout << std::endl << "------------------- Sequence" << s->id << " duration:" << std::endl;
      for (int i = 0; i < s->durations->size; i++) {
        uint64_t duration = s->durations->at(i);
        std::cout << "\t" << duration << std::endl;
      }
      std::cout << std::endl << "------------------- End of sequence" << s->id << " durations." << std::endl;
    }

    if (cmd & show_sequence_timestamps) {
      std::cout << std::endl << "------------------- Sequence" << s->id << " timestamps:" << std::endl;
      for (int i = 0; i < s->timestamps->size; i++) {
        uint64_t timestamp = s->timestamps->at(i);
        std::cout << "\t" << timestamp << std::endl;
      }
      std::cout << std::endl << "------------------- End of sequence" << s->id << " timestamps." << std::endl;
    }
}
}

void info_loop_header() {
    std::cout << std::left << "#";
    std::cout << std::setw(14) << std::left << "Loop_id";
    std::cout << std::setw(14) << std::left << "Sequence_id";
    std::cout << std::setw(35) << std::left << "Loop_name";
    std::cout << std::setw(18) << std::right << "Nb_occurence";
    std::cout << std::setw(18) << std::right << "Min_nb_iterations";
    std::cout << std::setw(18) << std::right << "Max_nb_iterations";
    std::cout << std::setw(18) << std::right << "Mean_nb_iterations";
    std::cout << std::endl;
}

void info_loop(Thread* t, int index) {
    Loop* l = &t->loops[index];
    std::string loop_name = l->guessName(t);
    std::cout << std::left << "L" << std::setw(14) << std::left << index;
    std::cout << std::left << "S" << std::setw(14) << std::left << l->repeated_token.id;
    std::cout << std::setw(35) << std::left << loop_name;
    std::cout << std::setw(18) << std::right << l->nb_iterations;
    std::cout << std::endl;
}

void info_thread(Thread* t) {
  if (!_should_print_thread(t->id))
    return;

  if ((cmd & show_thread_content) == 0)
    return;

  info_thread_header();
  info_thread_summary(t);

  printf("\nEvents {.nb_events: %lu}\n", t->nb_events);
  info_event_header();
  for (unsigned i = 0; i < t->nb_events; i++) {
    info_event(t, i);
  }

  printf("\nSequences {.nb_sequences: %lu}\n", t->nb_sequences);
  info_sequence_header();
  for (unsigned i = 0; i < t->nb_sequences; i++) {
    info_sequence(t, i);
  }

  printf("\nLoops {.nb_loops: %lu}\n", t->nb_loops);
  info_loop_header();
  for (unsigned i = 0; i < t->nb_loops; i++) {
    info_loop(t, i);
  }

  if ((cmd & show_sequence_content) || (cmd & show_sequence_durations)) {
    info_sequence_header();
    for (unsigned i = 0; i < t->nb_sequences; i++) {
      info_sequence(t, i, true);
    }
  }
}

void info_definitions(Definition& definitions) {
  if (!definitions.strings.empty()) {
    printf("\tStrings {.nb_strings: %zu } :\n", definitions.strings.size());

    for (auto& [stringRef, string] : definitions.strings) {
      printf("\t\t%d: '%s'\n", string.string_ref, string.str);
    }
  }

  if (!definitions.regions.empty()) {
    printf("\tRegions {.nb_regions: %zu } :\n", definitions.regions.size());
    for (auto& [regionRef, region] : definitions.regions) {
      printf("\t\t%d: %s\n", region.region_ref, definitions.getString(region.string_ref)->str);
    }
  }

  if (!definitions.groups.empty()) {
    printf("\tGroups {.nb_groups: %zu } :\n", definitions.groups.size());
    for (auto& [groupRef, group] : definitions.groups) {
      printf("\t\t%d: '%s' [", group.group_ref, definitions.getString(group.name)->str);
      for (uint32_t i = 0; i < group.numberOfMembers; i++) {
        printf("%s%lu", i > 0 ? ", " : "", group.members[i]);
      }
      printf("]\n");
    }
  }

  if (!definitions.comms.empty()) {
    printf("\tComms {.nb_comms: %zu } :\n", definitions.comms.size());
    for (auto& [commRef, comm] : definitions.comms) {
      printf("\t\t%d: '%s' (group, %d, parent: %d) \n", comm.comm_ref, definitions.getString(comm.name)->str, comm.group, comm.parent);
    }
  }
}

void info_global_archive(GlobalArchive* archive) {
  printf("Main archive:\n");

  if (cmd & show_archive_details) {
    printf("\tDirectory name:   %s\n", archive->dir_name);
    printf("\tTrace name: %s\n", archive->trace_name);
  }

  printf("\tFullpath:    %s\n", archive->fullpath);
  printf("\t# Processes: %lu\n", archive->location_groups.size());
  if (archive->nb_archives) {
    printf("\t# Archives: %d\n", archive->nb_archives);
  }

  std::cout << "\nConfiguration:\n"
            << "\tCompression Algorithm: " << toString(archive->parameter_handler->compressionAlgorithm) << "\n"
            << "\tEncoding algorithm: " << toString(archive->parameter_handler->encodingAlgorithm) << "\n"
            << "\tLoop-finding algorithm: " << toString(archive->parameter_handler->loopFindingAlgorithm) << "\n"
            << "\tMax loop length: " << archive->parameter_handler->maxLoopLength << "\n"
            << "\tZSTD compression level: " << archive->parameter_handler->zstdCompressionLevel << "\n"
            << "\tTimestamp storage: " << toString(archive->parameter_handler->timestampStorage) << "\n";

  if (cmd & show_definitions) {
    info_definitions(archive->definitions);
    if (!archive->location_groups.empty()) {
      printf("\tLocation_groups {.nb_lg: %zu }:\n", archive->location_groups.size());
      for (auto& locationGroup : archive->location_groups) {
        printf("\t\t%d: %s", locationGroup.id, archive->getString(locationGroup.name)->str);
        if (locationGroup.parent != PALLAS_LOCATION_GROUP_ID_INVALID)
          printf(", parent: %d", locationGroup.parent);
        printf("\n");
      }
    }

    if (!archive->getLocationList().empty()) {
      printf("\tLocations {.nb_loc: %zu }:\n", archive->getLocationList().size());
      for (auto location : archive->getLocationList()) {
        printf("\t\t%d: %s, parent: %d\n", location.id, archive->getArchive(location.parent)->getString(location.name)->str, location.parent);
      }
    }
  }

  printf("\n");
}

static bool _archiveContainsThread(Archive* archive, int thread_id) {
  for (int i = 0; i < archive->nb_threads; i++) {
    auto thread = archive->getThreadAt(i);
    if (thread->id == thread_id)
      return true;
  }
  return false;
}

void info_archive_header() {
  std::cout << std::left << "#";
  std::cout << std::setw(14) << std::left << "Archive_id";
  std::cout << std::setw(20) << std::left << "Archive_name";
  std::cout << std::setw(15) << std::right << "Nb_threads";
  std::cout << std::endl;
}

void info_archive(Archive* archive) {
  if (!_should_print_archive(archive->id)) {
    return;
  }

  std::cout << std::setw(15) << std::left << archive->id;
  std::cout << std::setw(20) << std::left << archive->getName();
  std::cout << std::setw(15) << std::right << archive->nb_threads;
  std::cout << std::endl;
}

void info_archive_definition(Archive* archive) {
  if (!_should_print_archive(archive->id)) {
    return;
  }
  printf("Archive %d:\n", archive->id);
  if (cmd & show_definitions) {
    info_definitions(archive->definitions);
    if (!archive->location_groups.empty()) {
      printf("\tLocation_groups {.nb_lg: %zu }:\n", archive->location_groups.size());
      for (auto& locationGroup : archive->location_groups) {
        printf("\t\t%d: %s", locationGroup.id, archive->getString(locationGroup.name)->str);
        if (locationGroup.parent != PALLAS_LOCATION_GROUP_ID_INVALID)
          printf(", parent: %d", locationGroup.parent);
        printf("\n");
      }
    }

    if (!archive->locations.empty()) {
      printf("\tLocations {.nb_loc: %zu }:\n", archive->locations.size());
      for (auto location : archive->locations) {
        printf("\t\t%d: %s, parent: %d\n", location.id, archive->getString(location.name)->str, location.parent);
      }
    }
  }
}

void info_thread_header() {
  std::cout << std::left << "#";
  std::cout << std::setw(19) << std::left << "Thread_name";
  std::cout << std::setw(15) << std::left << "Thread_id";
  std::cout << std::setw(15) << std::right << "First_ts";
  std::cout << std::setw(15) << std::right << "Last_ts";
  std::cout << std::setw(15) << std::right << "Duration(s)";
  std::cout << std::setw(15) << std::right << "Event_count";
  std::cout << std::setw(15) << std::right << "Nb_events";
  std::cout << std::setw(15) << std::right << "Nb_sequences";
  std::cout << std::setw(15) << std::right << "Nb_loops";
  std::cout << std::setw(15) << std::right << "Archive_id";
  std::cout << std::endl;
}

void info_thread_summary(Thread* thread) {
  if (thread && _should_print_thread(thread->id)) {
    std::cout << std::setw(20) << std::left << thread->getName();
    std::cout << std::setw(15) << std::left << thread->id;
    std::cout << std::setw(15) << std::right << thread->getFirstTimestamp();
    std::cout << std::setw(15) << std::right << thread->getLastTimestamp();
    std::cout << std::setw(15) << std::right << ns2s(thread->getDuration());
    std::cout << std::setw(15) << std::right << thread->getEventCount();
    std::cout << std::setw(15) << std::right << thread->nb_events;
    std::cout << std::setw(15) << std::right << thread->nb_sequences;
    std::cout << std::setw(15) << std::right << thread->nb_loops;
    std::cout << std::setw(15) << std::right << thread->archive->id;
    std::cout << std::endl;
  }
}

void info_threads(Archive* archive) {
  if (!(cmd & list_threads))
    return;
  if (thread_to_print >= 0 && !_archiveContainsThread(archive, thread_to_print)) {
    return;
  }

  if (archive->threads) {
    for (int i = 0; i < archive->nb_threads; i++) {
      auto thread = archive->getThreadAt(i);
      info_thread_summary(thread);
    }
  }
}

void info_trace(GlobalArchive* trace) {
  info_global_archive(trace);

  if (cmd & list_archives) {
    info_archive_header();
    for (int i = 0; i < trace->nb_archives; i++) {
      info_archive(trace->archive_list[i]);
    }
    printf("\n");
    for (int i = 0; i < trace->nb_archives; i++) {
      info_archive_definition(trace->archive_list[i]);
    }
  }

  if (cmd & list_threads) {
    info_thread_header();
    for (int i = 0; i < trace->nb_archives; i++) {
      info_threads(trace->archive_list[i]);
    }
  }

  if (cmd & show_thread_content) {
    for (int i = 0; i < trace->nb_archives; i++) {
      for (int j = 0; j < trace->archive_list[i]->nb_threads; j++) {
        auto thread = trace->archive_list[i]->getThreadAt(j);
        if (thread)
          info_thread(thread);
      }
    }
  }
}

void usage(const char* prog_name) {
  printf("Usage: %s [OPTION] trace_file\n", prog_name);
  printf("\t-v             Verbose mode\n");
  printf("\t-D             show definitions\n");
  printf("\n");
  printf("\t-la            list archives\n");
  printf("\t-lt            list threads\n");
  printf("\n");
  printf("\t-t             show thread details\n");
  printf("\t--content      show sequence content\n");
  printf("\t--durations    show sequence durations\n");
  printf("\t--timestamps   show sequence timestamps\n");
  printf("\n");
  printf("\t-da            show archive details\n");
  printf("\n");
  printf("\t--archive id   Only print archive <id>\n");
  printf("\t--thread id    Only print thread <id>\n");

  printf("\t-?  -h --help  Display this help and exit\n");
}

int main(int argc, char** argv) {
  int nb_opts = 0;
  char* trace_name = nullptr;

  for (nb_opts = 1; nb_opts < argc; nb_opts++) {
    if (!strcmp(argv[nb_opts], "-v")) {
      pallas_debug_level_set(DebugLevel::Debug);
    } else if (!strcmp(argv[nb_opts], "-D")) {
      cmd |= show_definitions;
    } else if (!strcmp(argv[nb_opts], "-t")) {
      cmd |= show_thread_content;
    } else if (!strcmp(argv[nb_opts], "-la")) {
      cmd |= list_archives;
    } else if (!strcmp(argv[nb_opts], "-lt")) {
      cmd |= list_threads;
    } else if (!strcmp(argv[nb_opts], "--content")) {
      cmd |= show_sequence_content;
    } else if (!strcmp(argv[nb_opts], "--durations")) {
      cmd |= show_sequence_durations;
    } else if (!strcmp(argv[nb_opts], "--timestamps")) {
      cmd |= show_sequence_timestamps;
    } else if (!strcmp(argv[nb_opts], "-da")) {
      cmd |= show_archive_details;
    } else if (!strcmp(argv[nb_opts], "--archive")) {
      archive_to_print = atoi(argv[nb_opts + 1]);
      nb_opts++;
    } else if (!strcmp(argv[nb_opts], "--thread")) {
      thread_to_print = atoi(argv[nb_opts + 1]);
      nb_opts++;
    } else if (!strcmp(argv[nb_opts], "-h") || !strcmp(argv[nb_opts], "-?") || !strcmp(argv[nb_opts], "--help")) {
      printf("invalid option '%s'\n", argv[nb_opts]);
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      /* Unknown parameter name. It's probably the program name. We can stop
       * parsing the parameter list.
       */
      trace_name = argv[nb_opts];
      break;
    }
  }

  if (trace_name == nullptr) {
    printf("Missing trace file\n");
    usage(argv[0]);
    return EXIT_SUCCESS;
  }

  auto trace = pallas_open_trace(trace_name);
  if (trace == nullptr) {
    return EXIT_FAILURE;
  }
  info_trace(trace);

  delete trace;
  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
