/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <cstdlib>
#include <cstring>
#include <iomanip>
#if __GNUC__ >= 13 || __clang__ >= 14 || _MSC_VER >= 1929
#include <format>
#include <climits>
#define HAS_FORMAT
#endif
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"

#define DURATION_WIDTH 15

using namespace pallas;

int thread_to_print   = -1;
int archive_to_print  = -1;

enum command {
  none                    = 0,
  show_thread_content     = 1<<0,
  show_definitions        = 1<<1,
  show_sequence_content   = 1<<2,
  show_sequence_durations = 1<<3,
  list_archives           = 1<<4,
  show_archive_details    = 1<<5,
  list_threads            = 1<<6,
};

int cmd = none;

static bool _should_print_thread(int thread_id) {
  return thread_to_print <0 || thread_to_print == thread_id;
}

static bool _should_print_archive(int archive_id) {
  return archive_to_print <0 || archive_to_print == archive_id;
}

static double ns2s(uint64_t ns) {
  return ns*1.0/1e9;
}


void info_archive_header();
void info_archive(Archive* archive);
void info_thread_header();
void info_thread_summary(Thread* thread);

std::string guess_sequence_name(Thread *t, Sequence*s);
std::string guess_loop_name(Thread *t, Loop*l);

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
  switch(t.type) {
  case TypeEvent:
    {
      Event* e = thread->getEvent(t);
      size_t buffer_size = 1024;
      char * event_name = new char[buffer_size];  
      thread->printEventToString(e, event_name, buffer_size);
      std::string ret(event_name);
      delete[] event_name;
      return ret;
      break;
    }
  case TypeSequence:
    {
      Sequence* s = thread->getSequence(t);
      return guess_sequence_name(thread, s);
      break;
    }
  case TypeLoop:
    {
      Loop* l = thread->getLoop(t);
      return guess_loop_name(thread, l);
      break;
    }
  default:
    return std::string("Unknown token");
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
  size_t buffer_size = 1024;
  char * event_name = new char[buffer_size];  
  t->printEventToString(&e->event, event_name, buffer_size);

  std::cout << std::left<< "E"<<std::setw(14) <<std::left <<index;
  std::cout << std::setw(35) << std::left<< event_name;
  std::cout << std::setw(20) << std::right << e->durations->size;
  std::cout << std::setw(20) << std::right << (e->durations->min == UINT64_MAX? 0 : e->durations->min);
  std::cout << std::setw(20) << std::right << (e->durations->max == UINT64_MAX? 0 : e->durations->max);
  std::cout << std::setw(20) << std::right << (e->durations->mean == UINT64_MAX? 0 : e->durations->mean);
  std::cout << std::endl;

  delete[](event_name);
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
  std::cout << std::setw(18) << std::right << "Event_count";
  std::cout << std::endl;
}

std::string guess_sequence_name(Thread *t, Sequence*s) {
  if(s->size() < 4) {
    Token t_start = s->tokens[0];
    if(t_start.type == TypeEvent) {
      Event* event = t->getEvent(t_start);
      const char* event_name = t->getRegionStringFromEvent(event);
      std::string prefix(event_name);

      if(s->size() == 3) {
	// that's probably an MPI call. To differentiate calls (eg
	// MPI_Send(dest=5) vs MPI_Send(dest=0)), we can add the 
	// the second token to the name
	Token t_second = s->tokens[1];

	std::string res = prefix + "_" + t->getTokenString(t_second);
	return res;
      }
      return prefix;
    }
  }
  char buff[128];
  snprintf(buff, sizeof(buff), "Sequence_%d", s->id);
  
  return std::string(buff);
}

void info_sequence(Thread*t, int index, bool details=false) {
  Sequence* s = t->sequences[index];


  if(details) {
    info_sequence_header();
  }

  std::string sequence_name = guess_sequence_name(t, s);
  
  std::cout << std::left<< "S"<<std::setw(14) <<std::left <<index;
  std::cout << std::setw(35) << std::left<< sequence_name;
  std::cout << std::setw(18) << std::right << s->durations->size;
  std::cout << std::setw(18) << std::right << ns2s(s->durations->min == UINT64_MAX? 0 : s->durations->min);
  std::cout << std::setw(18) << std::right << ns2s(s->durations->max == UINT64_MAX? 0 : s->durations->max);
  std::cout << std::setw(18) << std::right << ns2s(s->durations->mean == UINT64_MAX? 0 : s->durations->mean);
  std::cout << std::setw(18) << std::right << ns2s(s->durations->mean == UINT64_MAX? 0 : s->durations->mean * s->durations->size);
  std::cout << std::setw(18) << std::right << s->size();
  // std::cout << std::setw(18) << std::right << s->getEventCount(t);
  std::cout << std::endl;

  if(details) {
    if(cmd & show_sequence_content) {
      std::cout<<std::endl<<"------------------- Sequence" << s->id << " contains:" << std::endl;
      for(auto token: s->tokens) {
	std::cout << "\t" << std::left << getTokenString(t, token) <<std::endl;
      }
      std::cout<<"------------------- End of sequence"<< s->id << std::endl;
      std::cout<<std::endl;
    }

    if(cmd & show_sequence_durations) {
      std::cout<<std::endl<<"------------------- Sequence" << s->id << " duration:" << std::endl;
      for(int i=0; i<s->durations->size; i++) {
	uint64_t duration = s->durations->at(i);
	std::cout<<"\t"<<duration<<std::endl;
      }
      std::cout<<std::endl<<"------------------- End of sequence" << s->id << " durations." << std::endl;
    }
    
  }
}

void info_loop_header() {
  std::cout << std::left << "#";
  std::cout << std::setw(14) << std::left << "Loop_id";
  std::cout << std::setw(35) << std::left << "Loop_name";
  std::cout << std::setw(18) << std::right << "Nb_occurence";
  std::cout << std::setw(18) << std::right << "Min_nb_iterations";
  std::cout << std::setw(18) << std::right << "Max_nb_iterations";
  std::cout << std::setw(18) << std::right << "Mean_nb_iterations";
  std::cout << std::endl;
}

std::string guess_loop_name(Thread *t, Loop* l) {
  Sequence *s = t->getSequence(l->repeated_token);
  return guess_sequence_name(t, s);
}

void info_loop(Thread* t, int index) {
  Loop* l = &t->loops[index];

  std::string loop_name = guess_loop_name(t, l);
  
  uint64_t min_iteration = INT_MAX;
  uint64_t max_iteration = 0;
  uint64_t mean_iteration = 0;
  for(auto iter: l->nb_iterations) {
    mean_iteration += iter;
    min_iteration = iter < min_iteration ? iter: min_iteration;
    max_iteration = iter > max_iteration ? iter: max_iteration;
  }
  mean_iteration = mean_iteration/l->nb_iterations.size();

  std::cout << std::left<< "L"<<std::setw(14) <<std::left <<index;
  std::cout << std::setw(35) << std::left<< loop_name;
  std::cout << std::setw(18) << std::right << l->nb_iterations.size();
  std::cout << std::setw(18) << std::right << min_iteration;
  std::cout << std::setw(18) << std::right << max_iteration;
  std::cout << std::setw(18) << std::right << mean_iteration;
  std::cout << std::endl;
}

void info_thread(Thread* t) {
  if(! _should_print_thread(t->id))
    return;

  if((cmd & show_thread_content) == 0)
    return;

  info_thread_header();
  info_thread_summary(t);

  printf("\nEvents {.nb_events: %d}\n", t->nb_events);
  info_event_header();
  for (unsigned i = 0; i < t->nb_events; i++) {
    info_event(t, i);
  }

  printf("\nSequences {.nb_sequences: %d}\n", t->nb_sequences);
  info_sequence_header();
  for (unsigned i = 0; i < t->nb_sequences; i++) {
    info_sequence(t, i);
  }
  
  printf("\nLoops {.nb_loops: %d}\n", t->nb_loops);
  info_loop_header();
  for (unsigned i = 0; i < t->nb_loops; i++) {
    info_loop(t, i);
  }

  if((cmd & show_sequence_content) ||
     (cmd & show_sequence_durations)) {
    info_sequence_header();
    for (unsigned i = 0; i < t->nb_sequences; i++) {
      info_sequence(t, i, true);
    }

  }
}

void info_global_archive(GlobalArchive* archive) {
  printf("Main archive:\n");

  if(cmd & show_archive_details) {
    printf("\tdir_name:   %s\n", archive->dir_name);
    printf("\ttrace_name: %s\n", archive->trace_name);
  }

  printf("\tfullpath:    %s\n", archive->fullpath);
  printf("\tnb_archives: %d\n", archive->nb_archives);
  printf("\tnb_process: %lu\n", archive->location_groups.size());
  printf("\tnb_threads: %lu\n", archive->locations.size());

  if(cmd & show_definitions) {
    if (!archive->definitions.strings.empty()) {
      printf("\tStrings {.nb_strings: %zu } :\n", archive->definitions.strings.size());

      for (auto& [stringRef, string] : archive->definitions.strings) {
	printf("\t\t%d: '%s'\n", string.string_ref, string.str);
      }
    }

    if (!archive->definitions.regions.empty()) {
      printf("\tRegions {.nb_regions: %zu } :\n", archive->definitions.regions.size());
      for (auto& [regionRef, region] : archive->definitions.regions) {
	printf("\t\t%d: %s\n", region.region_ref, archive->getString(region.string_ref)->str);
      }
    }

    if (!archive->definitions.groups.empty()) {
      printf("\tGroups {.nb_groups: %zu } :\n", archive->definitions.groups.size());
      for (auto& [groupRef, group] : archive->definitions.groups) {
	printf("\t\t%d: '%s' [", group.group_ref, archive->getString(group.name)->str);
	for(uint32_t i = 0; i<group.numberOfMembers; i++) {
	  printf("%s%lu", i>0?", ":"", group.members[i]);
	}
	printf("]\n");
      }
    }

    if (!archive->definitions.comms.empty()) {
      printf("\tComms {.nb_comms: %zu } :\n", archive->definitions.comms.size());
      for (auto& [commRef, comm] : archive->definitions.comms) {
	printf("\t\t%d: '%s' (group, %d, parent: %d) \n", comm.comm_ref, archive->getString(comm.name)->str, comm.group, comm.parent);
      }
    }

    if (!archive->location_groups.empty()) {
      printf("\tLocation_groups {.nb_lg: %zu }:\n", archive->location_groups.size());
      for (auto& locationGroup : archive->location_groups) {
	printf("\t\t%d: %s", locationGroup.id, archive->getString(locationGroup.name)->str);
	if (locationGroup.parent != PALLAS_LOCATION_GROUP_ID_INVALID)
	  printf(", parent: %d", locationGroup.parent);
	if (locationGroup.mainLoc != PALLAS_THREAD_ID_INVALID)
	  printf(", mainLocation: %d", locationGroup.mainLoc);
	printf("\n");
      }
    }

    if (!archive->locations.empty()) {
      printf("\tLocations {.nb_loc: %zu }:\n", archive->locations.size());
      for (auto location : archive->locations) {
	printf("\t\t%d: %s, parent: %d\n", location.id, archive->getString(location.name)->str, location.parent);
      }
    }

    if (archive->nb_archives)
      printf("\tArchives {.nb_archives: %d}\n", archive->nb_archives);
  }

  printf("\n");
}

static bool _archiveContainsThread(Archive* archive, int thread_id) {
  for(int i=0; i< archive->nb_threads; i++) {
    auto thread = archive->getThreadAt(i);
    if(thread->id == thread_id)
      return true;
  }
  return false;
}

void info_archive_header() {
  std::cout << std::left << "#";
  std::cout << std::setw(14) << std::left  << "Archive_id";
  std::cout << std::setw(20) << std::left  << "Archive_name";
  std::cout << std::setw(15) << std::right << "Nb_threads";
  std::cout << std::endl;
}

void info_archive(Archive* archive) {
  if(! _should_print_archive(archive->id)) {
    return;
  }

  std::cout << std::setw(15) << std::left  << archive->id;
  std::cout << std::setw(20) << std::left  << archive->getName();
  std::cout << std::setw(15) << std::right << archive->nb_threads;
  std::cout << std::endl;
}

void info_thread_header() {
  std::cout<< std::left << "#";
  std::cout << std::setw(19) << std::left  << "Thread_name";
  std::cout << std::setw(15) << std::left  << "Thread_id";
  std::cout << std::setw(15) << std::right <<"First_ts";
  std::cout << std::setw(15) << std::right <<"Last_ts";
  std::cout << std::setw(15) << std::right <<"Duration(s)";
  std::cout << std::setw(15) << std::right <<"Event_count";
  std::cout << std::setw(15) << std::right <<"Nb_events";
  std::cout << std::setw(15) << std::right <<"Nb_sequences";
  std::cout << std::setw(15) << std::right <<"Nb_loops";
  std::cout << std::setw(15) << std::right <<"Archive_id";
  std::cout << std::endl;
}

void info_thread_summary(Thread* thread) {
  if (thread && _should_print_thread(thread->id)) {
    std::cout << std::setw(20) << std::left  << thread->getName();
    std::cout << std::setw(15) << std::left  << thread->id;
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
  if(! (cmd & list_threads)) return ;
  if(thread_to_print >= 0 && ! _archiveContainsThread(archive, thread_to_print)) {
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

  if(cmd & list_archives) {
    info_archive_header();
    for (int i = 0; i < trace->nb_archives; i++) {
      info_archive(trace->archive_list[i]);
    }
  }

  if(cmd & list_threads) {
    info_thread_header();
    for (int i = 0; i < trace->nb_archives; i++) {
      info_threads(trace->archive_list[i]);
    }
  }

  if(cmd & show_thread_content) {
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
    } else if (!strcmp(argv[nb_opts], "-da")) {
      cmd |= show_archive_details;
    } else if (!strcmp(argv[nb_opts], "--archive")) {
      archive_to_print = atoi(argv[nb_opts+1]);
      nb_opts++;
    } else if (!strcmp(argv[nb_opts], "--thread")) {
      thread_to_print = atoi(argv[nb_opts+1]);
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

  auto trace = GlobalArchive();
  pallasReadGlobalArchive(&trace, trace_name);
  info_trace(&trace);

  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
