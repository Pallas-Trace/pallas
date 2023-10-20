/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <iostream>
#include <string>
#include "htf/htf.h"
#include "htf/htf_archive.h"
#include "htf/htf_read.h"
#include "htf/htf_storage.h"

static int show_structure = 0;
static int per_thread = 0;
static std::string structure_indent[MAX_CALLSTACK_DEPTH];
static short store_timestamps = 1;

static int print_timestamp = 1;
static int print_duration = 0;
static int unroll_loops = 0;
static int verbose = 0;

static void _print_timestamp(htf_timestamp_t ts) {
  if (print_timestamp) {
    printf("%21.9lf\t", ts / 1e9);
  }
}
static void _print_timestamp_header() {
  if (print_timestamp) {
    printf("%21s\t", "Timestamp");
  }
}

static void _print_duration(htf_timestamp_t d) {
  if (print_duration) {
    printf("%21.9lf\t", d / 1e9);
  }
}

static void _print_duration_header() {
  if (print_duration) {
    printf("%21s\t", "Duration");
  }
}

static void _print_indent(std::string current_indent) {
  std::cout << current_indent;
}

/* Print one event */
static void print_event(std::string current_indent, htf::Thread* thread, htf::Token token, htf::EventOccurence* e) {
  _print_timestamp(e->timestamp);
  _print_duration(e->duration);
  _print_indent(current_indent);

  if (!per_thread)
    std::cout << thread->getName() << "\t";
  if (verbose) {
    thread->printToken(token);
    std::cout << "\t";
  }
  thread->printEvent(e->event);
  thread->printEventAttribute(e);
  std::cout << "\t";
}

static void print_sequence(std::string current_indent,
                           htf::Thread* thread,
                           htf::Token token,
                           htf::SequenceOccurence* sequenceOccurence,
                           htf::LoopOccurence* containingLoopOccurence = nullptr) {
  auto* sequence = sequenceOccurence->sequence;
  htf_timestamp_t ts = sequenceOccurence->timestamp;
  htf_timestamp_t duration = sequenceOccurence->duration;

  _print_timestamp(ts);
  _print_duration(duration);
  _print_indent(current_indent);

  if (show_structure) {
    printf("Sequence ");
    thread->printToken(token);
  }

  if (!per_thread)
    printf("%sequence\t", thread->getName());

  if (verbose) {
    thread->printToken(token);
    printf("\t");
    for (unsigned i = 0; i < sequence->size(); i++) {
      thread->printToken(sequence->tokens[i]);
      std::cout << " ";
    }
  }

  if (containingLoopOccurence) {
    htf_timestamp_t mean = containingLoopOccurence->duration / containingLoopOccurence->nb_iterations;
    std::cout << ((mean < duration) ? "-" : "+");
    uint64_t diff = (mean < duration) ? duration - mean : mean - duration;
    float percentile = (float)diff / (float)mean * 100;
    printf("%5.2f%%", percentile);
  }
  std::cout << std::endl;
}

static void print_loop(std::string current_indent,
                       htf::Thread* thread,
                       htf::Token token,
                       htf::LoopOccurence* loopOccurence) {
  _print_timestamp(loopOccurence->timestamp);
  _print_duration(loopOccurence->duration);
  _print_indent(current_indent);

  if (show_structure) {
    std::cout << "Loop ";
  }

  if (!per_thread)
    std::cout << thread->getName() << "\t";

  auto* loop = loopOccurence->loop;

  thread->printToken(token);
  std::cout << "\t" << loopOccurence->nb_iterations << " * ";
  htf_print_token(thread, loop->repeated_token);
  std::cout << std::endl;
}

static void print_token(htf::Thread* thread,
                        htf::Token* t,
                        htf::Occurence* e,
                        int depth,
                        int last_one,
                        htf::LoopOccurence* containing_loop = nullptr) {
  htf_log(htf::Verbose, "Reading repeated_token(%x.%x) for thread %s\n", t->type, t->id, thread->getName());
  if (depth < 1)
    depth = 1;
  // Prints the structure of the sequences and the loops
  std::string current_indent;
  if (show_structure) {
    structure_indent[depth - 1] = (last_one ? "╰" : "├");
    DOFOR(i, depth) {
      current_indent += structure_indent[i];
    }
    if (depth) {
      if (t->type != htf::HTF_TYPE_EVENT) {
        current_indent += (containing_loop) ? "─" : "┬";
      } else {
        current_indent += "─";
      }
    }
    structure_indent[depth - 1] = last_one ? " " : "│";
  }

  // Prints the repeated_token we first started with
  switch (t->type) {
  case htf::HTF_TYPE_INVALID:
    htf_error("Type is invalid\n");
    break;
  case htf::HTF_TYPE_EVENT:
    print_event(current_indent, thread, *t, &e->event_occurence);
    break;
  case htf::HTF_TYPE_SEQUENCE: {
    if (show_structure)
      print_sequence(current_indent, thread, *t, &e->sequence_occurence, containing_loop);
    break;
  }
  case htf::HTF_TYPE_LOOP: {
    print_loop(current_indent, thread, *t, &e->loop_occurence);
    break;
  }
  }
}

static void display_sequence(htf::ThreadReader* reader,
                             htf::Token token,
                             htf::SequenceOccurence* occurence,
                             int depth) {
  if (occurence) {
    load_savestate(reader, occurence->savestate);
    reader->enterBlock(token);
  }
  auto current_level = reader->readCurrentLevel();
  for (const auto& tokenOccurence : current_level) {
    print_token(reader->thread_trace, tokenOccurence.token, tokenOccurence.occurence, depth,
                &tokenOccurence == &current_level.back());
    if (tokenOccurence.token->type == htf::HTF_TYPE_SEQUENCE) {
      display_sequence(reader, *tokenOccurence.token, &tokenOccurence.occurence->sequence_occurence, depth + 1);
    }
    if (tokenOccurence.token->type == htf::HTF_TYPE_LOOP) {
      htf::LoopOccurence loop = tokenOccurence.occurence->loop_occurence;

      if (unroll_loops) {
        DOFOR(j, (int)loop.nb_iterations) {
          /* at the jth iteration of the loop */
          htf::SequenceOccurence* seq = &loop.full_loop[j];
          display_sequence(reader, loop.loop->repeated_token, seq, depth + 1);
        }
      } else {
        htf::SequenceOccurence* seq = &loop.full_loop[0];
        display_sequence(reader, loop.loop->repeated_token, seq, depth + 1);
      }
    }
  }
  if (occurence) {
    occurence->full_sequence = current_level.data();
  }
  reader->leaveBlock();
}

/* Print all the events of a thread */
static void print_thread(htf::Archive* trace, htf::Thread* thread) {
  printf("Reading events for thread %u (%s):\n", thread->id, thread->getName());
  _print_timestamp_header();
  _print_duration_header();

  printf("Event\n");

  int reader_options = htf::ThreadReaderOptions::None;
  if (show_structure)
    reader_options |= htf::ThreadReaderOptions::ShowStructure;
  if (!store_timestamps || trace->store_timestamps == 0)
    reader_options |= htf::ThreadReaderOptions::NoTimestamps;

  auto* reader = new htf::ThreadReader(trace, thread->id, reader_options);

  display_sequence(reader, htf::Token(htf::HTF_TYPE_SEQUENCE, 0), nullptr, 0);
}

/** Compare the timestamps of the current repeated_token on each thread and select the smallest timestamp.
 * This fills the Token t, and the event_occurence e if it's an event,
 * and returns the index of the selected thread (or -1 at the end of the trace).
 */
// static int get_next_token(struct htf_thread_reader* readers, int nb_threads, struct htf_token* t, htf_occurence* e) {
//   struct htf_token cur_t;
//   htf_timestamp_t min_ts = HTF_TIMESTAMP_INVALID;
//   int min_index = -1;
//
//  for (int i = 0; i < nb_threads; i++) {
  //    if (htf_read_thread_cur_token(&readers[i], &cur_t, NULL) == 0) {
  //      htf_timestamp_t ts = htf_get_starting_timestamp(&readers[i], cur_t);
  //      if (min_ts == HTF_TIMESTAMP_INVALID || ts < min_ts) {
  //        min_index = i;
  //        min_ts = ts;
  //      }
  //    }
  //  }

  //  if (min_index >= 0) {
  //    htf_read_thread_cur_token(&readers[min_index], t, e);
  //    htf_move_to_next_token(&readers[min_index]);
  //  }
//
//  return min_index;
//}

/* Print all the events of all the threads sorted by timestamp */
// void print_trace(struct htf_archive* trace) {
//   struct htf_thread_reader** readers = malloc(sizeof(struct htf_thread_reader*) * (trace->nb_threads));
//   int reader_options = htf_thread_reader_option_none;
//   if (show_structure)
//     reader_options |= htf_thread_reader_option_show_structure;
//
//   for (int i = 0; i < trace->nb_threads; i++) {
//     readers[i] = htf_new_thread_reader(trace, trace->threads[i]->id, reader_options);
//   }
//
//   printf("Timestamp\tDuration\tThread Name\tEvent\n");
//
//	htf_occurence e;
  //	struct Token t;
  //	int thread_index = -1;
  //	while ((thread_index = get_next_token(readers, trace->nb_threads, &t, &e)) >= 0) {
  //		print_token(&readers[thread_index], &t, &e);
  //	}
//}

void usage(const char* prog_name) {
  printf("Usage: %s [OPTION] trace_file\n", prog_name);
  printf("\t-T          Print events per thread\n");
  printf("\t-S          Structure mode\n");
  printf("\t-v          Verbose mode\n");
  printf("\t-d          Print duration of events\n");
  printf("\t-t          Don't print timestamps\n");
  printf("\t-u          Unroll loops\n");
  printf("\t-?  -h      Display this help and exit\n");
}

int main(int argc, char** argv) {
  int nb_opts = 0;
  char* trace_name = NULL;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-v")) {
      htf_debug_level_set(htf::DebugLevel::Debug);
      nb_opts++;
    } else if (!strcmp(argv[i], "-T")) {
      per_thread = 1;
      nb_opts++;
    } else if (!strcmp(argv[i], "-S")) {
      show_structure = 1;
      nb_opts++;
    } else if (!strcmp(argv[i], "-d")) {
      print_duration = 1;
      nb_opts++;
    } else if (!strcmp(argv[i], "-t")) {
      print_timestamp = 0;
      nb_opts++;
    } else if (!strcmp(argv[i], "-u")) {
      unroll_loops = 1;
      nb_opts++;
    } else if (!strcmp(argv[i], "--no-timestamps")) {
      setenv("STORE_TIMESTAMPS", "FALSE", 0);
      print_timestamp = 0;
      store_timestamps = 0;
      nb_opts++;
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?")) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      /* Unknown parameter name. It's probably the trace's path name. We can stop
       * parsing the parameter list.
       */
      break;
    }
  }

  if (!show_structure) {
    unroll_loops = 1;
  }

  trace_name = argv[nb_opts + 1];
  if (trace_name == NULL) {
    usage(argv[0]);
    return EXIT_SUCCESS;
  }

  auto* trace = new htf::Archive();
  htf_read_archive(trace, trace_name);
  store_timestamps = trace->store_timestamps;

  if (per_thread) {
    for (int i = 0; i < trace->nb_threads; i++) {
      printf("\n");
      print_thread(trace, trace->threads[i]);
    }
  } else {
    htf_error("This is currently buggy so don't do it\n");
    //    print_trace(trace);
  }

  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
