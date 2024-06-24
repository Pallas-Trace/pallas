/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <algorithm>
#include <iostream>
#include <string>
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"

static bool show_structure = false;
static bool per_thread = false;
static std::string structure_indent[MAX_CALLSTACK_DEPTH];
static short store_timestamps = 1;

static bool print_timestamp = true;
static bool print_duration = false;
static bool unroll_loops = true;
static bool explore_loop_sequences = true;
static bool verbose = false;

static void _print_timestamp(pallas_timestamp_t ts) {
  if (print_timestamp) {
    printf("%21.9lf\t", ts / 1e9);
  }
}
static void _print_timestamp_header() {
  if (print_timestamp) {
    printf("%21s\t", "Timestamp");
  }
}

static void _print_duration(pallas_timestamp_t d) {
  if (print_duration) {
    printf("%21.9lf\t", d / 1e9);
  }
}

static void _print_duration_header() {
  if (print_duration) {
    printf("%21s\t", "Duration");
  }
}

static void _print_indent(const std::string& current_indent) {
  std::cout << current_indent;
}

/* Print one event */
static void printEvent(const std::string& current_indent,
                       const pallas::Thread* thread,
                       const pallas::Token token,
                       const pallas::EventOccurence* e) {
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
  std::cout << std::endl;
}

static void printSequence(const std::string& current_indent,
                          const pallas::Thread* thread,
                          const pallas::Token token,
                          const pallas::SequenceOccurence* sequenceOccurence) {
  auto* sequence = sequenceOccurence->sequence;
  pallas_timestamp_t ts = sequenceOccurence->timestamp;
  pallas_timestamp_t duration = sequenceOccurence->duration;

  _print_timestamp(ts);
  _print_duration(duration);
  _print_indent(current_indent);

  if (!per_thread)
    std::cout << thread->getName() << "\t";
  if (show_structure) {
    std::cout << "Sequence ";
    thread->printToken(token);
  }

  if (verbose) {
    thread->printToken(token);
    printf("\t");
    for (unsigned i = 0; i < sequence->size(); i++) {
      thread->printToken(sequence->tokens[i]);
      std::cout << " ";
    }
  }
  if (sequence->isFunctionSequence(thread)) {
    auto frontEvent = thread->getEvent(sequence->tokens.front());
    pallas::RegionRef region_ref;
    memcpy(&region_ref, &frontEvent->event_data[0], sizeof(region_ref));
    const pallas::Region* region = thread->archive->getRegion(region_ref);
    const char* region_name = region ? thread->archive->getString(region->string_ref)->str : "INVALID";
    std::cout << " (" << region_name << ")";
  }
  std::cout << std::endl;
}

static void printLoop(const std::string& current_indent,
                      const pallas::Thread* thread,
                      const pallas::Token token,
                      const pallas::LoopOccurence* loopOccurence) {
  _print_timestamp(loopOccurence->timestamp);
  _print_duration(loopOccurence->duration);
  _print_indent(current_indent);

  if (!per_thread)
    std::cout << thread->getName() << "\t";

  if (show_structure) {
    std::cout << "Loop ";
  }

  auto* loop = loopOccurence->loop;

  thread->printToken(token);
  std::cout << "\t" << loopOccurence->nb_iterations << " * ";
  thread->printToken(loop->repeated_token);
  std::cout << std::endl;
}

static void printToken(const pallas::Thread* thread,
                       const pallas::Token* t,
                       const pallas::Occurence* e,
                       int depth = 0,
                       bool isLastOfSeq = false,
                       bool isInLoop = false) {
  pallas_log(pallas::DebugLevel::Verbose, "Reading repeated_token(%d.%d) for thread %s\n", t->type, t->id,
             thread->getName());
  // Prints the structure of the sequences and the loops
  std::string current_indent;
  if (show_structure && depth >= 1) {
    structure_indent[depth - 1] = (isLastOfSeq ? "╰" : "├");
    DOFOR(i, depth) {
      current_indent += structure_indent[i];
    }
    if (t->type != pallas::TypeEvent) {
      current_indent += ((t->type == pallas::TypeSequence && e->sequence_occurence.sequence->isFunctionSequence(thread))
                         || (isInLoop && !explore_loop_sequences)) ? "─" : "┬";
    } else {
      current_indent += "─";
    }
    structure_indent[depth - 1] = isLastOfSeq ? " " : "│";
  }

  // Prints the repeated_token we first started with
  switch (t->type) {
  case pallas::TypeInvalid:
    pallas_error("Type is invalid\n");
    break;
  case pallas::TypeEvent:
    printEvent(current_indent, thread, *t, &e->event_occurence);
    break;
  case pallas::TypeSequence: {
    if (show_structure)
      printSequence(current_indent, thread, *t, &e->sequence_occurence);
    break;
  }
  case pallas::TypeLoop: {
    if (show_structure)
      printLoop(current_indent, thread, *t, &e->loop_occurence);
    break;
  }
  }
}

/* Print all the events of a thread */
static void printThread(pallas::Archive& trace, pallas::Thread* thread) {
  printf("Reading events for thread %u (%s):\n", thread->id, thread->getName());
  _print_timestamp_header();
  _print_duration_header();

  printf("Event\n");

  int reader_options = pallas::ThreadReaderOptions::None;
  if (show_structure)
    reader_options |= pallas::ThreadReaderOptions::ShowStructure;
  if (!store_timestamps || !trace.store_timestamps)
    reader_options |= pallas::ThreadReaderOptions::NoTimestamps;

  auto reader = pallas::ThreadReader(&trace, thread->id, reader_options);

  while (reader.current_frame >= 0) {
    auto& token = reader.getCurToken();
    auto* occurrence = reader.getOccurence(token, reader.tokenCount[token]);

    bool printThisToken = true;
    auto curIterable = reader.getCurIterable();
    if (show_structure && curIterable.type == pallas::TypeSequence) {
      auto curSequence = reader.thread_trace->getSequence(curIterable);
      if (curSequence->isFunctionSequence(reader.thread_trace)) {
        printThisToken = false;
      }
    }
    if (printThisToken)
      printToken(reader.thread_trace, &token, occurrence, reader.current_frame, reader.isLastInCurrentArray(),
                 reader.isInLoop());
    // Update the reader
    reader.updateReadCurToken();
    if (token.type == pallas::TypeEvent)
      reader.moveToNextToken();
    delete occurrence;
  }
}

/**
 * Compare the timestamps of the current token on each thread and select the smallest timestamp.
 * Sets the values of tokenOccurence.
 * @returns The ThreadId of the earliest thread.
 *          You are responsible for the memory of the TokenOccurence.
 */
static pallas::ThreadId getNextToken(std::vector<pallas::ThreadReader>& threadReaders,
                                     pallas::TokenOccurence& tokenOccurence) {
  // Find the earliest threadReader
  pallas::ThreadReader* earliestReader = nullptr;
  for (auto& reader : threadReaders) {
    // Check if reader has finished reading its trace
    if (reader.current_frame < 0)
      continue;
    if (!earliestReader || earliestReader->referential_timestamp > reader.referential_timestamp) {
      earliestReader = &reader;
    }
  }

  // If no reader was available
  if (!earliestReader) {
    return PALLAS_THREAD_ID_INVALID;
  }

  // Grab the interesting information
  auto& token = earliestReader->getCurToken();
  auto* occurrence = earliestReader->getOccurence(token, earliestReader->tokenCount[token]);
  auto threadId = earliestReader->thread_trace->id;

  // Update the reader
  earliestReader->updateReadCurToken();
  if (token.type == pallas::TypeEvent)
    earliestReader->moveToNextToken();

  tokenOccurence.token = &token;
  tokenOccurence.occurence = occurrence;
  return threadId;
}

/** Print all the events of all the threads sorted by timestamp*/
void printTrace(pallas::Archive& trace) {
  auto readers = std::vector<pallas::ThreadReader>();
  int reader_options = pallas::ThreadReaderOptions::None;
  //  if (show_structure)
  //    reader_options |= pallas::ThreadReaderOptions::ShowStructure;
  readers.reserve(trace.nb_threads);
  for (int i = 0; i < trace.nb_threads; i++) {
    readers.emplace_back(&trace, trace.threads[i]->id, reader_options);
  }

  _print_timestamp_header();
  _print_duration_header();
  printf("Event\n");
  pallas::TokenOccurence tokenOccurence;
  pallas::ThreadId threadId = getNextToken(readers, tokenOccurence);
  while (threadId != PALLAS_THREAD_ID_INVALID) {
    auto currentReader = std::find_if(readers.begin(), readers.end(), [&threadId](const pallas::ThreadReader& reader) {
      return reader.thread_trace->id == threadId;
    });
    printToken(currentReader->thread_trace, tokenOccurence.token, tokenOccurence.occurence);
    // If you read the doc, you'll know that the memory of tokenOccurence is ours to manage
    delete tokenOccurence.occurence;
    tokenOccurence.occurence = nullptr;
    threadId = getNextToken(readers, tokenOccurence);
  }
}

void usage(const char* prog_name) {
  printf("Usage: %s [OPTION] trace_file\n", prog_name);
  printf("\t-T          Print events per thread\n");
  printf("\t-S          Structure mode\n");
  printf("\t-v          Verbose mode\n");
  printf("\t-d          Print duration of events\n");
  printf("\t-t          Don't print timestamps\n");
  printf("\t-u          Unroll loops\n");
  printf("\t-e          Explore sequences inside of loops\n");
  printf("\t-?  -h      Display this help and exit\n");
}

int main(int argc, char** argv) {
  int nb_opts = 0;
  char* trace_name = NULL;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-v")) {
      pallas_debug_level_set(pallas::DebugLevel::Debug);
      nb_opts++;
    } else if (!strcmp(argv[i], "-T")) {
      per_thread = true;
      nb_opts++;
    } else if (!strcmp(argv[i], "-d")) {
      print_duration = true;
      nb_opts++;
    } else if (!strcmp(argv[i], "-t")) {
      print_timestamp = false;
      nb_opts++;
    } else if (!strcmp(argv[i], "-S")) {
      per_thread = true;
      show_structure = true;
      print_timestamp = false;
      unroll_loops = false;
      explore_loop_sequences = true;
      nb_opts++;
    } else if (show_structure && !strcmp(argv[i], "-u")) {
      unroll_loops = true;
      nb_opts++;
    } else if (show_structure && !strcmp(argv[i], "-e")) {
      explore_loop_sequences = true;
      nb_opts++;
    } else if (!strcmp(argv[i], "--no-timestamps")) {
      setenv("STORE_TIMESTAMPS", "FALSE", 0);
      print_timestamp = true;
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

  trace_name = argv[nb_opts + 1];
  if (trace_name == NULL) {
    usage(argv[0]);
    return EXIT_SUCCESS;
  }

  auto trace = pallas::Archive();
  pallas_read_main_archive(&trace, trace_name);
  if (trace.store_timestamps == 0)
    store_timestamps = 0;

  if (per_thread) {
    for (int i = 0; i < trace.nb_threads; i++) {
      printf("\n");
      printThread(trace, trace.threads[i]);
    }
  } else {
    printTrace(trace);
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
