/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <iostream>
#include <limits>
#include <string>
#include <iomanip>

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_read.h"

#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_storage.h"

bool verbose = false;
bool per_thread = false;
bool show_durations = false;
bool show_timestamps = true;
int thread_to_print = -1;
bool flamegraph = false;
bool csv = false;
bool csv_bulk = false;
int nb_intervals = 0;

static void _print_timestamp(pallas_timestamp_t ts) {
  if (show_timestamps) {
    std::cout.precision(9);
    std::cout << std::right << std::setw(21) << std::fixed << ts / 1e9;
  }
}
static void _print_timestamp_header() {
  if (show_timestamps && (!flamegraph) && (!csv) && (!csv_bulk) ) {
    std::cout << std::right << std::setw(21) << "Timestamp";
  }
}

static void _print_duration(pallas_timestamp_t d) {
  if (show_durations) {
    std::cout.precision(9);
    std::cout << std::right << std::setw(21) << std::fixed << d / 1e9;
  }
}

static void _print_duration_header() {
  if (show_durations && (!flamegraph) && (!csv)) {
    std::cout << std::right << std::setw(21) << "Duration";
  }
}

/* Print one event */
static void printEvent(const pallas::Thread* thread, const pallas::Token token, const pallas::EventOccurence e) {
  _print_timestamp(e.timestamp);

  if (!per_thread)
    std::cout << std::right << std::setw(10) << thread->getName();
  if (verbose) {
    std::cout << std::right << std::setw(10) << thread->getTokenString(token);
  }
  std::cout << std::setw(4) << " " << thread->getEventString(e.event);
  thread->printEventAttribute(&e);
  std::cout << std::endl;
}

bool isReadingOver(const std::vector<pallas::ThreadReader>& readers) {
  for (const auto& reader : readers) {
    if (!reader.isEndOfTrace()) {
      return false;
    }
  }
  return true;
}

struct thread_data {
  std::vector<std::string> callstack{};
  std::vector<pallas_duration_t> callstack_duration{};
  std::vector<pallas_timestamp_t> callstack_timestamp{};
  pallas_timestamp_t last_timestamp;
};

void printFlame(std::map<pallas::ThreadReader*, struct thread_data> &threads_data,
		pallas::ThreadReader* min_reader,
		pallas::EventOccurence& e) {

  // This lambda prints the callstack in the flamegraph format
  auto  _print_callstack = [&]() {
    auto duration = 0;
    if(! threads_data[min_reader].callstack_duration.empty()) {
      duration = threads_data[min_reader].callstack_duration.back();
    }
    for(auto str: threads_data[min_reader].callstack) {
      std::cout<<";"<<str;
    }
    if(threads_data[min_reader].callstack.empty()) std::cout<<";";

    std::cout<<" "<<duration<<std::endl;
  };

  if(e.event->record == pallas::PALLAS_EVENT_ENTER) {
    // Start a new frame
    const char* function_name = min_reader->thread_trace->getRegionStringFromEvent(e.event);

    // Print the previous frame
    _print_callstack();

    // Start a new callstack duration
    threads_data[min_reader].callstack.emplace_back(std::string(function_name));
    // FIXME THIS SHOULD BE e.durations but it doesn't work yet
    threads_data[min_reader].callstack_duration.emplace_back(0);

  } else if(e.event->record == pallas::PALLAS_EVENT_LEAVE) {
    // End a frame
    const char* function_name = min_reader->thread_trace->getRegionStringFromEvent(e.event);
    // Print the current frame
    _print_callstack();

    // Start a new frame

    threads_data[min_reader].callstack.pop_back();
    threads_data[min_reader].callstack_duration.pop_back();
    if(threads_data[min_reader].callstack_duration.empty()) {
        threads_data[min_reader].callstack_duration.push_back(0);
    }
    // FIXME this should be e.duration
    threads_data[min_reader].callstack_duration.back() = 0; // reset the counter

  } else {
    // Accumulate duration in the current frame
    if(threads_data[min_reader].callstack_duration.empty()) threads_data[min_reader].callstack_duration.push_back(0);
    // FIXME this should be e.duration
    threads_data[min_reader].callstack_duration.back() += 0;
  }


}

void printCSV(std::map<pallas::ThreadReader*, struct thread_data> &threads_data,
	      pallas::ThreadReader* min_reader,
	      pallas::EventOccurence& e) {

  // This lambda prints the callstack in the flamegraph format
  auto  _print_callstack = [&]() {
    pallas_duration_t duration = 0;
    if(! threads_data[min_reader].callstack_duration.empty()) {
      duration = threads_data[min_reader].callstack_duration.back();
    }

    pallas_timestamp_t first_timestamp = min_reader->thread_trace->getFirstTimestamp();
    if(! threads_data[min_reader].callstack_timestamp.empty()) {
      first_timestamp = threads_data[min_reader].callstack_timestamp.back();
    }


    static bool first_line = true;
    if(first_line) {
      std::cout<<"Thread,Function,Start,Finish,Duration\n";
      first_line = false;
    }

    std::cout<<min_reader->thread_trace->getName()<<",";
    if(threads_data[min_reader].callstack.empty())
      std::cout<<"main";
    else
      std::cout<<threads_data[min_reader].callstack.back();

    std::cout<<","<<first_timestamp<<","<<first_timestamp+duration<<","<<duration<<std::endl;

     // Check that timestamps to not overlap
    pallas_assert_always(threads_data[min_reader].last_timestamp <= first_timestamp);
    threads_data[min_reader].last_timestamp = first_timestamp + duration;
  };

  if(e.event->record == pallas::PALLAS_EVENT_ENTER) {
    // Start a new frame
    const char* function_name = min_reader->thread_trace->getRegionStringFromEvent(e.event);

    // Print the previous frame
    _print_callstack();

    // Start a new callstack duration
    threads_data[min_reader].callstack.emplace_back(std::string(function_name));
    // FIXME this should be e.duration
    threads_data[min_reader].callstack_duration.emplace_back(0);
    threads_data[min_reader].callstack_timestamp.emplace_back(e.timestamp);

  } else if(e.event->record == pallas::PALLAS_EVENT_LEAVE) {
    // End a frame
    const char* function_name = min_reader->thread_trace->getRegionStringFromEvent(e.event);
    // Print the current frame
    _print_callstack();

    // Start a new frame

    threads_data[min_reader].callstack.pop_back();
    threads_data[min_reader].callstack_duration.pop_back();
    if(threads_data[min_reader].callstack_duration.empty()) threads_data[min_reader].callstack_duration.push_back(0);
    // FIXME this should be e.duration
    threads_data[min_reader].callstack_duration.back() = 0; // reset the counter

    threads_data[min_reader].callstack_timestamp.pop_back();
    if(threads_data[min_reader].callstack_timestamp.empty()) threads_data[min_reader].callstack_timestamp.push_back(0);
    threads_data[min_reader].callstack_timestamp.back() = e.timestamp; // reset the counter

  } else {
    // Accumulate duration in the current frame
    if(threads_data[min_reader].callstack_duration.empty()) threads_data[min_reader].callstack_duration.push_back(0);
    // FIXME this should be e.duration
    threads_data[min_reader].callstack_duration.back() += 0;

  }
}

/** Print the trace as a CSV. Each line contains:
 *   - Thread: the thread that called a function
 *   - Function: the name of the function
 *   - Start: the timestamp of the begining of the function
 *   - Finish: the timestamp of the end of the function
 *   - Duration: the duration of the function
 *
 * Unlike with printCSV, the function calls timestamp may overlap: if
 * foo calls bar, this will look like this:
 * T0,foo,17,25,8
 * T0,bar,19,22,3
 *
 * Moreover, the timestamps are not sorted
 */
void printCSVBulk(std::vector<pallas::ThreadReader> readers) {
  static bool first_line = true;
  for (auto & reader : readers) {
    std::map<pallas::Sequence*, std::string> sequence_names;
    reader.guessSequencesNames(sequence_names);

    // iterate over the sequences (ignoring sequence 0), and dump their timestamps
    for(int i=1; i<reader.thread_trace->nb_sequences; i++) {
      auto* s = &reader.thread_trace->sequences[i];
      const auto &seq_name = sequence_names[s];

      pallas_duration_t duration = s->durations->at(0);
      pallas_timestamp_t ts = s->timestamps->at(0);

      for(int occurence_id = 0; occurence_id < s->durations->size; occurence_id++) {
	pallas_duration_t duration = s->durations->at(occurence_id);
	pallas_timestamp_t ts = s->timestamps->at(occurence_id);

	if(first_line) {
	  std::cout<<"Thread,Function,Start,Finish,Duration\n";
	  first_line = false;
	}

	std::cout<<reader.thread_trace->getName()<<",";
	std::cout<<seq_name<<","<<ts<<","<<ts+duration<<","<<duration<<"\n";
      }
    }
  }
}

void printThread(pallas::Thread* thread) {
    size_t last_timestamp = 0;
    if (thread_to_print >= 0 && thread->id != thread_to_print) {
        return;
    }
    auto reader = pallas::ThreadReader(thread->archive, thread->id, PALLAS_READ_FLAG_UNROLL_ALL);
    _print_timestamp_header();
    _print_duration_header();
    std::cout << std::endl;
    auto current_token = reader.pollCurToken();
    while (current_token.isValid()) {
        if (current_token.type == pallas::TypeEvent) {
            auto event = reader.getEventOccurence(current_token, reader.currentState.currentFrame->tokenCount[current_token]);
            pallas_assert_inferior_equal(last_timestamp, event.timestamp);
            pallas_assert_equals(event.timestamp, reader.currentState.currentFrame->current_timestamp);
            last_timestamp = event.timestamp;
            printEvent(reader.thread_trace, current_token, event);
        }
        current_token = reader.getNextToken();
    }
}

void printIntervals(pallas::GlobalArchive& trace) {
  printf("PrintInterval(nb_intervals=%d)\n", nb_intervals);
  for (auto* thread : trace.getThreadList()) {

    pallas_timestamp_t start = thread->getFirstTimestamp();
    pallas_timestamp_t end = thread->getLastTimestamp();
    std::map<pallas::Token, pallas_duration_t> m = thread->getSnapshotViewFast(start, end);

    pallas_duration_t interval_duration = end-start;
    pallas_duration_t sum_duration = 0;
    for (auto it = m.begin(); it != m.end(); it++) {
      pallas::Token t = it->first;
      pallas::Sequence *seq = thread->getSequence(t);
      pallas_duration_t d = it->second;
      sum_duration += d;
      std::cout << thread->getTokenString(t) << ": "<<d<<"\t"<< seq->guessName(thread)<<"\n";
    }

    std::cout<<"total duration:    " <<sum_duration<<"\n";
    std::cout<<"Interval duration: " <<interval_duration<<"\n";
    std::cout<<"Start timestamp:   " <<start<<"\n";
    std::cout<<"End timestamp:     " <<end<<"\n";
    return;
  }
  return;
}

void printTrace(pallas::GlobalArchive& trace) {
    if (per_thread) {
        for (auto* thread : trace.getThreadList()) {
            printThread(thread);
        }
        return;
    }
    std::map<pallas::ThreadReader*, struct thread_data> threads_data;

    auto readers = std::vector<pallas::ThreadReader>();
    auto thread_list = trace.getThreadList();
    for (auto* thread : thread_list) {
        if (thread == nullptr)
            continue;
        if (!(thread_to_print < 0 || thread->id == thread_to_print))
            continue;
        readers.emplace_back(thread->archive, thread->id, PALLAS_READ_FLAG_UNROLL_ALL);
        threads_data[&readers.back()] = {};
    }

    _print_timestamp_header();
    _print_duration_header();
    std::cout << std::endl;

    if (csv_bulk) {
        printCSVBulk(readers);
        return;
    }

    while (!isReadingOver(readers)) {
        pallas::ThreadReader* min_reader = &readers[0];
        pallas_timestamp_t min_timestamp = std::numeric_limits<unsigned long>::max();
        for (auto& reader : readers) {
            if (!reader.isEndOfTrace() && reader.currentState.currentFrame->current_timestamp < min_timestamp) {
                min_reader = &reader;
                min_timestamp = reader.currentState.currentFrame->current_timestamp;
            }
        }

        auto token = min_reader->pollCurToken();
        if (token.type == pallas::TypeEvent) {
            if (flamegraph) {
                auto e = min_reader->getEventOccurence(token, min_reader->currentState.currentFrame->tokenCount[token]);
                printFlame(threads_data, min_reader, e);
            } else if (csv) {
                auto e = min_reader->getEventOccurence(token, min_reader->currentState.currentFrame->tokenCount[token]);
                printCSV(threads_data, min_reader, e);
            } else {
                printEvent(min_reader->thread_trace, token, min_reader->getEventOccurence(token, min_reader->currentState.currentFrame->tokenCount[token]));
            }
        }

        if (!min_reader->getNextToken().isValid()) {
            pallas_assert(min_reader->isEndOfTrace());
        }
    }
}

static std::string structure_indent[MAX_CALLSTACK_DEPTH];
std::string getCurrentIndent(const pallas::ThreadReader& tr) {
  if (tr.currentState.current_frame_index <= 1) {
    return "";
  }
  const auto t = tr.pollCurToken();
  std::string current_indent;
  bool isLastOfSeq = tr.isEndOfCurrentBlock();
  structure_indent[tr.currentState.current_frame_index - 2] = (isLastOfSeq ? "╰" : "├");
    DOFOR(i, tr.currentState.current_frame_index - 1) {
      current_indent += structure_indent[i];
    }
    if (t.type != pallas::TypeEvent) {
      if (t.type == pallas::TypeSequence && tr.thread_trace->getSequence(t)->type == pallas::SEQUENCE_BLOCK) {
        current_indent += "┬"; //"─";
      } else {
        current_indent += "┬";
      }
//      current_indent += ((t.type == pallas::TypeSequence && e->sequence_occurence.sequence->isFunctionSequence(thread))
//                         || (isInLoop && !explore_loop_sequences)) ? "─" : "┬";
    } else {
      current_indent += "─";
    }
    structure_indent[tr.currentState.current_frame_index - 2] = isLastOfSeq ? " " : "│";
    return current_indent;
}

void printThreadStructure(pallas::ThreadReader& tr) {
    std::cout << "--- Thread " << tr.thread_trace->id << "(" << tr.thread_trace->getName() << ")" << " ---" << std::endl;
    size_t last_timestamp = 0;
    auto current_token = tr.pollCurToken();
    while (true) {
        std::cout << getCurrentIndent(tr) << std::left << std::setw(15 - ((tr.currentState.current_frame_index <= 1) ? 0 : tr.currentState.current_frame_index))
                  << tr.thread_trace->getTokenString(current_token) << "";
        if (current_token.type == pallas::TypeEvent) {
            auto occ = tr.getEventOccurence(current_token, tr.currentState.currentFrame->tokenCount[current_token]);
            pallas_assert_inferior_equal(last_timestamp, occ.timestamp);
            pallas_assert_equals(occ.timestamp, tr.currentState.currentFrame->current_timestamp);
            last_timestamp = occ.timestamp;
            printEvent(tr.thread_trace, current_token, occ);
        } else if (current_token.type == pallas::TypeSequence) {
            auto occ = tr.getSequenceOccurence(current_token, tr.currentState.currentFrame->tokenCount[current_token]);
            pallas_assert_inferior_equal(last_timestamp, occ.timestamp);
            pallas_assert_equals(occ.timestamp, tr.currentState.currentFrame->current_timestamp);
            last_timestamp = occ.timestamp;
            _print_timestamp(occ.timestamp);
            if (show_durations) {
                auto d = tr.thread_trace->getSequence(current_token)->durations->at(tr.currentState.currentFrame->tokenCount[current_token]);
                std::cout << std::setw(21) << "";
                std::cout.precision(9);
                std::cout << std::right << std::setw(21) << std::fixed << d / 1e9;
            }
            std::cout << std::endl;
        } else if (current_token.type == pallas::TypeLoop) {
            if (show_durations) {
                auto d = tr.getLoopDuration(current_token);
                std::cout << std::setw(21) << "";
                std::cout.precision(9);
                std::cout << std::right << std::setw(21) << std::fixed << d / 1e9;
            }
            std::cout << std::endl;
        }
        auto next_token = tr.getNextToken();
        if (!next_token.isValid())
            break;
        current_token = next_token;
    }
}

void printStructure(const int flags, pallas::GlobalArchive& trace) {
  for (auto * thread: trace.getThreadList()) {
      if(thread_to_print >= 0 && thread->id != thread_to_print) continue;
      auto tr = pallas::ThreadReader(
        thread->archive,
        thread->id,
        flags
        );
      printThreadStructure(tr);
  }
}

void usage(const char* prog_name) {
  std::cout << "Usage : " << prog_name << " [options] <trace file>" << std::endl;
  std::cout << "\t" << "-h" << "\t" << "Show this help and exit" << std::endl;
  std::cout << "\t" << "-v" << "\t" << "Enable verbose mode" << std::endl;
  std::cout << "\t" << "-T" << "\t" << "Enable per thread mode" << std::endl;
  std::cout << "\t" << "-d" << "\t" << "Show durations" << std::endl;
  std::cout << "\t" << "-t" << "\t" << "Hide timestamps" << std::endl;
  std::cout << "\t" << "-S" << "\t" << "Enable structure mode (per thread mode only)" << std::endl;
  std::cout << "\t" << "-s" << "\t" << "Do not unroll sequences (structure mode only)" << std::endl;
  std::cout << "\t" << "-l" << "\t" << "Do not unroll loops (structure mode only)" << std::endl;
  std::cout << "\t" << "--thread thread_id" << "\t" << "Only print thread <thread_id>" << std::endl;
  std::cout << "\t" << "-i <n>" << "\t" << "Split the trace in n intervals and prints profiles for each interval" << std::endl;
  std::cout << "\t" << "-f" << "\t" << "Generate a flamegraph file" << std::endl;
  std::cout << "\t" << "-c" << "\t" << "Generate a csv file" << std::endl;
  std::cout << "\t" << "-cb"<< "\t" << "Generate a csv file (bulk mode)" << std::endl;
}

int main(const int argc, char* argv[]) {
  int flags = PALLAS_READ_FLAG_UNROLL_ALL;
  bool show_structure = false;
  char* trace_name = nullptr;

  for (int nb_opts = 1; nb_opts < argc; nb_opts++) {
    if (!strcmp(argv[nb_opts], "-v")) {
      pallas_debug_level_set(pallas::DebugLevel::Debug);
    } else if (!strcmp(argv[nb_opts], "-h") || !strcmp(argv[nb_opts], "-?")) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else if (!strcmp(argv[nb_opts], "-v")) {
      pallas_debug_level_set(pallas::DebugLevel::Debug);
    } else if (!strcmp(argv[nb_opts], "-T")) {
      per_thread = true;
    } else if (!strcmp(argv[nb_opts], "-d")) {
      show_durations = true;
    } else if (!strcmp(argv[nb_opts], "-t")) {
      show_timestamps = false;
    } else if (!strcmp(argv[nb_opts], "-S")) {
      show_structure = true;
    } else if (!strcmp(argv[nb_opts], "-s")) {
      flags &= ~PALLAS_READ_FLAG_UNROLL_SEQUENCE;
    } else if (!strcmp(argv[nb_opts], "-l")) {
      flags &= ~PALLAS_READ_FLAG_UNROLL_LOOP;
    } else if (!strcmp(argv[nb_opts], "-i")) {
      nb_opts++;
      nb_intervals = atoi(argv[nb_opts]);
    } else if (!strcmp(argv[nb_opts], "-f")) {
      flamegraph = true;
    } else if (!strcmp(argv[nb_opts], "-c")) {
      csv = true;
    } else if (!strcmp(argv[nb_opts], "-cb")) {
      csv_bulk = true;
    } else if (!strcmp(argv[nb_opts], "--thread")) {
      per_thread = true;
      thread_to_print = atoi(argv[nb_opts+1]);
      nb_opts++;
    } else {
      /* Unknown parameter name. It's probably the trace's path name. We can stop
       * parsing the parameter list.
       */
      trace_name = argv[nb_opts];
    }
  }

  if (trace_name == nullptr) {
    std::cout << "Missing trace file" << std::endl;
    usage(argv[0]);
    return EXIT_SUCCESS;
  }

  auto trace = pallas_open_trace(trace_name);
  if(trace == nullptr)
    return EXIT_FAILURE;

  if(nb_intervals>0) {
    printIntervals(*trace);
  } else if (show_structure)
    printStructure(flags, *trace);
  else
    printTrace(*trace);

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
