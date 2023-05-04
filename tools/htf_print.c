#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "htf.h"
#include "htf_archive.h"
#include "htf_read.h"

static int show_structure = 0;
static int per_thread = 0;
static long max_depth = MAX_CALLSTACK_DEPTH;

/* Print one event */
static void print_event(struct htf_thread* thread, htf_token_t token, struct htf_event_occurence* e) {
	printf("%.9lf\t\t", e->timestamp / 1e9);
	if (!per_thread)
		printf("%s\t", htf_get_thread_name(thread));
	htf_print_token(thread, token);
	printf("\t");
	htf_print_event(thread, &e->event);
	printf("\n");
}

static void print_sequence(struct htf_thread* thread, htf_token_t token, htf_timestamp_t ts) {
	if (ts != 0)
		printf("%.9lf\t\t", ts / 1e9);
	else
		printf("Sequence     \t\t");
	if (!per_thread)
		printf("%s\t", htf_get_thread_name(thread));
	htf_print_token(thread, token);
	printf("\t");
	struct htf_sequence* seq = htf_get_sequence(thread, HTF_SEQUENCE_ID(token.id));
	for (unsigned i = 0; i < seq->size; i++) {
		htf_print_token(thread, seq->token[i]);
		printf(" ");
	}
	printf("\n");
}

static void print_loop(struct htf_thread* thread, htf_token_t token, htf_timestamp_t ts) {
	if (ts != 0)
		printf("%.9lf\t\t", ts / 1e9);
	else
		printf("Loop         \t\t");
	if (!per_thread)
		printf("%s\t", htf_get_thread_name(thread));
	struct htf_loop* loop = htf_get_loop(thread, HTF_LOOP_ID(token.id));
	htf_print_token(thread, token);
	printf("\t%d * ", loop->nb_iterations);
	htf_print_token(thread, loop->token);
	printf("\n");
}

/* Print all the events of a thread */
static void print_thread(struct htf_archive* trace, struct htf_thread* thread) {
	printf("Reading events for thread %u (%s):\n", thread->id, htf_get_thread_name(thread));
	printf("Timestamp\t\tTag\tEvent\n");

	struct htf_thread_reader reader;
	htf_read_thread_iterator_init(trace, &reader, thread->id);

	struct htf_event_occurence e;
	struct htf_token t, copy_token;
	while (htf_read_thread_next_token(&reader, &t, &e) == 0) {
		// We need to copy the first token we encountered
		// Because we'll always have to print it, but t will be modified during the while loop
		memcpy(&copy_token, &t, sizeof(copy_token));
		htf_log(htf_dbg_lvl_verbose, "Reading token(%x.%x)\n", copy_token.type, copy_token.id);

		// This insures we don't go deeper than necessary
		while (reader.current_frame > max_depth) {
			htf_read_thread_next_token(&reader, &t, &e);
		}
		// Prints the structure of the sequences and the loops
		if (show_structure) {
			for (int i = 0; i < reader.current_frame - 1; i++)
				printf("│ ");
			if (reader.depth == reader.current_frame) {
				if (copy_token.type == HTF_TYPE_EVENT)
					printf("│ ");
				else
					printf("├─");
			}
			if (reader.depth > reader.current_frame) {
				// Means we ended some blocks
				if (reader.current_frame > 0)
					printf("│ ");
				for (int i = (reader.current_frame >= 0) ? reader.current_frame : 0; i < reader.depth; i++)
					printf("╰─");
			}
		}
		// Prints the token we first started with
		switch (copy_token.type) {
			case HTF_TYPE_INVALID:
				htf_error("Type is invalid\n");
				break;
			case HTF_TYPE_EVENT:
				print_event(thread, copy_token, &e);
				break;
			case HTF_TYPE_SEQUENCE:
				if (reader.depth == max_depth)
					print_sequence(thread, copy_token, htf_get_starting_timestamp(&reader, copy_token));
				else if (show_structure)
					print_sequence(thread, copy_token, 0);
				break;
			case HTF_TYPE_LOOP:
				if (reader.depth == max_depth)
					print_loop(thread, copy_token, htf_get_starting_timestamp(&reader, copy_token));
				else if (show_structure)
					print_loop(thread, copy_token, 0);
				break;
		}
		reader.depth = reader.current_frame;
	}
}

/* compare the timestamps of the current event on each thread and select the smallest timestamp
 * This fills the event_occurence e and returns the index of the selected thread (or -1 at the end
 * of the trace)
 */
static int get_next_event(struct htf_thread_reader *readers,
			  int nb_threads,
			  struct htf_event_occurence *e) {

  struct htf_event_occurence cur_e;
  htf_timestamp_t min_ts = HTF_TIMESTAMP_INVALID;
  int min_index = -1;

  for(int i=0; i< nb_threads; i++) {
    if(htf_read_thread_cur_event(&readers[i], &cur_e) == 0) {
      if( min_ts == HTF_TIMESTAMP_INVALID || min_ts > cur_e.timestamp) {
	min_index = i;
	min_ts = cur_e.timestamp;
      }
    }
  }

  if(min_index>=0) {
    htf_read_thread_next_event(&readers[min_index], e);
  }

  return  min_index;
}

/* Print all the events of all the threads sorted by timestamp */
void print_trace(struct htf_archive *trace) {
	int nb_threads = trace->nb_threads;
	struct htf_thread_reader* readers = malloc(sizeof(struct htf_thread_reader) * (nb_threads));
	for (int i = 0; i < trace->nb_threads; i++) {
		htf_read_thread_iterator_init(trace, &readers[i], trace->threads[i]->id);
	}

	printf("Timestamp\t\tThread Name\tTag\tEvent\n");

	struct htf_event_occurence e;
	int thread_index = -1;
	while ((thread_index = get_next_event(readers, nb_threads, &e)) >= 0) {
		print_event(readers[thread_index].thread_trace, HTF_TOKENIZE(1, 0), &e);
	}
}

void usage(const char *prog_name) {
	printf("Usage: %s [OPTION] trace_file\n", prog_name);
	printf("\t-T          Print events per thread\n");
	printf("\t-S          Structure mode\n");
	printf("\t--max-depth Next args is the max-depth which shall be printed\n");
	printf("\t-v          Verbose mode\n");
	printf("\t-?  -h      Display this help and exit\n");
}

int main(int argc, char**argv) {
	int nb_opts = 0;
	char* trace_name = NULL;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-v")) {
      htf_debug_level_set(htf_dbg_lvl_debug);
      nb_opts++;
    } else if (!strcmp(argv[i], "-T")) {
			per_thread = 1;
			nb_opts++;
		} else if (!strcmp(argv[i], "-S")) {
			show_structure = 1;
			nb_opts++;
		} else if (!strcmp(argv[i], "--max-depth")) {
			max_depth = strtol(argv[i + 1], NULL, 10);
			nb_opts += 2;
			i++;
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

  struct htf_archive trace;
  htf_read_archive(&trace, trace_name);

  if(per_thread) {
    for(int i=0; i< trace.nb_threads; i++) {
			printf("\n");
			print_thread(&trace, trace.threads[i]);
		}
	} else {
    print_trace(&trace);
  }

  return EXIT_SUCCESS;
}
