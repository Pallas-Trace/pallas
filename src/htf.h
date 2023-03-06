#ifndef HTF_H
#define HTF_H

#include "htf_timestamp.h"
#include "htf_dbg.h"

/* A token is either:
   - an event
   - a sequence (ie a list of tokens)
   - a loop (a repetition of sequences)
*/

typedef int event_id;
typedef int sequence_id;
typedef int loop_id;

#define EVENT_ID_INVALID ((unsigned)-1)

/* Token types */
#define TYPE_INVALID  0
#define TYPE_EVENT    1
#define TYPE_SEQUENCE 2
#define TYPE_LOOP     3

/* identify a token */
typedef uint32_t token_t;
#define TOKEN_TYPE(t) ( (t) >> 30)
#define TOKEN_ID(t) ( (t) & (~(3<<30)))
#define TOKENIZE(type, id) ((type)<<30 | TOKEN_ID(id))

#define MAIN_SEQUENCE TOKENIZE(TYPE_SEQUENCE, EVENT_ID_INVALID)

/*************************** Events **********************/

enum event_type {
   function_entry,
   function_exit,
   singleton,
};

struct event {
  enum event_type event_type;
  /* TODO: update the content of the event */

  int function_id;
  //uintptr_t ptr;
  //  pthread_t tid;
  //  enum intercepted_function function;
};

/*************************** Sequence **********************/

struct sequence {
  unsigned length;
  token_t *token;		/* TODO: don't use a pointer here! */
};

struct ongoing_sequence {
  struct sequence seq;
  int nb_allocated_tokens;
  struct ongoing_sequence* enclosing_seq;  
};

/*************************** Loop **********************/

struct loop {
  unsigned nb_iterations;
  token_t token;
};


struct event_occurence {
  struct event event;
  timestamp_t timestamp;
};

struct event_summary {
  struct event event;
  timestamp_t *timestamps;
  unsigned nb_allocated_timestamps;
  unsigned nb_timestamps;
};

struct thread_trace {
  struct trace *trace;
  token_t *tokens;
  unsigned nb_allocated_tokens;
  unsigned nb_tokens;

  struct ongoing_sequence *ongoing_sequence;
  
  struct event_summary *events;
  unsigned nb_allocated_events;
  unsigned nb_events;

  struct sequence *sequences;
  unsigned nb_allocated_sequences;
  unsigned nb_sequences;

  struct loop *loops;
  unsigned nb_allocated_loops;
  unsigned nb_loops;
};

struct trace {
  struct thread_trace **threads;
  _Atomic int nb_threads;
  pthread_mutex_t lock;
};

struct thread_writer {
  struct thread_trace thread_trace;
  struct ongoing_sequence **og_seq;
  int cur_depth;
  int max_depth;
  int thread_rank;
};

struct thread_reader {
  struct trace *trace;
  struct thread_trace *thread_trace;

  token_t *callstack_sequence;	/* each entry contains the sequence/loop being read */
  int     *callstack_index;	/* each entry contains the index in the sequence or the loop iteration */
  int     callstack_depth;

  int *event_index;
};

/* Initialize a trace in write mode */
void htf_write_init(struct trace *trace, const char* dirname);
void htf_write_init_thread(struct trace* trace,
			   struct thread_writer *thread_writer,
			   int thread_rank);

void htf_record_event(struct thread_writer* thread_writer,
		      enum event_type event_type,
		      int function_id);
void htf_write_finalize(struct trace* trace);

void htf_storage_init();

void htf_storage_finalize(struct trace*trace);




void htf_read_trace(struct trace* trace, char* filename);

void htf_read_thread_iterator_init(struct thread_reader *reader,
				   struct trace* trace,
				   int thread_index);

/* return the current event in a thread and move to the next one.
 * Return -1 in case of an error (such as the end of the trace)
 */
int htf_read_thread_next_event(struct thread_reader *reader,
			       struct event_occurence *e);
/* return the current event in a thread.
 * Return -1 in case of an error (such as the end of the trace)
 */
int htf_read_thread_cur_event(struct thread_reader *reader,
			      struct event_occurence *e);

#endif /* EVENT_H */
