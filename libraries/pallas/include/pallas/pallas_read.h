/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Everything needed to read a trace.
 */
#pragma once

#include "pallas.h"
#include "pallas_attribute.h"
#include "pallas_timestamp.h"

#ifdef __cplusplus
namespace pallas {
#endif
/** Maximum Callstack Size. */
#define MAX_CALLSTACK_DEPTH 100

/**
 * Options for the ThreadReader. Values have to be powers of 2.
 */
enum ThreadReaderOptions {
  None = 0,          /**< No specific options. */
  ShowStructure = 1, /**< Read and return the structure (Sequences and Loop). */
  NoTimestamps = 2,  /**< Do not load the timestamps / durations. */
};

/** Represents one occurence of an Event. */
typedef struct EventOccurence {
  struct Event* event;          /**< Pointer to the Event.*/
  pallas_timestamp_t timestamp; /**< Timestamp for that occurence.*/
  pallas_duration_t duration;   /**< Duration of that occurence.*/
  AttributeList* attributes;    /**< Attributes for that occurence.*/
} EventOccurence;

/**
 * Represents one occurence of a Sequence.
 */
typedef struct SequenceOccurence {
  struct Sequence* sequence;            /**< Pointer to the Sequence.*/
  struct Savestate* savestate;          /**< Savestate of the reader before entering the sequence.*/
  pallas_timestamp_t timestamp;         /**< Timestamp for that occurence.*/
  pallas_duration_t duration;           /**< Duration of that occurence.*/
  struct TokenOccurence* full_sequence; /** Array of the occurrences in this sequence. */
} SequenceOccurence;

/**
 * Represents one occurence of a Loop.
 */
typedef struct LoopOccurence {
  struct Loop* loop;                     /**< Pointer to the Loop.*/
  unsigned int nb_iterations;            /**< Number of iterations for that occurence.*/
  pallas_timestamp_t timestamp;          /**< Timestamp for that occurence.*/
  pallas_duration_t duration;            /**< Duration for that occurence.*/
  struct SequenceOccurence* full_loop;   /**< Array of the Sequences in this loop.*/
  struct SequenceOccurence loop_summary; /**< False SequenceOccurence that represents a summary of all the
                                          * occurrences in full_loop. */
} LoopOccurence;

/**
 * Represents any kind of Occurrence.
 */
typedef union Occurence {
  struct LoopOccurence loop_occurence;         /**< Occurence for a Loop.*/
  struct SequenceOccurence sequence_occurence; /**< Occurence for a Sequence.*/
  struct EventOccurence event_occurence;       /**< Occurence for an Event.*/
} Occurence;

/**
 * Tuple containing a Token and its corresponding Occurence.
 */
typedef struct TokenOccurence {
  /** Token for the occurence. */
  const Token* token;
  /** Occurence corresponding to the Token. */
  Occurence* occurence;

#ifdef __cplusplus
  ~TokenOccurence();
#endif
} TokenOccurence;

/**
 * Reads one thread from an Pallas trace.
 */
typedef struct ThreadReader {
  /** Archive being read by this reader. */
  const struct Archive* archive;
  /** Thread being read. */
  struct Thread* thread_trace;
  /** The current referential timestamp. */
  pallas_timestamp_t referential_timestamp;

  /** Stack containing the sequences/loops being read. */
  Token callstack_iterable[MAX_CALLSTACK_DEPTH];

  /** Stack containing the index in the sequence or the loop iteration. */
  int callstack_index[MAX_CALLSTACK_DEPTH];

  /** Current frame = index of the event/loop being read in the callstacks.
   * You can view this as the "depth" of the callstack. */
  int current_frame;

  /** At any point, a token t has been seen tokenCount[t] times. */
  DEFINE_TokenCountMap(tokenCount);

  /**
   * Options as defined in pallas::ThreadReaderOptions.
   */
  int options;
#ifdef __cplusplus
  /**
   * Make a new ThreadReader from an Archive and a threadId.
   * @param archive Archive to read.
   * @param threadId Id of the thread to read.
   * @param options Options as defined in ThreadReaderOptions.
   */
  ThreadReader(const Archive* archive, ThreadId threadId, int options);

 private:
  /** Returns the Sequence being run at the given frame. */
  [[nodiscard]] const Token& getFrameInCallstack(int frame_number) const;
  /** Returns the token being run at the given frame. */
  [[nodiscard]] const Token& getTokenInCallstack(int frame_number) const;
  /** Prints the current Token. */
  void printCurToken() const;
  /** Prints the current Sequence. */
  void printCurSequence() const;
  /** Prints the whole current callstack. */
  void printCallstack() const;
  /** Returns the EventSummary of the given Event. */
  [[nodiscard]] EventSummary* getEventSummary(Token event) const;
  /** Returns the timestamp of the given event occurring at the given index. */
  [[nodiscard]] pallas_timestamp_t getEventTimestamp(Token event, int occurence_id) const;
  /** Returns whether the given sequence still has more Tokens after the given current_index. */
  [[nodiscard]] bool isEndOfSequence(int current_index, Token sequence_id) const;
  /** Returns whether the given loop still has more Tokens after the given current_index. */
  [[nodiscard]] bool isEndOfLoop(int current_index, Token loop_id) const;
  /** Returns the duration of the given Loop. */
  [[nodiscard]] pallas_duration_t getLoopDuration(Token loop_id) const;

  /** Returns an EventOccurence for the given Token appearing at the given occurence_id.
   * Timestamp is set to Reader's referential timestamp.*/
  [[nodiscard]] EventOccurence getEventOccurence(Token event_id, size_t occurence_id) const;
  /** Returns an SequenceOccurence for the given Token appearing at the given occurence_id.
   * Timestamp is set to Reader's referential timestamp.*/
  [[nodiscard]] SequenceOccurence getSequenceOccurence(Token sequence_id,
                                                       size_t occurence_id,
                                                       bool saveReaderState) const;
  /** Returns an LoopOccurence for the given Token appearing at the given occurence_id.
   * Timestamp is set to Reader's referential timestamp.*/
  [[nodiscard]] LoopOccurence getLoopOccurence(Token loop_id, size_t occurence_id) const;

  /** Returns a pointer to the AttributeList for the given occurence of the given Event. */
  [[nodiscard]] AttributeList* getEventAttributeList(Token event_id, size_t occurence_id) const;

  /** Skips the given Token and updates the reader. */
  static void skipToken([[maybe_unused]] Token token) { pallas_error("Not implemented yet\n"); };

 public:
  /** Returns the current Sequence*/
  [[nodiscard]] const Token& getCurIterable() const;
  /** Enter a block (push a new frame in the callstack) */
  void enterBlock(Token new_block);
  /** Leaves the current block */
  void leaveBlock();
  /** Checks if there is a Loop in the current callstack. */
  bool isInLoop();
  /** Checks if c,urrent Token is the last of the current Sequence/Loop */
  bool isLastInCurrentArray();
  /** Moves the reader's position to the next token.
   *
   * Moves to the next token in the current block, or exits it recursively as long as it's at the end of a block.*/
  void moveToNextToken();
  /** Updates the reader's callstacks et al. by reading the current token.
   *
   * If the current Token is an event, its index is updated and the referential timestamp as well.
   * If it's a Loop or a Sequence, their indexes are updated, and the reader enters the corresponding block. */
  void updateReadCurToken();
  /** Fetches the next Token in the trace. Changes the state of the reader to match. */
  [[nodiscard]] Token getNextToken();
  /** Returns the current Token. */
  [[nodiscard]] const Token& getCurToken() const;
  /** Returns an Occurence for the given Token appearing at the given occurence_id.
   *
   * Timestamp is set to Reader's referential timestamp.*/
  [[nodiscard]] union Occurence* getOccurence(Token id, size_t occurence_id) const;
  /** Loads the given savestate. */
  void loadSavestate(struct Savestate* savestate);
  /** Reads the current level of the thread, and returns it as an array of TokenOccurences. */
  [[nodiscard]] std::vector<TokenOccurence> readCurrentLevel();
  /** Skips the given Sequence and updates the reader. */
  void skipSequence([[maybe_unused]] Token token) { pallas_error("Not implemented yet\n"); };
  ~ThreadReader();
#endif
} ThreadReader;

/**
 * A savestate of a pallas_thread_reader.
 */
typedef struct Savestate {
  /** The current referential timestamp. */
  pallas_timestamp_t referential_timestamp;

  /** Stack containing the sequences/loops being read. */
  Token* callstack_iterable;

  /** Stack containing the index in the sequence or the loop iteration. */
  int* callstack_index;

  /** Current frame = index of the event/loop being read in the callstacks.
   * You can view this as the "depth" of the callstack. */
  int current_frame;

  DEFINE_TokenCountMap(tokenCount);
#ifdef __cplusplus
  /** Creates a savestate of the given reader.
   * @param reader Reader whose state of reading we want to take a screenshot. */
  Savestate(const ThreadReader* reader);
  ~Savestate();
#endif
} Savestate;

#ifdef __cplusplus
}; /* namespace pallas */

extern "C" {
#endif

/** Creates and initializes a ThreadReader. */
extern PALLAS(ThreadReader) * pallas_new_thread_reader(const PALLAS(Archive) * archive, PALLAS(ThreadId) thread_id, int options);
/** Enter a block (push a new frame in the callstack) */
extern void pallas_thread_reader_enter_block(PALLAS(ThreadReader) * reader, PALLAS(Token) new_block);
/** Leaves the current block */
extern void pallas_thread_reader_leave_block(PALLAS(ThreadReader) * reader);

/** Moves the reader's position to the next token.
 *
 * Moves to the next token in the current block, or exits it recursively as long as it's at the end of a block.*/
extern void pallas_thread_reader_move_to_next_token(PALLAS(ThreadReader) * reader);

/** Updates the reader's callstacks et al. by reading the current token.
 *
 * If the current Token is an event, its index is updated and the referential timestamp as well.
 * If it's a Loop or a Sequence, their indexes are updated, and the reader enters the corresponding block. */
extern void pallas_thread_reader_update_reader_cur_token(PALLAS(ThreadReader) * reader);

/** Fetches the next Token in the trace. Changes the state of the reader to match. */
extern PALLAS(Token) pallas_thread_reader_get_next_token(PALLAS(ThreadReader) * reader);

/** Returns the current Token. */
extern PALLAS(Token) pallas_read_thread_cur_token(const PALLAS(ThreadReader) * reader);

/** Returns an Occurence for the given Token appearing at the given occurence_id.
 *
 * Timestamp is set to Reader's referential timestamp.*/
extern union PALLAS(Occurence) *
  pallas_thread_reader_get_occurence(const PALLAS(ThreadReader) * reader, PALLAS(Token) id, int occurence_id);

/** Creates a new savestate from the reader. */
extern PALLAS(Savestate) * create_savestate(const PALLAS(ThreadReader) * reader);

/** Loads the given savestate. */
extern void load_savestate(PALLAS(ThreadReader) * reader, PALLAS(Savestate) * savestate);

/** Reads the current level of the thread, and returns it as an array of TokenOccurences. */
extern PALLAS(TokenOccurence) * pallas_thread_reader_read_current_level(PALLAS(ThreadReader) * reader);

/** Skips the given Sequence and updates the reader. */
extern void skip_sequence(PALLAS(ThreadReader) * reader, PALLAS(Token) token);

#ifdef __cplusplus
};
#endif

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
