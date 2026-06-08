/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Everything needed to read a trace.
 */
#pragma once
#ifndef __cplusplus
#include <stdbool.h>
#include <stddef.h>
#endif
#include "pallas.h"
#include "pallas_archive.h"
#include "pallas_attribute.h"
#include "utils/pallas_timestamp.h"

#ifdef __cplusplus
#include <vector>
#include <cstddef>
namespace pallas {
#endif

/** getNextToken flags */
#define PALLAS_READ_FLAG_NONE            0
#define PALLAS_READ_FLAG_NO_UNROLL       (1 << 0)
#define PALLAS_READ_FLAG_UNROLL_SEQUENCE (1 << 2)
#define PALLAS_READ_FLAG_UNROLL_LOOP     (1 << 3)
#define PALLAS_READ_FLAG_UNROLL_ALL (PALLAS_READ_FLAG_UNROLL_SEQUENCE|PALLAS_READ_FLAG_UNROLL_LOOP)


/** Represents one occurrence of an Event. */
typedef struct EventOccurrence {
  struct EventData* event;          /**< Pointer to the Event.*/
  pallas_timestamp_t timestamp; /**< Timestamp for that occurrence.*/
  AttributeList* attributes;    /**< Attributes for that occurrence.*/
} EventOccurrence;

/**
 * Represents one occurrence of a Sequence.
 */
typedef struct SequenceOccurrence {
  struct Sequence* sequence;            /**< Pointer to the Sequence.*/
  pallas_timestamp_t timestamp;         /**< Timestamp for that occurrence.*/
  pallas_duration_t duration;           /**< Duration of that occurrence.*/
  struct TokenOccurrence* full_sequence; /** Array of the occurrences in this sequence. */
  struct Cursor *checkpoint;
} SequenceOccurrence;

/**
 * Represents one occurrence of a Loop.
 */
typedef struct LoopOccurrence {
  struct Loop* loop;                     /**< Pointer to the Loop.*/
  unsigned int nb_iterations;            /**< Number of iterations for that occurrence.*/
  pallas_timestamp_t timestamp;          /**< Timestamp for that occurrence.*/
  pallas_duration_t duration;            /**< Duration for that occurrence.*/
  struct SequenceOccurrence* full_loop;   /**< Array of the Sequences in this loop.*/
  struct SequenceOccurrence loop_summary; /**< False SequenceOccurrence that represents a summary of all the
                                          * occurrences in full_loop. */
} LoopOccurrence;

/**
 * Represents any kind of Occurrence.
 */
typedef union Occurrence {
  struct LoopOccurrence loop_occurrence;         /**< Occurrence for a Loop.*/
  struct SequenceOccurrence sequence_occurrence; /**< Occurrence for a Sequence.*/
  struct EventOccurrence event_occurrence;       /**< Occurrence for an Event.*/
} Occurrence;

/**
 * Tuple containing a Token and its corresponding Occurrence.
 */
typedef struct TokenOccurrence {
  /** Token for the occurrence. */
  const Token* token;
  /** Occurrence corresponding to the Token. */
  Occurrence* occurrence;

#ifdef __cplusplus
  ~TokenOccurrence();
#endif
} TokenOccurrence;

/** Represents a frame in the callstack of the trace. */
typedef struct CallstackFrame {
  /** The current timestamp. */
  pallas_timestamp_t current_timestamp;

  /** Current iterable in this frame. */
  Token callstack_iterable;

  /** Stack containing the index in the sequence or the loop iteration. */
  int frame_index;


  DEFINE_TokenCountMap(tokenCount);
#ifdef __cplusplus
  /** Creates an empty CallstackFrame. */
  CallstackFrame();
  ~CallstackFrame();
#endif
} CallstackFrame;

/** A Cursor represents a state of the trace being read. It stores information about the callstacks, mostly. */
typedef struct Cursor {
  /** Index of currentFrame in callstack. */
  int current_frame_index;

    /** Pointer to the current CallstackFrame in callstack. */
  CallstackFrame *currentFrame;

    /** Callstack. */
  size_t callstack_capacity;
  CallstackFrame *callstack;
#ifdef __cplusplus
  explicit Cursor(const Cursor& other);
  Cursor& operator=(const Cursor& other);
  Cursor();
  ~Cursor();
#endif
} Cursor;

/**
 * Reads one thread from a Pallas trace. Owns the memory for the thread.
 */
typedef struct ThreadReader {
    /** Archive being read by this reader. */
    struct Archive *archive;
    /** Thread being read. */
    struct Thread *thread_trace;

    /** Current state, as represented by a Cursor. */
    Cursor currentState;

    /**
     * Options as defined in pallas::ThreadReaderOptions.
     */
    int pallas_read_flag;
#ifdef __cplusplus
    /**
     * Make a new ThreadReader from an Archive and a threadId.
     * @param archive Archive to read.
     * @param threadId ID of the thread to read.
     * @param pallas_read_flag Default flag when reading
     */
    ThreadReader(Archive *archive, ThreadId threadId, int pallas_read_flag);

    /**
     * This is just for convenience and should not be used as is.
     * Using an empty ThreadReader can and **will** segfault
     */
    ThreadReader() = default;

    /** Returns the Sequence being run at the given frame. */
    [[nodiscard]] const Token &getFrameInCallstack(int frame_number) const;

    /** Returns the token being run at the given frame. */
    [[nodiscard]] const Token &getTokenInCallstack(int frame_number) const;

    /** Prints the current Token. */
    void printCurToken() const;

    /** Gets the current Iterable. */
    [[nodiscard]] const Token &getCurIterable() const;

    /** Prints the current Sequence. */
    void printCurSequence() const;

    /** Prints the whole current callstack. */
    void printCallstack() const;

    /** Returns the Event of the given Event. */
    [[nodiscard]] Event *getEvent(Token event) const;

    /** Returns the timestamp of the given event occurring at the given index. */
    [[nodiscard]] pallas_timestamp_t getEventTimestamp(Token event, int occurrence_id) const;

    /** Returns whether the given sequence still has more Tokens after the given current_index. */
    [[nodiscard]] bool isEndOfSequence(int current_index, Token sequence_id) const;

    /** Returns whether the given loop still has more Tokens after the given current_index. */
    [[nodiscard]] bool isEndOfLoop(int current_index, Token loop_id) const;

    /** Returns whether the given iterable token still has more Tokens after the given current_index. */
    [[nodiscard]] bool isEndOfBlock(int index, Token iterable_token) const;

    /** Returns whether the cursor is at the end of the current block. */
    [[nodiscard]] bool isEndOfCurrentBlock() const;

    /** Returns whether the cursor is at the end of the trace. */
    [[nodiscard]] bool isEndOfTrace() const;

    /** Returns the duration of the given Loop. */
    [[nodiscard]] pallas_duration_t getLoopDuration(Token loop_id) const;

    /** Returns an EventOccurrence for the given Token appearing at the given occurrence_id.
     * Timestamp is set to Reader's referential timestamp.*/
    [[nodiscard]] EventOccurrence getEventOccurrence(Token event_id, size_t occurrence_id) const;

    /** Returns an SequenceOccurrence for the given Token appearing at the given occurrence_id.
     * Timestamp is set to Reader's referential timestamp.*/
    [[nodiscard]] SequenceOccurrence getSequenceOccurrence(Token sequence_id,
                                                         size_t occurrence_id) const;

    /** Returns an LoopOccurrence for the given Token appearing at the given occurrence_id.
     * Timestamp is set to Reader's referential timestamp.*/
    [[nodiscard]] LoopOccurrence getLoopOccurrence(Token loop_id, size_t occurrence_id) const;
    /** Returns the current token count for given token.*/
    [[nodiscard]] size_t getCurrentTokenCount(Token t) const;

    /** Returns the current timestamp. */
    [[nodiscard]] pallas_timestamp_t getCurrentTimestamp() const;

    /** Returns a pointer to the AttributeList for the given occurrence of the given Event. */
    [[nodiscard]] AttributeList *getEventAttributeList(Token event_id, size_t occurrence_id) const;

    /** Returns a map that assigns names to sequences */
    void guessSequencesNames(std::map<pallas::Sequence *, std::string> &names) const;

    //******************* EXPLORATION FUNCTIONS ********************

    /** Gets the current Token. */
    [[nodiscard]] const Token &pollCurToken() const;

    /** Peeks at and return the next token without actually updating the state */
    [[nodiscard]] Token pollNextToken(int flags = PALLAS_READ_FLAG_NONE) const;

    /** Updates the internal state, returns true if internal state was actually changed */
    bool moveToNextToken(int flags = PALLAS_READ_FLAG_NONE);

    /** Equivalent to moveToNextToken(PALLAS_READ_FLAG_NO_UNROLL) */
    bool moveToNextTokenInBlock();

    /** Gets the next token and updates the reader's state if it returns a value.
     * It is exactly equivalent to `moveToNextToken()` then `pollCurToken()` */
    Token getNextToken(int flags = PALLAS_READ_FLAG_NONE);

    /** Peeks at and return the previous token without actually updating the state */
    [[nodiscard]] Token pollPrevToken(int flags = PALLAS_READ_FLAG_NONE) const;

    /** Updates the internal state, returns true if internal state was actually changed */
    bool moveToPrevToken(int flags = PALLAS_READ_FLAG_NONE);

    /** Equivalent to moveToPrevToken(PALLAS_READ_FLAG_NO_UNROLL) */
    bool moveToPrevTokenInBlock();

    /** Gets the previous token and updates the reader's state if it returns a value.
     * It is exactly equivalent to `moveToPrevToken()` then `pollCurToken()` */
    Token getPrevToken(int flags = PALLAS_READ_FLAG_NONE);

    /** Enters a block */
    void enterBlock();

    /** Leaves the current block */
    void leaveBlock();

    /** Exits a block if at the end of it and flags allow it, returns a boolean representing if the rader actually exited a block */
    bool exitIfEndOfBlock(int flags = PALLAS_READ_FLAG_UNROLL_ALL);

    /** Enter a block if the current token starts a block, returns a boolean representing if the rader actually entered a block */
    bool enterIfStartOfBlock(int flags = PALLAS_READ_FLAG_UNROLL_ALL);

    Cursor createCheckpoint() const;

    void loadCheckpoint(Cursor *checkpoint);

    /** Frees the memory of the ThreadReader. Also clears up the memory of the thread from the archive. */
    ~ThreadReader();

    ThreadReader(const ThreadReader &);

    ThreadReader(ThreadReader &&other) noexcept;

    ThreadReader &operator=(const ThreadReader &);

    ThreadReader &operator=(ThreadReader &&other) noexcept;
#endif
} ThreadReader;

/** Similar to the ThreadReader but iterates over many threads at the same time. */
typedef struct MultiThreadReader {
    /** Number of threads being read.*/
    size_t n_threads;
    /** Array of ThreadReader. */
    ThreadReader *readers;
    /** Current ThreadReader, ie whose ThreadReader::current_timestamp is the lowest. */
    ThreadReader *current_reader;
    #ifdef __cplusplus
    /** Create a MultiThreadReader from a vector of Threads.*/
    MultiThreadReader(std::vector<Thread *> threads);

    /** Used to get a multi-thread reader of every thread in a trace */
    MultiThreadReader(GlobalArchive &trace);
    ~MultiThreadReader();

    /** Gets the current Token. */
    [[nodiscard]] Token pollCurToken() const;

    /** Updates the internal state to update current_reader to the earlier one. */

    bool updateMinReader();

    /** Updates the internal state, returns true if internal state was actually changed */
    bool moveToNextToken();

    /** Gets the next token and updates the reader's state if it returns a value.
     * It is exactly equivalent to `moveToNextToken()` then `pollCurToken()` */
    Token getNextToken();
    #endif
} MultiThreadReader;

/* C bindings */

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Make a new ThreadReader from an Archive and a threadId.
 * @param archive Archive to read.
 * @param threadId Id of the thread to read.
 * @param options Options as defined in ThreadReaderOptions.
 */
extern ThreadReader pallasCreateThreadReader(Archive* archive, ThreadId threadId, int options);
/** Prints the current Token. */
extern void pallasPrintCurToken(ThreadReader *thread_reader);
/** Gets the current Iterable. */
extern Token pallasGetCurIterable(ThreadReader *thread_reader);
/** Prints the current Sequence. */
extern void pallasPrintCurSequence(ThreadReader *thread_reader);
/** Prints the whole current callstack. */
extern void pallasPrintCallstack(ThreadReader *thread_reader);
/** Returns the Event from the given token. */
extern Event* pallasGetEvent(ThreadReader *thread_reader, Token event);
/** Returns the timestamp of the given event occurring at the given index. */
extern pallas_timestamp_t pallasGetEventTimestamp(ThreadReader *thread_reader, Token event, int occurrence_id);
/** Returns whether the given sequence still has more Tokens after the given current_index. */
extern bool pallasIsEndOfSequence(ThreadReader *thread_reader, int current_index, Token sequence_id);
/** Returns whether the given loop still has more Tokens after the given current_index. */
extern bool pallasIsEndOfLoop(ThreadReader *thread_reader, int current_index, Token loop_id);
/** Returns whether the given iterable token still has more Tokens after the given current_index. */
extern bool pallasIsEndOfBlock(ThreadReader *thread_reader, int index, Token iterable_token);
/** Returns whether the cursor is at the end of the current block. */
extern bool pallasIsEndOfCurrentBlock(ThreadReader *thread_reader);
/** Returns whether the cursor is at the end of the trace. */
extern bool pallasIsEndOfTrace(ThreadReader *thread_reader);
/** Returns the duration of the given Loop. */
extern pallas_duration_t pallasGetLoopDuration(ThreadReader *thread_reader, Token loop_id);

/** Returns an EventOccurrence for the given Token appearing at the given occurrence_id.
 * Timestamp is set to Reader's referential timestamp.*/
extern EventOccurrence pallasGetEventOccurrence(ThreadReader *thread_reader, Token event_id, size_t occurrence_id);
/** Returns an SequenceOccurrence for the given Token appearing at the given occurrence_id.
 * Timestamp is set to Reader's referential timestamp.*/
extern SequenceOccurrence pallasGetSequenceOccurrence(ThreadReader *thread_reader,
                                             Token sequence_id,
                                             size_t occurrence_id,
                                             bool create_checkpoint);
/** Returns an LoopOccurrence for the given Token appearing at the given occurrence_id.
 * Timestamp is set to Reader's referential timestamp.*/
extern LoopOccurrence pallasGetLoopOccurrence(ThreadReader *thread_reader, Token loop_id, size_t occurrence_id);

/** Returns a pointer to the AttributeList for the given occurrence of the given Event. */
extern AttributeList* pallasGetEventAttributeList(ThreadReader *thread_reader, Token event_id, size_t occurrence_id);

//******************* EXPLORATION FUNCTIONS ********************

/** Gets the current Token. */
extern Token pallasPollCurToken(ThreadReader *thread_reader);
/** Peeks at and return the next token without actually updating the state */
extern Token pallasPollNextToken(ThreadReader *thread_reader, int flags);
/** Peeks at and return the previous token without actually updating the state */
extern Token pallasPollPrevToken(ThreadReader *thread_reader, int flags);
/** Updates the internal state, returns true if internal state was actually changed */
extern bool pallasMoveToNextToken(ThreadReader *thread_reader, int flags);
/** Equivalent to pallasMoveToNextToken(PALLAS_READ_FLAG_NO_UNROLL) */
extern bool pallasMoveToNextTokenInBlock(ThreadReader *thread_reader);
/** Updates the internal state, returns true if internal state was actually changed */
extern bool pallasMoveToPrevToken(ThreadReader *thread_reader, int flags);
/** Equivalent to pallasMoveToPrevToken(PALLAS_READ_FLAG_NO_UNROLL) */
extern bool pallasMoveToPrevTokenInBlock(ThreadReader *thread_reader);
/** Gets the next token and updates the reader's state if it returns a value.
 * It is more or less equivalent to `moveToNextToken()` then `pollCurToken()` */
extern Token pallasGetNextToken(ThreadReader *thread_reader, int flags);
/** Gets the previous token and updates the reader's state if it returns a value.
 * It is exactly equivalent to `moveToPrevToken()` then `pollCurToken()` */
extern Token pallasGetPrevToken(ThreadReader *thread_reader, int flags);
/** Enters a block */
extern void pallasEnterBlock(ThreadReader *thread_reader);
/** Leaves the current block */
extern void pallasLeaveBlock(ThreadReader *thread_reader);
/** Exits a block if at the end of it and flags allow it, returns a boolean representing if the reader actually exited a block */
extern bool pallasExitIfEndOfBlock(ThreadReader *thread_reader, int flags);
/** Enter a block if the current token starts a block, returns a boolean representing if the rader actually entered a block */
extern bool pallasEnterIfStartOfBlock(ThreadReader *thread_reader, int flags);
/** Creates a copy of the given ThreadReader to be used as a "checkpoint" and be reloaded later */
extern Cursor pallasCreateCheckpoint(ThreadReader *thread_reader);
/** Loads a checkpoint `ThreadReader` into another one */
extern void pallasLoadCheckpoint(ThreadReader *thread_reader, Cursor *checkpoint);

#ifdef __cplusplus
} /* extern C */
}; /* namespace pallas */
#endif

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
