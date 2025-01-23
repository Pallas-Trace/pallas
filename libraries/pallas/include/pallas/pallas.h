/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * The main file of Pallas. Here are defined the most basic elements of the trace
 * (Tokens, Events, Sequences and Loops), as well as Threads, representing an execution stream.
 */
#pragma once

#include <pthread.h>
#include "pallas_config.h"
#include "pallas_dbg.h"
#include "pallas_linked_vector.h"
#include "pallas_timestamp.h"

#ifdef __cplusplus
#include <cstring>
#include <map>
#include <unordered_map>
#else
#include <stdbool.h>
#include <string.h>
#endif

/**
 * A simple alias to make some code clearer. We use uint8 because they're the size of a byte.
 */
typedef uint8_t byte;

#ifdef __cplusplus
namespace pallas {
#endif

/*************************** Tokens **********************/

/**
 * A trace is composed of basic units called tokens.
 * A token is either:
 *   - an event
 *   - a sequence (ie a list of tokens)
 *   - a loop (a repetition of sequences)
 */

/**
 * Enumeration of token types
 */
enum TokenType { TypeInvalid = 0, TypeEvent = 1, TypeSequence = 2, TypeLoop = 3 };

/**
 * Match numerical token type with a character
 * TypeInvalid = 'I'
 * TypeEvent = 'E'
 * TypeSequence = 'S'
 * TypeLoop = 'L'
 * 'U' otherwise
 */
#define PALLAS_TOKEN_TYPE_C(t)       \
  ((t).type) == TypeInvalid    ? 'I' \
  : ((t).type) == TypeEvent    ? 'E' \
  : ((t).type) == TypeSequence ? 'S' \
  : ((t).type) == TypeLoop     ? 'L' \
                               : 'U'

/**
 * Useful macros
 */
#define PALLAS_TOKEN_ID_INVALID 0x3fffffff

/**
 * Definition of the type for a token ID
 */
typedef uint32_t TokenId;

/**
 * Most basic element representing Events, Loops or Sequences in Pallas.
 */
typedef struct Token {
  enum TokenType type : 2; /**< Type of our Token. */
  TokenId id : 30;         /**< ID of our Token. */
#ifdef __cplusplus
  /**
   * Construct a Token.
   * @param type Type of the Token.
   * @param id ID of the Token.
   */
  Token(TokenType type, uint32_t id) {
    this->type = type;
    this->id = id;
  }
  /**
   * Construct an Invalid Token.
   */
  Token() {
    type = TypeInvalid;
    id = PALLAS_TOKEN_ID_INVALID;
  }

 public:
  /** Checks for equality between Tokens.
   * @param other Token to check for equality.
   * @return Boolean indicating if the Tokens are equals.
   */
  bool operator==(const Token& other) const { return (other.type == type && other.id == id); }
  bool operator!=(const Token& other) const { return ! operator==(other); }
  /** Checks for ordering between Tokens. Absolute order is decided first on type then on id.
   * @param other Token to check for ordering.
   * @return Boolean indicating if this < other.
   */
  bool operator<(const Token& other) const { return (type < other.type || (type == other.type && id < other.id)); }
  /** Returns true if the Token is a Sequence or a Loop. */
  [[nodiscard]] inline bool isIterable() const { return type == TypeSequence || type == TypeLoop; }
  [[nodiscard]] inline bool isValid() const { return type != TypeInvalid; }
#endif
} Token;
/** Creates a Token for an Event. */
#define PALLAS_EVENT_ID(i) PALLAS(Token)(PALLAS(TypeEvent), i)
/** Creates a Token for a Sequence. */
#define PALLAS_SEQUENCE_ID(i) PALLAS(Token)(PALLAS(TypeSequence), i)
/** Creates a Token for a Loop. */
#define PALLAS_LOOP_ID(i) PALLAS(Token)(PALLAS(TypeLoop), i)

/*************************** Events **********************/
/**
 * Enumeration of event types
 */
enum EventType {
  PALLAS_BLOCK_START,
  PALLAS_BLOCK_END,
  PALLAS_SINGLETON,
};

/**
 * Enumeration of the different events that are recorded by Pallas
 */
enum Record {
  PALLAS_EVENT_BUFFER_FLUSH = 0,                        /**< Signals that the internal buffer was flushed at the given time. */
  PALLAS_EVENT_MEASUREMENT_ON_OFF = 1,                  /**< Signals where the measurement system turned measurement on or off. */
  PALLAS_EVENT_ENTER = 2,                               /**< Indicates that the program enters a code region. */
  PALLAS_EVENT_LEAVE = 3,                               /**< Indicates that the program leaves a code region. */
  PALLAS_EVENT_MPI_SEND = 4,                            /**< Indicates that an MPI send operation was initiated (MPI_SEND).  */
  PALLAS_EVENT_MPI_ISEND = 5,                           /**< Indicates that a non-blocking MPI send operation was initiated (MPI_ISEND). */
  PALLAS_EVENT_MPI_ISEND_COMPLETE = 6,                  /**< Indicates the completion of a non- blocking MPI send operation.  */
  PALLAS_EVENT_MPI_IRECV_REQUEST = 7,                   /**< Indicates that a non-blocking MPI receive operation was initiated (MPI_IRECV). */
  PALLAS_EVENT_MPI_RECV = 8,                            /**< Indicates that an MPI message was received (MPI_RECV).   */
  PALLAS_EVENT_MPI_IRECV = 9,                           /**< Indicates the completion of a non-blocking MPI receive operation completed (MPI_IRECV).  */
  PALLAS_EVENT_MPI_REQUEST_TEST = 10,                   /**< This events appears if the program tests if a request has already completed but the test failed. */
  PALLAS_EVENT_MPI_REQUEST_CANCELLED = 11,              /**< This events appears if the program canceled a request. */
  PALLAS_EVENT_MPI_COLLECTIVE_BEGIN = 12,               /**< An MpiCollectiveBegin record marks the begin of an MPI collective operation (MPI_GATHER, MPI_SCATTER etc.). */
  PALLAS_EVENT_MPI_COLLECTIVE_END = 13,                 /**< Marks the end of an MPI collective */
  PALLAS_EVENT_OMP_FORK = 14,                           /**< Marks that an OpenMP Thread forks a thread team. */
  PALLAS_EVENT_OMP_JOIN = 15,                           /**< Marks that a team of threads is joint and only the master thread continues execution. */
  PALLAS_EVENT_OMP_ACQUIRE_LOCK = 16,                   /**< Marks that a thread acquires an OpenMP lock. */
  PALLAS_EVENT_OMP_RELEASE_LOCK = 17,                   /**< Marks that a thread releases an OpenMP lock. */
  PALLAS_EVENT_OMP_TASK_CREATE = 18,                    /**< Marks that an OpenMP Task was/will be created in the current region. */
  PALLAS_EVENT_OMP_TASK_SWITCH = 19,                    /**< Indicates that the execution of the current task will be suspended and another task starts/restarts its execution.*/
  PALLAS_EVENT_OMP_TASK_COMPLETE = 20,                  /**< Indicates that the execution of an OpenMP task has finished. */
  PALLAS_EVENT_METRIC = 21,                             /**< A metric, stored at the location that recorded it. */
  PALLAS_EVENT_PARAMETER_STRING = 22,                   /**< Marks that in the current region, the specified string parameter has the specified value. */
  PALLAS_EVENT_PARAMETER_INT = 23,                      /**< Marks that in the current region, the specified integer parameter has the specified value. */
  PALLAS_EVENT_PARAMETER_UNSIGNED_INT = 24,             /**< Marks that in the current region, the specified unsigned integer parameter has the specified value. */
  PALLAS_EVENT_THREAD_FORK = 25,                        /**< Marks that a thread forks a thread team. */
  PALLAS_EVENT_THREAD_JOIN = 26,                        /**< Marks that a team of threads is joint and only the master thread continues execution. */
  PALLAS_EVENT_THREAD_TEAM_BEGIN = 27,                  /**< The current location enters the specified thread team. */
  PALLAS_EVENT_THREAD_TEAM_END = 28,                    /**< The current location leaves the specified thread team. */
  PALLAS_EVENT_THREAD_ACQUIRE_LOCK = 29,                /**< Marks that a thread acquires a lock. */
  PALLAS_EVENT_THREAD_RELEASE_LOCK = 30,                /**< Marks that a thread releases a lock. */
  PALLAS_EVENT_THREAD_TASK_CREATE = 31,                 /**< Marks that a task in was/will be created and will be processed by the specified thread team. */
  PALLAS_EVENT_THREAD_TASK_SWITCH = 32,                 /**< Indicates that the execution of the current task will be suspended and another task starts/restarts its execution. Please note that this may change the current call stack of the executing location. */
  PALLAS_EVENT_THREAD_TASK_COMPLETE = 33,               /**< Indicates that the execution of an OpenMP task has finished. */
  PALLAS_EVENT_THREAD_CREATE = 34,                      /**< The location created successfully a new thread. */
  PALLAS_EVENT_THREAD_BEGIN = 35,                       /**< Marks the begin of a thread created by another thread. */
  PALLAS_EVENT_THREAD_WAIT = 36,                        /**< The location waits for the completion of another thread. */
  PALLAS_EVENT_THREAD_END = 37,                         /**< Marks the end of a thread. */
  PALLAS_EVENT_IO_CREATE_HANDLE = 38,                   /**< Marks the creation of a new active I/O handle that can be used by subsequent I/O operation events.*/
  PALLAS_EVENT_IO_DESTROY_HANDLE = 39,                  /**< Marks the end of an active I/O handle's lifetime.*/
  PALLAS_EVENT_IO_SEEK = 41,                            /**< Marks a change of the position, e.g., within a file.*/
  PALLAS_EVENT_IO_CHANGE_STATUS_FLAGS = 42,             /**< Marks a change to the status flags associated with an active I/O handle.*/
  PALLAS_EVENT_IO_DELETE_FILE = 43,                     /**< Marks the deletion of an I/O file.*/
  PALLAS_EVENT_IO_OPERATION_BEGIN = 44,                 /**< Marks the begin of a file operation (read, write, etc.).*/
  PALLAS_EVENT_IO_DUPLICATE_HANDLE = 40,                /**< Marks the duplication of an already existing active I/O handle.*/
  PALLAS_EVENT_IO_OPERATION_TEST = 45,                  /**< Marks an unsuccessful test whether an I/O operation has already finished.*/
  PALLAS_EVENT_IO_OPERATION_ISSUED = 46,                /**< Marks the successful initiation of a non-blocking operation (read, write, etc.) on an active I/O handle.*/
  PALLAS_EVENT_IO_OPERATION_COMPLETE = 47,              /**< Marks the end of a file operation (read, write, etc.) on an active I/O handle.*/
  PALLAS_EVENT_IO_OPERATION_CANCELLED = 48,             /**< Marks the successful cancellation of a non-blocking operation (read, write, etc.) on an active I/O handle.*/
  PALLAS_EVENT_IO_ACQUIRE_LOCK = 49,                    /**< Marks the acquisition of an I/O lock.*/
  PALLAS_EVENT_IO_RELEASE_LOCK = 50,                    /**< Marks the release of an I/O lock.*/
  PALLAS_EVENT_IO_TRY_LOCK = 51,                        /**< Marks when an I/O lock was requested but not granted.*/
  PALLAS_EVENT_PROGRAM_BEGIN = 52,                      /**< Marks the begin of the program.*/
  PALLAS_EVENT_PROGRAM_END = 53,                        /**< Marks the end of the program.*/
  PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_REQUEST = 54,    /**< Indicates that a non-blocking collective operation was initiated.*/
  PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_COMPLETE = 55,   /**< Indicates that a non- blocking collective operation completed.*/
  PALLAS_EVENT_COMM_CREATE = 56,                        /**< Denotes the creation of a communicator.*/
  PALLAS_EVENT_COMM_DESTROY = 57,                       /**< Marks the communicator for destruction at the end of the enclosing MpiCollectiveBegin and MpiCollectiveEnd event pair. */
  PALLAS_EVENT_GENERIC = 58,                            /**< Event record identifier for any other event. */

  PALLAS_EVENT_MAX_ID /**< Max Event Record ID */
};

/**
 * Structure to store an event in PALLAS.
 */
typedef struct Event {
  enum Record record;      /**< ID of the event recorded in the above enumeration of events. */
  uint8_t event_size;      /**< Size of the event. */
  uint8_t event_data[256]; /**< data related to the events. (parameters of functions etc)*/
                           // todo: align on 256
} __attribute__((packed)) Event;

/*************************** Sequences **********************/
#ifdef __cplusplus
/**
 * A Map for counting Tokens.
 *
 * For each token, the size_t member counts the number of time the token appeared in the trace so far.
 *
 *  This class also comes with addition and multiplication, so that we can easily use them.
 */
struct TokenCountMap : public std::map<Token, size_t> {
  /** Adds each (key, value) pair of the other map to this one. */
  void operator+=(const TokenCountMap& other) {
    for (const auto& [key, value] : other) {
      if (count(key) == 0) {
        insert({key, value});
      } else {
        at(key) += value;
      }
    }
  }
  /** Substracts each (key, value) pair of the other map to this one. */
  void operator-=(const TokenCountMap& other) {
    for (const auto& [key, value] : other) {
      if (count(key) == 0) {
        insert({key, -value});
      } else {
        at(key) -= value;
      }
    }
  }
  /** Returns a new map with the same keys, but each value has been multiplied by the given value.
   * @param multiplier Constant multiplier for each value.
   * @returns New map with a copy of the keys and the values. Each value has been multiplied by `multiplier`.
   */
  TokenCountMap operator*(size_t multiplier) const {
    auto otherMap = TokenCountMap();
    for (const auto& [key, value] : *this) {
      otherMap[key] = value * multiplier;
    }
    return otherMap;
  }

  void operator*=(size_t multiplier)  {
    for (const auto& [key, value] : *this) {
      this->at(key) = value * multiplier;
    }
  }

  /** Return the value associated with t, or 0 if t was not found.
   *
   *  This is useful when searching for a token count: if the token has never been encountered, it
   *  won't be found by the map find() function, and we return 0 (instead of an errornous value such
   *  as -1).
   *  @param t Token whose mapped value is accessed.
   *  @returns Mapped value associated with `t`, or 0 if t was not found..
   */
  [[nodiscard]] size_t get_value(const Token& t) const {
    auto res = find(t);
    if (res == end())
      return 0;
    return res->second;
  }
  /** Count the number of Events in the tokenCountMap. */
  [[nodiscard]] size_t getEventCount() const {
    size_t sum = 0;
    for (auto keyValue : *this) {
      Token t = keyValue.first;
      if(t.type == TypeEvent)
	      sum += keyValue.second;
    }
    return sum;
  }
};
#endif

/** Defines a TokenCountMap. In C, defines a char[] of size sizeof(TokenCountMap). */
#define DEFINE_TokenCountMap(name) C_CXX(byte, TokenCountMap) name C([MAP_SIZE])
/** Defines a C++ vector. In C, defines a char[] of size sizeof(std::vector). */
#define DEFINE_Vector(type, name) C_CXX(byte, std::vector<type>) name C_CXX([VECTOR_SIZE], { std::vector<type>() })

/**
 * Structure to store a sequence in PALLAS format.
 */
typedef struct Sequence {
  TokenId id CXX({PALLAS_TOKEN_ID_INVALID});         /**< ID of that sequence. */
  LinkedDurationVector* durations CXX({new LinkedDurationVector()}); /**< Vector of durations for these type of sequences. */
  LinkedVector* timestamps CXX({new LinkedVector()});
  uint32_t hash CXX({0});                            /**< Hash value according to the hash32 function.*/
  DEFINE_Vector(Token, tokens);                      /**< Vector of Token to store the sequence of tokens */
  CXX(private:)
  /**
   * A TokenCountMap counting each token in this Sequence (recursively).
   * It might not be initialized, which is why ::getTokenCount (writing or reading) exists.*/
  DEFINE_TokenCountMap(tokenCount);
#ifdef __cplusplus
 public:
  /** Getter for the size of that Sequence.
   * @returns Number of tokens in that Sequence. */
  [[nodiscard]] size_t size() const { return tokens.size(); }
  /** Indicates whether this Sequence comes from a function
   * (ie begins with Enter and ends with End) or a detected sequence.
   */
  bool isFunctionSequence(const struct Thread* thread) const;
  /** Getter for #tokenCount during the writting process.
   * If need be, counts the number of Token in that Sequence to initialize it.
   * When counting these tokens, it does so backwards. offsetMap allows you to start the count with an offset.
   * @returns Reference to #tokenCount.*/
  TokenCountMap getTokenCountWriting(const Thread* thread, const TokenCountMap* offset = nullptr);
  /** Getter for #tokenCount during the reading process.
   * If need be, counts the number of Token in that Sequence to initialize it.
   * When counting these tokens, it does so forward. offsetMap allows you to start the count with an offset.
   * @returns Reference to #tokenCount.*/
  TokenCountMap getTokenCountReading(const pallas::Thread* thread,
                              const TokenCountMap& threadReaderTokenCountMap,
                              bool isReversedOrder = false);

  /** Tries to guess the name of the sequence
   * @returns A string that describes the sequence.
   */
  std::string guessName(const pallas::Thread* thread);
  size_t getEventCount(const struct Thread* thread);
  ~Sequence() { delete durations; delete timestamps; };
#endif
} Sequence;

/*************************** Loop **********************/

/**
 * Structure to store a Loop in PALLAS format.
 */
typedef struct Loop {
  Token repeated_token;               /**< Token of the Sequence being repeated. */
  Token self_id;                      /**< Token identifying that Loop. */
  uint nb_iterations;                 /**< Number of iterations of that loop. */
#ifdef __cplusplus
  CXX(void addIteration();)           /**< Adds an iteration to the lastest occurence of that loop. */

  /** Tries to guess the name of the loop
   * @returns A string that describes the loop.
   */
  std::string guessName(const pallas::Thread* thread);
#endif
} Loop;

/**
 * Summary for an pallas::Event.
 *
 * Contains the durations for each occurence of that event
 * as well as the number of occurences for that event,
 * and its attributes.
 */
typedef struct EventSummary {
  TokenId id;              /**< ID of the Event */
  Event event;             /**< The Event being summarized.*/
  LinkedDurationVector* durations; /**< Durations for each occurrence of that Event.*/
  size_t nb_occurences;    /**< Number of times that Event has happened. */

  byte* attribute_buffer;       /**< Storage for Attribute.*/
  size_t attribute_buffer_size; /**< Size of #attribute_buffer.*/
  size_t attribute_pos;         /**< Position of #attribute_buffer.*/
#ifdef __cplusplus
 public:
  /** Initialize and EventSummary */
  void initEventSummary(TokenId, const Event&);
#endif
} EventSummary;

typedef uint32_t ThreadId;                                                   /**< Reference for a pallas::Thread. */
#define PALLAS_THREAD_ID_INVALID ((PALLAS(ThreadId))PALLAS_UNDEFINED_UINT32) /**< Invalid ThreadId. */
typedef uint32_t LocationGroupId; /**< Reference for a pallas::LocationGroup. */
#define PALLAS_LOCATION_GROUP_ID_INVALID \
  ((PALLAS(LocationGroupId))PALLAS_UNDEFINED_UINT32) /**< Invalid LocationGroupId. */
#define PALLAS_MAIN_LOCATION_GROUP_ID ((PALLAS(LocationGroupId))PALLAS_LOCATION_GROUP_ID_INVALID - 1)
/**< Main LocationGroupId \todo What is that ?*/

/** A reference for everything after that. */
typedef uint32_t Ref;

#define PALLAS_UNDEFINED_UINT8 ((uint8_t)(~((uint8_t)0u)))
#define PALLAS_UNDEFINED_INT8 ((int8_t)(~(PALLAS_UNDEFINED_UINT8 >> 1)))
#define PALLAS_UNDEFINED_UINT16 ((uint16_t)(~((uint16_t)0u)))
#define PALLAS_UNDEFINED_INT16 ((int16_t)(~(PALLAS_UNDEFINED_UINT16 >> 1)))
#define PALLAS_UNDEFINED_UINT32 ((uint32_t)(~((uint32_t)0u)))
#define PALLAS_UNDEFINED_INT32 ((int32_t)(~(PALLAS_UNDEFINED_UINT32 >> 1)))
#define PALLAS_UNDEFINED_UINT64 ((uint64_t)(~((uint64_t)0u)))
#define PALLAS_UNDEFINED_INT64 ((int64_t)(~(PALLAS_UNDEFINED_UINT64 >> 1)))
#define PALLAS_UNDEFINED_TYPE PALLAS_UNDEFINED_UINT8

/** Reference for a pallas::String */
typedef Ref StringRef;
/** Invalid StringRef */
#define PALLAS_STRINGREF_INVALID ((PALLAS(StringRef))PALLAS_UNDEFINED_UINT32)
/**
 * Define a String reference structure used by PALLAS format.
 *
 * It has an ID and an associated char* with its length
 */
typedef struct String {
  StringRef string_ref; /**< Id of that String.*/
  char* str;            /**< Actual C String */
  int length;           /**< Length of #str.*/
} String;

/** Reference for a pallas::Region */
typedef Ref RegionRef;
/** Invalid RegionRef */
#define PALLAS_REGIONREF_INVALID ((PALLAS(RegionRef))PALLAS_UNDEFINED_UINT32)
/**
 * Define a Region that has an ID and a description.
 */
typedef struct Region {
  RegionRef region_ref; /**< Id of that Region. */
  StringRef string_ref; /**< Description of that Region. */
  /* TODO: add other information (eg. file, line number, etc.)  */
} Region;

/** Reference for an pallas::Attribute. */
typedef Ref AttributeRef;

/** Wrapper for enum pallas::AttributeType. */
typedef uint8_t pallas_type_t;

/**
 * Define an Attribute of a function call.
 */
typedef struct Attribute {
  AttributeRef attribute_ref; /**< Id of that Attribute. */
  StringRef name;             /**< Name of that Attribute. */
  StringRef description;      /**< Description of that Attribute. */
  pallas_type_t type;         /**< Type of that Attribute. */
} Attribute;

/** Reference for a pallas::Group */
typedef Ref GroupRef;
/** Invalid GroupRef */
#define PALLAS_GROUPREF_INVALID ((PALLAS(StringRef))PALLAS_UNDEFINED_UINT32)
/**
 * Define a Group reference structure used by PALLAS format.
 *
 */
typedef struct Group {
  GroupRef group_ref;    /**< Id of that Group.*/
  StringRef name;
  uint32_t numberOfMembers;
  uint64_t* members;
} Group;

/** Reference for a pallas::Comm */
typedef Ref CommRef;
/** Invalid CommRef */
#define PALLAS_COMMREF_INVALID ((PALLAS(StringRef))PALLAS_UNDEFINED_UINT32)
/**
 * Define a Comm reference structure used by PALLAS format.
 *
 */
typedef struct Comm {
  CommRef comm_ref;     /**< Id of that Comm.*/
  StringRef name;
  GroupRef group;
  CommRef parent;
} Comm;

/**
 * A thread contains streams of events.
 *
 * It can be a regular thread (eg. a pthread), or a GPU stream.
 */
typedef struct Thread {
  struct Archive* archive; /**< pallas::Archive containing this Thread. */
  ThreadId id;             /**< Id of this Thread. */

  EventSummary* events;         /**< Array of events recorded in this Thread. */
  unsigned nb_allocated_events; /**< Number of blocks of size pallas:EventSummary allocated in #events. */
  unsigned nb_events;           /**< Number of pallas::EventSummary in #events. */

  Sequence** sequences;            /**< Array of pallas::Sequence recorded in this Thread. */
  unsigned nb_allocated_sequences; /**< Number of blocks of size pallas:Sequence allocated in #sequences. */
  unsigned nb_sequences;           /**< Number of pallas::Sequence in #sequences. */

  /** Map to associate the hash of the pallas::Sequence to their id.*/
#ifdef __cplusplus
  std::unordered_map<uint32_t, std::vector<TokenId>> hashToSequence;
  std::unordered_map<uint32_t, std::vector<TokenId>> hashToEvent;
#else
  byte hashToSequence[UNO_MAP_SIZE];
  byte hashToEvent[UNO_MAP_SIZE];
#endif
  Loop* loops;                 /**< Array of pallas::Loop recorded in this Thread. */
  unsigned nb_allocated_loops; /**< Number of blocks of size pallas:Loop allocated in #loops. */
  unsigned nb_loops;           /**< Number of pallas::Loop in #loops. */
#ifdef __cplusplus
  void loadTimestamps(); /**< Loads all the timestamps for all the Events and Sequences. */
  /** Returns the ID corresponding to the given Event.
   * If there isn't already one, creates a corresponding EventSummary.
   */
  TokenId getEventId(Event* e);
  /** Returns the Event corresponding to the given Token. */
  [[nodiscard]] Event* getEvent(Token) const;
  /** Returns the EventSummary corresponding to the given Token. */
  [[nodiscard]] EventSummary* getEventSummary(Token) const;
  [[nodiscard]] Sequence* getSequence(Token) const;
  [[nodiscard]] Loop* getLoop(Token) const;
  /** Returns the n-th token in the given Sequence/Loop. */
  [[nodiscard]] Token& getToken(Token, int) const;

  /**
   * Return the duration of the thread
   */
  pallas_duration_t getDuration() const;
  /**
   * Return the first timestamp of the thread
   */
  pallas_timestamp_t getFirstTimestamp() const;
  /**
   * Return the last timestamp of the thread
   */
  pallas_timestamp_t getLastTimestamp() const;
  /**
   * Return the number of events of the thread
   */
  size_t getEventCount() const;

  /**
   * Prints the given Token, along with its id.
   * E_E, E_L, E_S indicates an Enter, Leave or Singleton Event.
   * S and L indicates a Sequence or a Loop.
   */
  void printToken(Token) const;
  std::string getTokenString(Token) const;
  void printTokenArray(const Token* array, size_t start_index, size_t len) const; /**< Prints an array of Tokens. */
  void printTokenVector(const std::vector<Token>&) const;                         /**< Prints a vector of Token. */
  void printSequence(Token) const; /**< Prints the Sequence corresponding to the given Token. */
  void printEvent(Event*) const;   /**< Prints an Event. */
  void printEventToString(pallas::Event* e, char* output_str, size_t buffer_size) const;
  void printAttribute(AttributeRef) const;    /**< Prints an Attribute. */
  void printString(StringRef) const;          /**< Prints a String (checks for validity first). */
  void printAttributeRef(AttributeRef) const; /**< Prints an AttributeRef (checks for validity first). */
  void printCommRef(CommRef) const; /**< Prints an CommRef (checks for validity first). */
  void printGroupRef(GroupRef) const; /**< Prints an GroupRef (checks for validity first). */
  void printLocation(Ref) const;              /**< Prints a Ref for a Location (checks for validity first). */
  void printRegion(RegionRef) const;          /**< Prints an RegionRef (checks for validity first). */
  const char* getRegionStringFromEvent(pallas::Event* e) const;
  void printAttributeValue(const struct AttributeData* attr, pallas_type_t type) const; /**< Prints an AttributeValue.*/
  void printAttribute(const struct AttributeData* attr) const;                          /**< Prints an AttributeData. */
  void printAttributeList(const struct AttributeList* attribute_list) const;            /**< Prints an AttributeList. */
  void printEventAttribute(const struct EventOccurence* es) const; /**< Prints an EventOccurence. */
  [[nodiscard]] const char* getName() const;                       /**< Returns the name of this thread. */
  /** Search for a sequence_id that matches the given array as a Sequence.
   * If none of the registered sequence match, register a new Sequence.
   */
  Token getSequenceIdFromArray(Token* token_array, size_t array_len);
  /** Returns the duration for the last - offset given Sequence.*/
  pallas_duration_t getLastSequenceDuration(Sequence* sequence, size_t offset = 0) const;
  void finalizeThread();
  /** Initializes the Thread from an archive and an id.
   * This is used when writing the trace, because it creates Threads using malloc. */
  void initThread(Archive* a, ThreadId id);

  /** Create a blank new Thread. This is used when reading the trace. */
  Thread();

  // Make sure this object is never copied
  Thread(const Thread&) = delete;
  void operator=(const Thread&) = delete;
  ~Thread();
#endif
} Thread;

CXX(
};) /* namespace pallas */
#ifdef __cplusplus
extern "C" {
#endif
  /*************************** C Functions **********************/
  /** Allocates a new thread */
  extern PALLAS(Thread) * pallas_thread_new(void);
  /**
   * Return the thread name of the thread.
   */
  extern const char* pallas_thread_get_name(PALLAS(Thread) * thread);

  /**
   * Return the duration of the thread
   */
  pallas_duration_t get_duration(PALLAS(Thread) *t);

  /**
   * Return the first timestamp of the thread
   */
  pallas_timestamp_t get_first_timestamp(PALLAS(Thread) *t);

  /**
   * Return the last timestamp of the thread
   */
  pallas_timestamp_t get_last_timestamp(PALLAS(Thread) *t);

  /**
   * Return the number of events of the thread
   */
  size_t get_event_count(PALLAS(Thread) *t);


  /**
   * Print the content of sequence seq_id
   */
  extern void pallas_print_sequence(PALLAS(Thread) * thread, PALLAS(Token) seq_id);

  /**
   * Print the subset of a repeated_token array
   */
  extern void pallas_print_token_array(PALLAS(Thread) * thread, PALLAS(Token) * token_array, int index_start,
                                       int index_stop);

  /**
   * Print a repeated_token
   */
  extern void pallas_print_token(PALLAS(Thread) * thread, PALLAS(Token) token);

  /**
   * Print an event
   */
  extern void pallas_print_event(PALLAS(Thread) * thread, PALLAS(Event) * e);

  /**
   * Return the loop whose id is loop_id
   *  - return NULL if loop_id is unknown
   */
  extern struct PALLAS(Loop) * pallas_get_loop(PALLAS(Thread) * thread_trace, PALLAS(Token) loop_id);

  /**
   * Return the sequence whose id is sequence_id
   * @returns NULL if sequence_id is unknown
   */
  extern struct PALLAS(Sequence) * pallas_get_sequence(PALLAS(Thread) * thread_trace, PALLAS(Token) seq_id);

  /**
   * Return the event whose id is event_id
   *  - return NULL if event_id is unknown
   */
  extern struct PALLAS(Event) * pallas_get_event(PALLAS(Thread) * thread_trace, PALLAS(Token) evt_id);

  /**
   * Get the nth token of a given Sequence.
   */
  extern PALLAS(Token) pallas_get_token(PALLAS(Thread) * trace, PALLAS(Token) sequence, int index);
  // Says here that we shouldn't send a pallas::Token, but that's because it doesn't know
  // We made the pallas_token type that matches it. This works as long as the C++ and C version
  // of the struct both have the same elements. Don't care about the rest.

  /** Returns the size of the given sequence. */
  extern size_t pallas_sequence_get_size(PALLAS(Sequence) * sequence);
  /** Returns the nth token of the given sequence. */
  extern PALLAS(Token) pallas_sequence_get_token(PALLAS(Sequence) * sequence, int index);

  /** Does a safe-ish realloc the the given buffer.
   * Given the use of realloc, it does not call the constructor  of the newly created objects.
   *
   * Given a buffer, its current size, a new desired size and its containing object's datatype,
   * changes the size of the buffer using realloc, or if it fails, malloc and memmove, then frees the old buffer.
   * This is better than a realloc because it moves the data around, but it is also slower.
   * Checks for error at malloc.
   */
  extern void* pallas_realloc(void* buffer, int cur_size, int new_size, size_t datatype_size);

#ifdef __cplusplus
};
#endif

/** Doubles the memory allocated for the given buffer. Does not call the constructor for the given objects.
 *
 * Given a buffer, a counter that indicates the number of object it holds, and this object's datatype,
 * doubles the size of the buffer using realloc, or if it fails, malloc and memmove then frees the old buffer.
 * This is better than a realloc because it moves the data around, but it is also slower.
 * Checks for error at malloc.
 */
#define DOUBLE_MEMORY_SPACE(buffer, counter, datatype)                                           \
  do {                                                                                           \
    buffer = (datatype*)pallas_realloc((void*)buffer, counter, (counter) * 2, sizeof(datatype)); \
    counter = (counter) * 2;                                                                     \
  } while (0)

/**
 * Doubles the memory allocated for the given buffer and calls the constructor for the given objects.
 *
 * Given a buffer, a counter that indicates the number of object it holds, and this object's datatype,
 * creates a new buffer using C++'s new keyword, then copies the old data to that buffer, and frees the old buffer.
 */
#define DOUBLE_MEMORY_SPACE_CONSTRUCTOR(buffer, counter, datatype) \
  do {                                                             \
    auto new_buffer = new datatype[counter * 2];                   \
    std::copy(buffer, &buffer[counter], new_buffer);               \
    counter *= 2;                                                  \
    delete[] buffer;                                               \
    buffer = new_buffer;                                           \
  } while (0)

/** Increments the memory allocated for the given buffer by one.
 *
 * Given a buffer, a counter that indicates the number of object it holds, and this object's datatype,
 * Increments the size of the buffer by 1 using realloc, or if it fails, malloc and memmove then frees the old buffer.
 * This is better than a realloc because it moves the data around, but it is also slower.
 * Checks for error at malloc.
 */
#define INCREMENT_MEMORY_SPACE(buffer, counter, datatype)                                        \
  do {                                                                                           \
    buffer = (datatype*)pallas_realloc((void*)buffer, counter, (counter) + 1, sizeof(datatype)); \
    counter = (counter) + 1;                                                                     \
  } while (0)

/**
 * Primitive for DOFOR loops
 */
#define DOFOR(var_name, max) for (int var_name = 0; var_name < max; var_name++)

/* -*-
   mode: c++;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
