/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_hash.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_parameter_handler.h"
#include "pallas/pallas_storage.h"
#include "pallas/pallas_timestamp.h"
#include "pallas/pallas_write.h"

#include <pallas/pallas_read.h>

thread_local int pallas_recursion_shield = 0;
namespace pallas {
/**
 * Compares two arrays of tokens array1 and array2
 */
static inline bool _pallas_arrays_equal(Token* array1, size_t size1, Token* array2, size_t size2) {
    if (size1 != size2)
        return false;
    return memcmp(array1, array2, sizeof(Token) * size1) == 0;
}


static Token getFirstEvent(Token t, const Thread* thread) {
    while (t.type != TypeEvent) {
        if (t.type == TypeSequence) {
            t = thread->getSequence(t)->tokens[0];
        } else {
            t = thread->getLoop(t)->repeated_token;
        }
    }
    return t;
}

static Token getLastEvent(Token t, const Thread* thread) {
    while (t.type != TypeEvent) {
        if (t.type == TypeSequence) {
            t = thread->getSequence(t)->tokens.back();
        } else {
            t = thread->getLoop(t)->repeated_token;
        }
    }
    return t;
}

Sequence& ThreadWriter::getOrCreateSequenceFromArray(pallas::Token* token_array, size_t array_len) {
    if (array_len == 1 && token_array->type == TypeSequence) {
        return thread->sequences[token_array->id];
    }
    // First match it in the thread
    uint32_t hash = hash32_Token(token_array, array_len, SEED);
    auto matched_id = thread->matchSequenceIdFromArray(token_array, array_len, hash);
    if (matched_id.isValid()) {
        return *thread->getSequence(matched_id);
    }

    // Then if it doesn't exist, create it

    if (thread->nb_sequences >= thread->nb_allocated_sequences) {
        pallas_log(DebugLevel::Debug, "Doubling mem space of sequence for thread trace %p\n", this);
        doubleMemorySpaceConstructor(thread->sequences, thread->nb_allocated_sequences);
        for (uint i = thread->nb_allocated_sequences / 2; i < thread->nb_allocated_sequences; i++) {
            thread->sequences[i].durations = new LinkedDurationVector(*parameter_handler);
            thread->sequences[i].exclusive_durations = new LinkedDurationVector(*parameter_handler);
            thread->sequences[i].timestamps = new LinkedVector(*parameter_handler);
        }
    }

    const auto index = thread->nb_sequences++;
    const auto sid = PALLAS_SEQUENCE_ID(index);
    pallas_log(DebugLevel::Debug, "getOrCreateSequenceFromArray: \tSequence not found. Adding it with id=S%lu\n", index);

    Sequence* s = thread->getSequence(sid);
    s->tokens.resize(array_len);
    memcpy(s->tokens.data(), token_array, sizeof(Token) * array_len);
    auto& sequencesWithSameHash = thread->hashToSequence[hash];
    s->hash = hash;
    s->id = sid;
    sequencesWithSameHash.push_back(index);
    return *s;
}

Loop* ThreadWriter::createLoop(Token sequence_id) {
    // for (int i = 0; i < thread_trace->nb_loops; i++) {
    //   if (thread_trace->loops[i].repeated_token.id == sid.id) {
    //     index = i;
    //     pallas_log(DebugLevel::Debug, "createLoop:\tLoop already exists: id=L%d containing S%d\n", index, sid.id);
    //     break;
    //   }
    // }
    if (thread->nb_loops >= thread->nb_allocated_loops) {
        pallas_log(DebugLevel::Debug, "Doubling mem space of loops for thread writer %p's thread trace, cur=%lu\n", this, thread->nb_allocated_loops);
        doubleMemorySpaceConstructor(thread->loops, thread->nb_allocated_loops);
    }
    size_t index = thread->nb_loops++;
    pallas_log(DebugLevel::Debug, "createLoop:\tLoop not found. Adding it with id=L%lu containing S%d\n", index, sequence_id.id);

    Loop& l = thread->loops[index];
    l.nb_iterations = 1;
    l.nb_occurrences = 1;
    l.repeated_token = sequence_id;
    l.self_id = PALLAS_LOOP_ID(index);
    return &l;
}

void ThreadWriter::storeTimestamp(Event* es, pallas_timestamp_t ts) {
    es->timestamps->add(ts);
    if (thread->first_timestamp == PALLAS_TIMESTAMP_INVALID) {
        thread->first_timestamp = ts;
    }
    last_timestamp = ts;
}

void ThreadWriter::storeAttributeList(pallas::Event* es, struct pallas::AttributeList* attribute_list, const size_t occurence_index) {
    attribute_list->index = occurence_index;
    if (es->attribute_pos + attribute_list->struct_size >= es->attribute_buffer_size) {
        if (es->attribute_buffer_size == 0) {
            pallas_log(DebugLevel::Debug, "Allocating attribute memory for event %u\n", es->id);
            es->attribute_buffer_size = NB_ATTRIBUTE_DEFAULT * sizeof(struct pallas::AttributeList);
            es->attribute_buffer = new byte[es->attribute_buffer_size];
            pallas_assert(es->attribute_buffer != nullptr);
        } else {
            pallas_log(DebugLevel::Debug, "Doubling mem space of attributes for event %u\n", es->id);
            doubleMemorySpaceConstructor(es->attribute_buffer, es->attribute_buffer_size);
        }
        pallas_assert(es->attribute_pos + attribute_list->struct_size < es->attribute_buffer_size);
    }

    memcpy(&es->attribute_buffer[es->attribute_pos], attribute_list, attribute_list->struct_size);
    es->attribute_pos += attribute_list->struct_size;

    pallas_log(DebugLevel::Debug, "storeAttributeList: {index: %d, struct_size: %d, nb_values: %d}\n", attribute_list->index, attribute_list->struct_size,
               attribute_list->nb_values);
}

void ThreadWriter::storeToken(Token t, size_t i) {
    pallas_log(DebugLevel::Debug, "storeToken: (%c%d) in seq at callstack[%d] (size: %zu)\n", PALLAS_TOKEN_TYPE_C(t), t.id, cur_depth, sequence_stack[cur_depth].size() + 1);
    sequence_stack[cur_depth].push_back(t);
    index_stack[cur_depth].push_back(i);
    pallas_log(DebugLevel::Debug, "storeToken: %s\n",thread->getTokenArrayString(sequence_stack[cur_depth].data(), 0, sequence_stack[cur_depth].size()).c_str());
    findLoop();
}

void ThreadWriter::incrementLoop(Loop* loop) {
    pallas_log(DebugLevel::Debug, "incrementLoop: + 1 to L%d (to %u)\n", loop->self_id.id, loop->nb_iterations + 1);
    loop->nb_iterations++;
}

Loop* ThreadWriter::unsquashLoop(Loop* loop) {
    pallas_assert(loop->nb_occurrences > 1);
    Loop* newLoop = createLoop(loop->repeated_token);
    loop->nb_occurrences --;
    newLoop->nb_iterations = loop->nb_iterations;
    return newLoop;
}

Loop* ThreadWriter::squashLoop(Loop* loop) {
    for (size_t i = 0; i < loop->self_id.id; i++) {
        auto& otherLoop = thread->loops[i];
        if (otherLoop.repeated_token == loop->repeated_token && otherLoop.nb_iterations == loop->nb_iterations) {
            otherLoop.nb_occurrences ++;
            // Reinitialize the old loop
            if (loop->self_id.id == thread->nb_loops - 1) {
                thread->nb_loops--;
            } else {
                pallas_warn("Could not delete L%d after squashing\n", loop->self_id.id);
            }
            loop->repeated_token = Token();
            loop->self_id = Token();
            loop->nb_iterations = 0;
            loop->nb_occurrences = 0;
            return &otherLoop;
        }
    }
    return loop;
}



void ThreadWriter::replaceTokensInLoop(int loop_len, size_t index_first_iteration, size_t index_second_iteration) {
    if (index_first_iteration > index_second_iteration) {
        const size_t tmp = index_second_iteration;
        index_second_iteration = index_first_iteration;
        index_first_iteration = tmp;
    }

    auto& curTokenSeq = getCurrentTokenSequence();
    auto& curIndexSeq = getCurrentIndexSequence();
    auto& loop_sequence = getOrCreateSequenceFromArray(&curTokenSeq[index_first_iteration], loop_len);
    Loop* loop = createLoop(loop_sequence.id);
    pallas_assert(loop->repeated_token.isValid());
    pallas_assert(loop->self_id.isValid());
    pallas_assert_equals(loop_sequence.id.id, loop->repeated_token.id);
    bool sequence_existed = loop_len == 1 && curTokenSeq[index_first_iteration].type == TypeSequence;
    if (sequence_existed) {
        pallas_assert(loop_sequence.durations->size >= 2);
        pallas_assert(loop_sequence.timestamps->size >= 2);
    }


    if (!sequence_existed) {
        // We need to go back in the current sequence in order to correctly calculate our durations
        // But only if those are new sequences
        // Compute the durations
        const auto [dur_frst_it, excl_dur_frst_it] = getLastSequenceDuration(loop_sequence, 1);
        const auto [dur_scnd_it, excl_dur_scnd_it] = getLastSequenceDuration(loop_sequence, 0);

        loop_sequence.durations->add(dur_frst_it);
        loop_sequence.exclusive_durations->add(excl_dur_frst_it);
        loop_sequence.durations->add(dur_scnd_it);
        loop_sequence.exclusive_durations->add(excl_dur_scnd_it);
#ifdef DEBUG
        bool contains_sequence = false;
        for (const auto t: loop_sequence.tokens ) {
            if (t.type != TypeEvent) {
                contains_sequence = true;
                break;
            }
        }
        if (contains_sequence) {
            pallas_assert_inferior(excl_dur_frst_it, dur_frst_it);
            pallas_assert_inferior(excl_dur_scnd_it, dur_scnd_it);
        }
#endif

        // And add that timestamp to the vectors
        auto first_token = loop_sequence.tokens.front();
        if (first_token.type == TypeEvent) {
            auto first_event = thread->getEvent(first_token);
            loop_sequence.timestamps->add(first_event->timestamps->at(curIndexSeq[index_first_iteration]));
            loop_sequence.timestamps->add(first_event->timestamps->at(curIndexSeq[index_second_iteration]));
        }
        if (first_token.type == TypeSequence) {
            auto first_sequence = thread->getSequence(first_token);
            loop_sequence.timestamps->add(first_sequence->timestamps->at(curIndexSeq[index_first_iteration]));
            loop_sequence.timestamps->add(first_sequence->timestamps->at(curIndexSeq[index_second_iteration]));
        }
        if (first_token.type == TypeLoop) {
            auto& first_loop = *thread->getLoop(first_token);
            auto first_sequence = thread->getSequence(first_loop.repeated_token);
            loop_sequence.timestamps->add(first_sequence->timestamps->at(curIndexSeq[index_first_iteration]));
            loop_sequence.timestamps->add(first_sequence->timestamps->at(curIndexSeq[index_second_iteration]));
        }
        // The current sequence last_timestamp does not need to be updated
    }

    // Resize the Token array and the index array
    curTokenSeq.resize(index_first_iteration);
    curIndexSeq.resize(index_first_iteration);
    curTokenSeq.push_back(loop->self_id);

    // Index of a loop is the occurrence of the first sequence of the loop
    if (sequence_existed) {
        // Then we know we just saw twice the same sequence, hence - 2
        auto* sequence = thread->getSequence(loop_sequence.id);
        curIndexSeq.push_back( sequence->durations->size - 2 );
    } else {
        curIndexSeq.push_back( 0 );
    }
    // Then we increment the loop. We also use this opportunity to check for duplicates

    if (loop->nb_occurrences > 1) {
        loop = unsquashLoop(loop);
        curTokenSeq.back() = loop->self_id;
    }
    incrementLoop(loop);
    loop = squashLoop(loop);
    curTokenSeq.back() = loop->self_id;
}

void ThreadWriter::checkLoopBefore() {
    auto& curTokenSeq = getCurrentTokenSequence();
    auto& curIndexSeq = getCurrentIndexSequence();
    const size_t cur_index = curTokenSeq.size() - 1;

    // First we check if we are repeating the loop exactly
    // Should this happen, it would do:
    // E1 E2 E3 E1 E2 E3 -> L1 = 2 * S1
    // L1 E1 E2 E3 -> L1 S1
    // L1 S1 -> L1 = 3 * S1
    auto* loop = thread->getLoop(curTokenSeq[cur_index - 1]);
    if (loop->repeated_token == curTokenSeq[cur_index]) {
        pallas_log(DebugLevel::Debug, "checkLoopBefore: Last token was the sequence from L%d: S%d\n",
            loop->self_id.id, loop->repeated_token.id);
        if (loop->nb_occurrences > 1) {
            loop = unsquashLoop(loop);
            curTokenSeq[cur_index - 1] = loop->self_id;
        }
        incrementLoop(loop);
        loop = squashLoop(loop);
        curTokenSeq[cur_index - 1] = loop->self_id;
        curTokenSeq.resize(cur_index);
        curIndexSeq.resize(cur_index);
        pallas_log(DebugLevel::Debug, "checkLoopBefore: %s\n", thread->getTokenArrayString(curTokenSeq.data(), 0, curTokenSeq.size()).c_str());
    }
}

void ThreadWriter::findLoopBasic(size_t maxLoopLength) {
    auto& curTokenSeq = getCurrentTokenSequence();
    auto& curIndexSeq = getCurrentIndexSequence();
    if (curTokenSeq.size() <= 1)
        return;
    // First, we check the case where there's a loop before a sequence containing it
    size_t cur_index = curTokenSeq.size() - 1;
    if (curTokenSeq[cur_index - 1].type == TypeLoop) {
        checkLoopBefore();
    }
    cur_index = curTokenSeq.size() - 1;
    for (int loopLength = 1; loopLength < maxLoopLength && loopLength <= cur_index; loopLength++) {
        // search for a loop of loopLength tokens
        const size_t startS1 = cur_index + 1 - loopLength;
        if (cur_index + 1 >= 2 * loopLength) {
            const size_t startS2 = cur_index + 1 - 2 * loopLength;
            /* search for a loop of loopLength tokens */
            if (_pallas_arrays_equal(&curTokenSeq[startS1], loopLength, &curTokenSeq[startS2], loopLength)) {
                pallas_log(DebugLevel::Debug, "findLoopBasic: Found a loop of len %d\n", loopLength);
                replaceTokensInLoop(loopLength, startS1, startS2);
                pallas_log(DebugLevel::Debug, "findLoopBasic: %s\n", thread->getTokenArrayString(curTokenSeq.data(), 0, curTokenSeq.size()).c_str());
                return;
            }
        }
    }
}

void ThreadWriter::findSequence(size_t n) {
    auto& curTokenSeq = getCurrentTokenSequence();
    auto& curTokenIndex = index_stack[cur_depth];
    size_t currentIndex = curTokenSeq.size() - 1;
    n = std::min(currentIndex, n);

    unsigned found_sequence_id = 0;
    for (int array_len = 1; array_len <= n; array_len++) {
        auto token_array = &curTokenSeq[currentIndex - array_len + 1];
        uint32_t hash = hash32_Token(token_array, array_len, SEED);
        if (thread->hashToSequence.find(hash) != thread->hashToSequence.end()) {
            auto& sequencesWithSameHash = thread->hashToSequence[hash];
            if (!sequencesWithSameHash.empty()) {
                for (const auto sid : sequencesWithSameHash) {
                    if (_pallas_arrays_equal(token_array, array_len, thread->sequences[sid].tokens.data(), thread->sequences[sid].size())) {
                        found_sequence_id = sid;
                        break;
                    }
                }
            }
        }
        if (found_sequence_id) {
            pallas_log(DebugLevel::Debug, "Found S%d in %d last tokens\n", found_sequence_id, array_len);
            pallas_assert_equals(curTokenIndex.size(), curTokenSeq.size());

            auto sequence_token = Token(TypeSequence, found_sequence_id);
            auto sequence = thread->getSequence(sequence_token);

            const auto [sequence_duration, exclusive_sequence_duration] = getLastSequenceDuration(*sequence, 0);
            sequence->durations->add(sequence_duration);
            sequence->exclusive_durations->add(exclusive_sequence_duration);
            auto first_token = sequence->tokens.front();
            auto first_token_index = curTokenIndex.size() - array_len;
            if (first_token.type == TypeEvent) {
                auto first_event = thread->getEvent(first_token);
                sequence->timestamps->add(first_event->timestamps->at(curTokenIndex[first_token_index]));
            }
            if (first_token.type == TypeSequence) {
                auto first_sequence = thread->getSequence(first_token);
                sequence->timestamps->add(first_sequence->timestamps->at(curTokenIndex[first_token_index]));
            }
            if (first_token.type == TypeLoop) {
                auto first_loop = thread->getLoop(first_token);
                auto first_sequence = thread->getSequence(first_loop->repeated_token);
                sequence->timestamps->add(first_sequence->timestamps->at(curTokenIndex[first_token_index]));
            }
#ifdef DEBUG
            bool contains_sequence = false;
            for (const auto t: sequence->tokens ) {
                if (t.type != TypeEvent) {
                    contains_sequence = true;
                    break;
                }
            }
            if (contains_sequence) {
                pallas_assert_inferior(exclusive_sequence_duration, sequence_duration);
            }
#endif

            curTokenSeq.resize(curTokenSeq.size() - array_len);
            curTokenIndex.resize(curTokenIndex.size() - array_len);
            storeToken(sequence_token, sequence->timestamps->size - 1);
            pallas_log(DebugLevel::Debug, "findSequence: %s\n", thread->getTokenArrayString(curTokenSeq.data(), 0, curTokenSeq.size()).c_str());

            return;
        }
    }
}

void ThreadWriter::findLoop() {
    auto loopFindingAlgorithm = parameter_handler->getLoopFindingAlgorithm();
    if (loopFindingAlgorithm == LoopFindingAlgorithm::None) {
        return;
    }
    size_t maxLoopLength = (loopFindingAlgorithm == LoopFindingAlgorithm::BasicTruncated) ? parameter_handler->getMaxLoopLength() : SIZE_MAX;
    // First we check if the last tokens are of a Sequence we already know
    findSequence(maxLoopLength);

    // Then we check for loops we haven't found yet
    switch (loopFindingAlgorithm) {
    case LoopFindingAlgorithm::None:
        return;
    case LoopFindingAlgorithm::Basic:
    case LoopFindingAlgorithm::BasicTruncated: {
        findLoopBasic(maxLoopLength);
    } break;
    default:
        pallas_error("Invalid LoopFinding algorithm\n");
    }
}

void ThreadWriter::recordEnterFunction() {
    cur_depth++;
    if (cur_depth >= max_depth) {
        pallas_error("Depth = %d >= max_depth (%d) \n", cur_depth, max_depth);
    }
}

static Record getMatchingRecord(Record r) {
    switch (r) {
    case PALLAS_EVENT_ENTER:
        return PALLAS_EVENT_LEAVE;
    case PALLAS_EVENT_MPI_COLLECTIVE_BEGIN:
        return PALLAS_EVENT_MPI_COLLECTIVE_END;
    case PALLAS_EVENT_OMP_FORK:
        return PALLAS_EVENT_OMP_JOIN;
    case PALLAS_EVENT_THREAD_FORK:
        return PALLAS_EVENT_THREAD_JOIN;
    case PALLAS_EVENT_THREAD_TEAM_BEGIN:
        return PALLAS_EVENT_THREAD_TEAM_END;
    case PALLAS_EVENT_THREAD_BEGIN:
        return PALLAS_EVENT_THREAD_END;
    case PALLAS_EVENT_PROGRAM_BEGIN:
        return PALLAS_EVENT_PROGRAM_END;
    case PALLAS_EVENT_LEAVE:
        return PALLAS_EVENT_ENTER;
    case PALLAS_EVENT_MPI_COLLECTIVE_END:
        return PALLAS_EVENT_MPI_COLLECTIVE_BEGIN;
    case PALLAS_EVENT_OMP_JOIN:
        return PALLAS_EVENT_OMP_FORK;
    case PALLAS_EVENT_THREAD_JOIN:
        return PALLAS_EVENT_THREAD_FORK;
    case PALLAS_EVENT_THREAD_TEAM_END:
        return PALLAS_EVENT_THREAD_TEAM_BEGIN;
    case PALLAS_EVENT_THREAD_END:
        return PALLAS_EVENT_THREAD_BEGIN;
    case PALLAS_EVENT_PROGRAM_END:
        return PALLAS_EVENT_PROGRAM_BEGIN;
    default:
        return PALLAS_EVENT_MAX_ID;
    }
}

void ThreadWriter::recordExitFunction() {
    auto& curTokenSeq = getCurrentTokenSequence();

    // We'll check if the LEAVE Event matches the ENTER
    // This is especially useful to debug stuff in EZTrace
    // If the LEAVE doesn't match the first Event, we'll just ignore it and treat it as a normal event.

    Token first_token = curTokenSeq.front();
    Token last_token = curTokenSeq.back();
    EventData* last_event = &thread->getEvent(last_token)->data;

    if (first_token.type != TypeEvent) {
        pallas_warn("Unexpected Leave event in sequence starting with non-Event: %s/%s\n",
            thread->getTokenString(first_token).c_str(),
            thread->getEventString(last_event).c_str());
        return;
    }

    EventData* first_event = &thread->getEvent(first_token)->data;

    enum Record expected_record = getMatchingRecord(first_event->record);
    if (expected_record == PALLAS_EVENT_MAX_ID) {
        pallas_warn("Unexpected Leave event in sequence starting with non-Enter event: %s/%s\n",
            thread->getEventString(first_event).c_str(),
            thread->getEventString(last_event).c_str()
            );
        return;
    }

    if (last_event->record != expected_record) {
        pallas_warn("Unexpected Leave event in sequence starting with non-matching Enter event:%s/%s\n",
        thread->getEventString(first_event).c_str(),
        thread->getEventString(last_event).c_str());
        return;
    }

    auto& sequence = getOrCreateSequenceFromArray(curTokenSeq.data(), curTokenSeq.size());


    const auto  [computed_duration, computed_exclusive_duration] = getLastSequenceDuration(sequence, 0);
#ifdef DEBUG
    const pallas_duration_t sequence_duration = last_timestamp - sequence_start_timestamp[cur_depth];
    pallas_log(DebugLevel::Debug, "Computed duration = %lu\nSequence duration = %lu\n", computed_duration, sequence_duration);
    pallas_assert(computed_duration == sequence_duration);
#endif

    pallas_log(DebugLevel::Debug, "Exiting function, closing %s, start=%lu\n", thread->getTokenString(sequence.id).c_str(), sequence_start_timestamp[cur_depth]);
    sequence.timestamps->add(sequence_start_timestamp[cur_depth]);
    sequence.exclusive_durations->add(computed_exclusive_duration);
    sequence.durations->add(computed_duration);

#ifdef DEBUG
    bool contains_sequence = false;
    for (const auto t: sequence.tokens ) {
        if (t.type != TypeEvent) {
            contains_sequence = true;
            break;
        }
    }
    if (contains_sequence) {
        pallas_assert_inferior(computed_exclusive_duration, computed_duration);
    }
#endif



    cur_depth--;
    storeToken(sequence.id, sequence.timestamps->size - 1);
    curTokenSeq.clear();
    index_stack[cur_depth+1].clear();

    // We need to reset the token vector
    // Calling vector::clear() might be a better way to do that,
    // but depending on the implementation it might force a bunch of realloc, which isn't great.
}  // namespace pallas

size_t ThreadWriter::storeEvent(enum EventType event_type, TokenId event_id, pallas_timestamp_t ts, AttributeList* attribute_list) {
    ts = timestamp(ts);
    if (event_type == PALLAS_BLOCK_START) {
        recordEnterFunction();
        sequence_start_timestamp[cur_depth] = ts;
    }

    Token token = Token(TypeEvent, event_id);

    Event* es = &thread->events[event_id];
    size_t occurrence_index = es->nb_occurrences++;
    pallas_log(DebugLevel::Debug, "storeEvent: %s @ %lu\n", thread->getTokenString(token).c_str(), ts);
    storeTimestamp(es, ts);
    storeToken(token, occurrence_index);

    if (attribute_list)
        storeAttributeList(es, attribute_list, occurrence_index);

    if (event_type == PALLAS_BLOCK_END) {
        recordExitFunction();
    }
    return occurrence_index;
}

void ThreadWriter::threadClose() {
    while (cur_depth > 0) {
        pallas_warn("Closing unfinished sequence (lvl %d)\n", cur_depth);
        recordExitFunction();
    }
    // Then we need to store the main sequence
    auto& mainSequence = thread->sequences[0];
    mainSequence.tokens = sequence_stack[0];
    pallas_log(DebugLevel::Debug, "Last sequence token: (%d.%d)\n", mainSequence.tokens.back().type, mainSequence.tokens.back().id);
    pallas_timestamp_t duration = last_timestamp - thread->first_timestamp;
    mainSequence.durations->add(duration);
    mainSequence.exclusive_durations->add(0);
    // TODO Maybe not the correct exclusive duration for the main thread ? Who knows, who cares.
    mainSequence.timestamps->add(thread->first_timestamp);
    thread->store(thread->archive->dir_name, parameter_handler);
}
ThreadWriter::~ThreadWriter() {
    delete[] sequence_stack;
    delete[] index_stack;
    delete[] sequence_start_timestamp;
}

ThreadWriter::ThreadWriter(Archive& a, ThreadId thread_id) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    pallas_log(DebugLevel::Debug, "ThreadWriter(%u)::open\n", thread_id);
    parameter_handler = new ParameterHandler();
    if (a.global_archive) {
        a.global_archive->parameter_handler = parameter_handler;
    }

    pthread_mutex_lock(&a.lock);
    while (a.nb_threads >= a.nb_allocated_threads) {
        doubleMemorySpaceConstructor(a.threads, a.nb_allocated_threads);
    }
    thread = new Thread;
    pallas_thread_rank = a.nb_threads;
    a.threads[a.nb_threads++] = thread;
    thread->archive = &a;
    thread->id = thread_id;

    thread->nb_allocated_events = NB_EVENT_DEFAULT;
    thread->events = new Event[thread->nb_allocated_events]();
    thread->nb_events = 0;

    thread->nb_allocated_sequences = NB_SEQUENCE_DEFAULT;
    thread->sequences = new Sequence[thread->nb_allocated_sequences]();
    thread->nb_sequences = 0;
    for (int i = 0; i < thread->nb_allocated_sequences; i++) {
        thread->sequences[i].durations = new LinkedDurationVector(*parameter_handler);
        thread->sequences[i].exclusive_durations = new LinkedDurationVector(*parameter_handler);
        thread->sequences[i].timestamps = new LinkedVector(*parameter_handler);
    }

    thread->hashToSequence = std::unordered_map<uint32_t, std::vector<TokenId>>();
    thread->hashToEvent = std::unordered_map<uint32_t, std::vector<TokenId>>();

    thread->nb_allocated_loops = NB_LOOP_DEFAULT;
    thread->loops = new Loop[thread->nb_allocated_loops]();
    thread->nb_loops = 0;

    pthread_mutex_unlock(&a.lock);
    max_depth = CALLSTACK_DEPTH_DEFAULT;
    sequence_stack = new std::vector<Token>[max_depth];
    index_stack = new std::vector<size_t>[max_depth];

    // We need to initialize the main Sequence (Sequence 0)
    auto& mainSequence = thread->sequences[0];
    mainSequence.id = PALLAS_SEQUENCE_ID(0);
    thread->nb_sequences = 1;

    last_timestamp = PALLAS_TIMESTAMP_INVALID;
    sequence_start_timestamp = new pallas_timestamp_t[max_depth];

    cur_depth = 0;

    pallas_recursion_shield--;
}

TokenId ThreadWriter::getEventId(EventData* e) {
    pallas_log(DebugLevel::Max, "getEventId: Searching for event {.event_type=%d}\n", e->record);

    uint32_t hash = hash32(reinterpret_cast<byte*>(e), sizeof(EventData), SEED);
    auto& eventWithSameHash = thread->hashToEvent[hash];
    if (!eventWithSameHash.empty()) {
        if (eventWithSameHash.size() > 1) {
            pallas_log(DebugLevel::Debug, "Found more than one event with the same hash: %lu\n", eventWithSameHash.size());
        }
        for (const auto eid : eventWithSameHash) {
            if (memcmp(e, &thread->events[eid].data, e->event_size) == 0) {
                pallas_log(DebugLevel::Debug, "getEventId: \t found with id=%u\n", eid);
                return eid;
            }
        }
    }

    if (thread->nb_events >= thread->nb_allocated_events) {
        pallas_log(DebugLevel::Debug, "Doubling mem space of events for thread trace %p\n", this);
        doubleMemorySpaceConstructor(thread->events, thread->nb_allocated_events);
    }

    TokenId index = thread->nb_events++;
    pallas_log(DebugLevel::Max, "getEventId: \tNot found. Adding it with id=%d\n", index);

    auto* new_event = new (&thread->events[index]) Event(index, *e);
    new_event->timestamps = new LinkedVector(*parameter_handler);

    // In-place initialisation
    thread->hashToEvent[hash].push_back(index);

    return index;
}

std::array<pallas_duration_t, 2> ThreadWriter::getLastSequenceDuration(const Sequence& sequence, size_t offset) const {
    pallas_timestamp_t start_ts;
    pallas_timestamp_t end_ts;
    auto& curIndexSeq = getCurrentIndexSequence();
    // First we need to compute the inclusive duration. That's end_ts - start_ts
    // Computing start_ts
    Token start_token = sequence.tokens.front();
    size_t start_index = curIndexSeq[curIndexSeq.size() - sequence.tokens.size() * ( 1 + offset ) ];
    switch (start_token.type) {
    case TypeEvent:
        start_ts = thread->getEvent(start_token)->timestamps->at(start_index);
        break;
    case TypeSequence:
        start_ts = thread->getSequence(start_token)->timestamps->at(start_index);
        break;
    case TypeLoop: {
        auto* l = thread->getLoop(start_token);
        start_ts = thread->getSequence(l->repeated_token)->timestamps->at(start_index);
        break;
    }
    default:
        pallas_error("Incorrect Token\n");
    }

    // Computing end_ts
    Token end_token = sequence.tokens.back();
    size_t end_index = curIndexSeq[curIndexSeq.size() - 1 - sequence.tokens.size() * offset];
    switch (end_token.type) {
    case TypeEvent:
        end_ts = thread->getEvent(end_token)->timestamps->at(end_index);
        break;
    case TypeSequence: {
        auto* s = thread->getSequence(end_token);
        end_ts = s->timestamps->at(end_index) + s->durations->at(end_index);
        break;
    }
    case TypeLoop: {
        auto* l = thread->getLoop(end_token);
        auto* s = thread->getSequence(l->repeated_token);
        size_t last_sequence_index = end_index + l->nb_iterations - 1;
        end_ts = s->timestamps->at(last_sequence_index) + s->durations->at(last_sequence_index);
        break;
    }
    default:
        pallas_error("Incorrect Token\n");
    }
    pallas_duration_t inclusive_duration = end_ts - start_ts;

    // Then compute the exclusive duration or the block duration, depending on the sequence's type
    pallas_duration_t block_duration = 0;
    for (size_t i = 0; i< sequence.tokens.size(); i ++) {
        auto token = sequence.tokens[i];
        auto index = curIndexSeq[curIndexSeq.size() - sequence.tokens.size() * ( 1 + offset ) + i];
        if (token.type == TypeEvent) {
            continue;
        }
        if (token.type == TypeSequence) {
            auto* s = thread->getSequence(token);
            block_duration += (s->type == SEQUENCE_BLOCK)?s->durations->at(index):s->exclusive_durations->at(index);
            continue;
        }
        if (token.type == TypeLoop) {
            auto* l = thread->getLoop(token);
            auto* s = thread->getSequence(l->repeated_token);
            for (size_t j = 0; j < l->nb_iterations; j ++) {
                block_duration += (s->type == SEQUENCE_BLOCK)?s->durations->at(index+j):s->exclusive_durations->at(index+j);
            }
            continue;
        }
    }
    pallas_assert_inferior_equal(block_duration,  inclusive_duration);

    return {inclusive_duration, (sequence.type == SEQUENCE_BLOCK)? (inclusive_duration - block_duration) : block_duration};
}
}  // namespace pallas

/* C Callbacks */
pallas::ThreadWriter* pallas_thread_writer_new(pallas::Archive* archive, pallas::ThreadId thread_id) {
    return new pallas::ThreadWriter(*archive, thread_id);
}

extern void pallas_global_archive_close(pallas::GlobalArchive* archive) {
    archive->store(archive->dir_name, archive->parameter_handler);
};

extern void pallas_thread_writer_close(pallas::ThreadWriter* thread_writer) {
    thread_writer->threadClose();
};

extern void pallas_archive_close(PALLAS(Archive) * archive) {
    archive->store(archive->dir_name, nullptr);
};

extern void pallas_store_event(PALLAS(ThreadWriter) * thread_writer,
                               enum PALLAS(EventType) event_type,
                               PALLAS(TokenId) id,
                               pallas_timestamp_t ts,
                               PALLAS(AttributeList) * attribute_list) {
    thread_writer->storeEvent(event_type, id, ts, attribute_list);
};
extern void pallas_thread_writer_delete(PALLAS(ThreadWriter) * thread_writer) {
    delete thread_writer;
};
/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
