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

Token ThreadWriter::getSequenceIdFromArray(pallas::Token* token_array, size_t array_len) {
    uint32_t hash = hash32((uint8_t*)(token_array), array_len * sizeof(pallas::Token), SEED);
    pallas_log(DebugLevel::Debug, "getSequenceIdFromArray: Searching for sequence {.size=%zu, .hash=%x}\n", array_len, hash);
    auto& sequencesWithSameHash = thread->hashToSequence[hash];
    if (!sequencesWithSameHash.empty()) {
        if (sequencesWithSameHash.size() > 1) {
            pallas_log(DebugLevel::Debug, "Found more than one sequence with the same hash\n");
        }
        for (const auto sid : sequencesWithSameHash) {
            if (_pallas_arrays_equal(token_array, array_len, thread->sequences[sid]->tokens.data(), thread->sequences[sid]->size())) {
                pallas_log(DebugLevel::Debug, "getSequenceIdFromArray: \t found with id=%u\n", sid);
                return PALLAS_SEQUENCE_ID(sid);
            }
        }
    }

    if (thread->nb_sequences >= thread->nb_allocated_sequences) {
        pallas_log(DebugLevel::Debug, "Doubling mem space of sequence for thread trace %p\n", this);
        doubleMemorySpaceConstructor(thread->sequences, thread->nb_allocated_sequences);
        for (uint i = thread->nb_allocated_sequences / 2; i < thread->nb_allocated_sequences; i++) {
            thread->sequences[i] = new Sequence;
            thread->sequences[i]->durations = new LinkedDurationVector(*parameter_handler);
            thread->sequences[i]->exclusive_durations = new LinkedDurationVector(*parameter_handler);
            thread->sequences[i]->timestamps = new LinkedVector(*parameter_handler);
        }
    }

    const auto index = thread->nb_sequences++;
    const auto sid = PALLAS_SEQUENCE_ID(index);
    pallas_log(DebugLevel::Debug, "getSequenceIdFromArray: \tSequence not found. Adding it with id=S%lu\n", index);

    Sequence* s = thread->getSequence(sid);
    s->tokens.resize(array_len);
    memcpy(s->tokens.data(), token_array, sizeof(Token) * array_len);
    s->hash = hash;
    s->id = sid.id;
    sequencesWithSameHash.push_back(index);
    return sid;
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

    Loop* l = &thread->loops[index];
    l->nb_iterations = 1;
    l->repeated_token = sequence_id;
    l->self_id = PALLAS_LOOP_ID(index);
    return l;
}

void ThreadWriter::storeTimestamp(EventSummary* es, pallas_timestamp_t ts) {
    es->timestamps->add(ts);
    if (thread->first_timestamp == PALLAS_TIMESTAMP_INVALID)
        thread->first_timestamp = ts;

    last_timestamp = ts;
}

void ThreadWriter::storeAttributeList(pallas::EventSummary* es, struct pallas::AttributeList* attribute_list, const size_t occurence_index) {
    attribute_list->index = occurence_index;
    if (es->attribute_pos + attribute_list->struct_size >= es->attribute_buffer_size) {
        if (es->attribute_buffer_size == 0) {
            pallas_log(DebugLevel::Debug, "Allocating attribute memory for event %u\n", es->id);
            es->attribute_buffer_size = NB_ATTRIBUTE_DEFAULT * sizeof(struct pallas::AttributeList);
            es->attribute_buffer = new uint8_t[es->attribute_buffer_size];
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

void Loop::addIteration() {
    pallas_log(DebugLevel::Debug, "addIteration: + 1 to L%d (to %u)\n", self_id.id, nb_iterations + 1);
    nb_iterations++;
}

void ThreadWriter::replaceTokensInLoop(int loop_len, size_t index_first_iteration, size_t index_second_iteration) {
    if (index_first_iteration > index_second_iteration) {
        const size_t tmp = index_second_iteration;
        index_second_iteration = index_first_iteration;
        index_first_iteration = tmp;
    }

    auto& curTokenSeq = getCurrentTokenSequence();
    auto& curIndexSeq = getCurrentIndexSequence();
    bool sequence_existed = false;
    Token loop_sequence_id;
    // If the configuration is just "S3 S3"
    // Then we don't need to create S4 = S3, we can just repeat S3
    if (loop_len == 1 && curTokenSeq[index_first_iteration].type == TypeSequence) {
        sequence_existed = true;
        loop_sequence_id = curTokenSeq[index_first_iteration];

    } else {
        loop_sequence_id = getSequenceIdFromArray(&curTokenSeq[index_first_iteration], loop_len);
    }
    Loop* loop = createLoop(loop_sequence_id);
    Sequence* loop_sequence = thread->getSequence(loop_sequence_id);
    pallas_assert(loop->repeated_token.isValid());
    pallas_assert(loop->self_id.isValid());
    pallas_assert_equals(loop_sequence_id.id, loop->repeated_token.id);
    if (sequence_existed) {
        pallas_assert(loop_sequence->durations->size >= 2);
        pallas_assert(loop_sequence->timestamps->size >= 2);
    }


    if (!sequence_existed) {
        // We need to go back in the current sequence in order to correctly calculate our durations
        // But only if those are new sequences
        // Compute the durations
        const auto [duration_first_iteration, exclusive_duration_first_iteration] = getLastSequenceDuration(loop_sequence, 1);
        const auto [duration_second_iteration, exclusive_duration_second_iteration] = getLastSequenceDuration(loop_sequence, 0);

        loop_sequence->durations->add(duration_first_iteration);
        loop_sequence->exclusive_durations->add(exclusive_duration_first_iteration);
        loop_sequence->durations->add(duration_second_iteration);
        loop_sequence->exclusive_durations->add(exclusive_duration_second_iteration);

        // And add that timestamp to the vectors
        auto& tokenCount = loop_sequence->getTokenCountWriting(thread);
        auto first_event = getFirstEvent(loop_sequence->tokens.front(), thread);
        auto first_token_summary = thread->events[first_event.id];
        size_t nb_first_token = first_token_summary.timestamps->size;
        loop_sequence->timestamps->add(first_token_summary.timestamps->at(nb_first_token- 2 * tokenCount[first_event]));
        loop_sequence->timestamps->add(first_token_summary.timestamps->at(nb_first_token - tokenCount[first_event]));

        // The current sequence last_timestamp does not need to be updated
    }

    // Resize the Token array and the index array
    curTokenSeq.resize(index_first_iteration);
    curIndexSeq.resize(index_first_iteration);
    curTokenSeq.push_back(loop->self_id);

    // Index of a loop is the occurrence of the first sequence of the loop
    if (sequence_existed) {
        // Then we know we just saw twice the same sequence, hence - 2
        auto* sequence = thread->getSequence(loop_sequence_id);
        curIndexSeq.push_back( sequence->durations->size - 2 );
    } else {
        curIndexSeq.push_back( 0 );
    }

    loop->addIteration();
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
    auto l = thread->getLoop(curTokenSeq[cur_index - 1]);
    if (l->repeated_token == curTokenSeq[cur_index]) {
        pallas_log(DebugLevel::Debug, "findLoopBasic: Last token was the sequence from L%d: S%d\n", l->self_id.id, l->repeated_token.id);
        l->addIteration();
        curTokenSeq.resize(cur_index);
        curIndexSeq.resize(cur_index);
        pallas_log(DebugLevel::Debug, "findLoopBasic: %s\n", thread->getTokenArrayString(curTokenSeq.data(), 0, curTokenSeq.size()).c_str());
        return;
    }
    // We are checking to see if the loop at cur_index-1 could be extended
    // For that, we check if the current token could be the start of a new loop
    Token first_loop_token = getFirstEvent(l->repeated_token, thread);
    Token first_cur_token = getFirstEvent(curTokenSeq[cur_index], thread);
    if (first_cur_token == first_loop_token)
        return;
    // That means we're sure we're not in another iteration of our loop
    // Now we'll check all the other loops to see if there's one that's the exact same as this one
    bool found_loop = false;
    TokenId identical_other_loop = 0;
    for (TokenId lid = 0; lid < l->self_id.id && lid < thread->nb_loops; lid++) {
        if (thread->loops[lid].repeated_token == l->repeated_token && thread->loops[lid].nb_iterations == l->nb_iterations) {
            found_loop = true;
            identical_other_loop = lid;
            break;
        }
    }
    if (!found_loop)
        return;
    // We just found a loop that's the same as this one !
    // And it's not going to be edited
    // So we can replace it
    curTokenSeq[cur_index - 1] = thread->loops[identical_other_loop].self_id;
    pallas_log(DebugLevel::Debug, "findLoopBasic: replaced a Loop by its earlier occurrence: L%d -> L%d\n", l->self_id.id, identical_other_loop);
    pallas_log(DebugLevel::Debug, "findLoopBasic: %s\n", thread->getTokenArrayString(curTokenSeq.data(), 0, curTokenSeq.size()).c_str());
    if (l->self_id.id == thread->nb_loops - 1) {
        // If this was the last loop added ( which it will often be ), then remove it
        pallas_log(DebugLevel::Debug, "findLoopBasic: Remove last loop L%d ( S%d )\n", l->self_id.id, l->repeated_token.id);
        thread->nb_loops--;
    } else {
        pallas_log(DebugLevel::Error, "findLoopBasic: Couldn't remove duplicated loop L%d ( S%d ). It will stay in the grammar.\n", l->self_id.id, l->repeated_token.id);
    }
    // Cleaning it at least.
    l->nb_iterations = 0;
    l->repeated_token = {TypeInvalid, PALLAS_TOKEN_ID_INVALID};
    l->self_id = {TypeInvalid, PALLAS_TOKEN_ID_INVALID};
    // And now, we have to check that by changing this loop, we didn't just create a repeating pattern
    // For example, if we had " E1 L1 E1 L2 E2"
    // Which was changed to   " E1 L1 E1 L1 E2"
    // Then we'd want to detect that repetition for sure !

    // So first things first: we need to remove that last event
    Token last_token = curTokenSeq.back();
    size_t last_token_index = curIndexSeq.back();
    curTokenSeq.pop_back();
    curIndexSeq.pop_back();

    findLoop();

    storeToken(last_token, last_token_index);
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
    if (n >= currentIndex)
        n = currentIndex;

    unsigned found_sequence_id = 0;
    for (int array_len = 1; array_len <= n; array_len++) {
        auto token_array = &curTokenSeq[currentIndex - array_len + 1];
        uint32_t hash = hash32(reinterpret_cast<uint8_t*>(token_array), array_len * sizeof(Token), SEED);
        if (thread->hashToSequence.find(hash) != thread->hashToSequence.end()) {
            auto& sequencesWithSameHash = thread->hashToSequence[hash];
            if (!sequencesWithSameHash.empty()) {
                for (const auto sid : sequencesWithSameHash) {
                    if (_pallas_arrays_equal(token_array, array_len, thread->sequences[sid]->tokens.data(), thread->sequences[sid]->size())) {
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

            const auto [sequence_duration, exclusive_sequence_duration] = getLastSequenceDuration(sequence, 0);
            sequence->durations->add(sequence_duration);
            sequence->exclusive_durations->add(exclusive_sequence_duration);
            sequence->timestamps->add(last_timestamp - sequence_duration);

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

#ifdef DEBUG
    // check that the sequence is not weird

    Token first_token = curTokenSeq.front();
    Token last_token = curTokenSeq.back();

    if (first_token.type == TypeEvent) {
        Event* first_event = thread->getEvent(first_token);
        Event* last_event = thread->getEvent(last_token);

        enum Record expected_record = getMatchingRecord(first_event->record);
        if (expected_record == PALLAS_EVENT_MAX_ID) {
            pallas_warn("Unexpected start_event record: %s\n", thread->getEventString(first_event).c_str());
            pallas_abort();
        }

        if (last_event->record != expected_record) {
            pallas_warn("Unexpected close event:\n\tStart_sequence event: \t%s as E%d\n\tEnd_sequence event: \t%s as E%d\n", thread->getEventString(first_event).c_str(),
                        first_token.id, thread->getEventString(last_event).c_str(), last_token.id);
            if (cur_depth > 1) {
                auto& underSequence = sequence_stack[cur_depth - 1];
                enum Record expected_start_record = getMatchingRecord(last_event->record);
                if (expected_start_record == PALLAS_EVENT_MAX_ID) {
                    pallas_warn("Unexpected last_event record:\n\t%s\n", thread->getEventString(last_event).c_str());
                    pallas_abort();
                }
                pallas_warn("Currently recorded last event is wrong by one layer, adding the correct Leave Event.\n");
                curTokenSeq.resize(curTokenSeq.size() - 1);
                Event e;
                e.event_size = offsetof(Event, event_data);
                e.record = expected_record;
                memcpy(e.event_data, first_event->event_data, first_event->event_size);
                e.event_size = first_event->event_size;
                TokenId e_id = getEventId(&e);
                pallas_warn("\tInserting %s as E%d at end of curSequence\n", thread->getEventString(&e).c_str(), e_id);
                storeEvent(PALLAS_BLOCK_END, e_id, getTimestamp(), nullptr);
                pallas_warn("\tInserting %s as E%d at end of layer under curSequence\n", thread->getEventString(last_event).c_str(), last_token.id);
                underSequence.push_back(last_token);
                recordExitFunction();
                return;
            }
        }
    }

    if (curTokenSeq != sequence_stack[cur_depth]) {
        pallas_error("cur_seq=%p, but og_seq[%d] = %p\n", &curTokenSeq, cur_depth, &sequence_stack[cur_depth]);
    }
#endif

    const Token seq_id = getSequenceIdFromArray(curTokenSeq.data(), curTokenSeq.size());
    auto* seq = thread->sequences[seq_id.id];


    const auto  [computed_duration, computed_exclusive_duration] =
        getLastSequenceDuration(seq, 0);
#ifdef DEBUG
    const pallas_duration_t sequence_duration = last_timestamp - sequence_start_timestamp[cur_depth];
    pallas_log(DebugLevel::Debug, "Computed duration = %lu\nSequence duration = %lu\n", computed_duration, sequence_duration);
    pallas_assert(computed_duration == sequence_duration);
#endif

    pallas_log(DebugLevel::Debug, "Exiting function, closing %s, start=%lu\n", thread->getTokenString(seq_id).c_str(), sequence_start_timestamp[cur_depth]);

    cur_depth--;
    /* upper_seq is the sequence that called cur_seq */
    auto& upperTokenSeq = getCurrentTokenSequence();

    seq->timestamps->add(sequence_start_timestamp[cur_depth]);
    seq->exclusive_durations->add(computed_exclusive_duration);
    seq->durations->add(computed_duration);
    storeToken(seq_id, seq->timestamps->size - 1);

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

    EventSummary* es = &thread->events[event_id];
    size_t occurrence_index = es->nb_occurences++;
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
    // Then we need to close the main sequence
    auto& mainSequence = thread->sequences[0];
    mainSequence->tokens = sequence_stack[0];
    pallas_log(DebugLevel::Debug, "Last sequence token: (%d.%d)\n", mainSequence->tokens.back().type, mainSequence->tokens.back().id);
    pallas_timestamp_t duration = last_timestamp - thread->first_timestamp;
    mainSequence->durations->add(duration);
    mainSequence->exclusive_durations->add(duration);
    // TODO Maybe not the correct exclusive duration for the main thread ? Who knows, who cares.
    mainSequence->timestamps->add(thread->first_timestamp);
    thread->finalizeThread();
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
    thread_rank = a.nb_threads;
    a.threads[a.nb_threads++] = thread;
    thread->archive = &a;
    thread->id = thread_id;

    thread->nb_allocated_events = NB_EVENT_DEFAULT;
    thread->events = new EventSummary[thread->nb_allocated_events]();
    thread->nb_events = 0;

    thread->nb_allocated_sequences = NB_SEQUENCE_DEFAULT;
    thread->sequences = new Sequence*[thread->nb_allocated_sequences]();
    thread->nb_sequences = 0;
    for (int i = 0; i < thread->nb_allocated_sequences; i++) {
        thread->sequences[i] = new Sequence();
        thread->sequences[i]->durations = new LinkedDurationVector(*parameter_handler);
        thread->sequences[i]->exclusive_durations = new LinkedDurationVector(*parameter_handler);
        thread->sequences[i]->timestamps = new LinkedVector(*parameter_handler);
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
    mainSequence->id = 0;
    thread->nb_sequences = 1;

    last_timestamp = PALLAS_TIMESTAMP_INVALID;
    sequence_start_timestamp = new pallas_timestamp_t[max_depth];

    cur_depth = 0;

    pallas_recursion_shield--;
}

void Archive::close() {
    pallasStoreArchive(this);
}

void GlobalArchive::close() {
    pallasStoreGlobalArchive(this);
}

TokenId ThreadWriter::getEventId(Event* e) {
    pallas_log(DebugLevel::Max, "getEventId: Searching for event {.event_type=%d}\n", e->record);

    uint32_t hash = hash32(reinterpret_cast<uint8_t*>(e), sizeof(Event), SEED);
    auto& eventWithSameHash = thread->hashToEvent[hash];
    if (!eventWithSameHash.empty()) {
        if (eventWithSameHash.size() > 1) {
            pallas_log(DebugLevel::Debug, "Found more than one event with the same hash: %lu\n", eventWithSameHash.size());
        }
        for (const auto eid : eventWithSameHash) {
            if (memcmp(e, &thread->events[eid].event, e->event_size) == 0) {
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

    auto* new_event = new (&thread->events[index]) EventSummary(index, *e);
    new_event->timestamps = new LinkedVector(*parameter_handler);

    // In-place initialisation
    thread->hashToEvent[hash].push_back(index);

    return index;
}

std::array<pallas_duration_t, 2> ThreadWriter::getLastSequenceDuration(Sequence* sequence, size_t offset) const {
    pallas_timestamp_t start_ts;
    pallas_timestamp_t end_ts;
    auto& curIndexSeq = getCurrentIndexSequence();
    Token start_token = sequence->tokens.front();
    size_t start_index = curIndexSeq[curIndexSeq.size() - sequence->tokens.size() * ( 1 + offset ) ];

    switch (start_token.type) {
    case TypeEvent:
        start_ts = thread->getEventSummary(start_token)->timestamps->at(start_index);
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

    Token end_token = sequence->tokens.back();
    size_t end_index = curIndexSeq[curIndexSeq.size() - 1 - sequence->tokens.size() * offset];
    switch (end_token.type) {
    case TypeEvent:
        end_ts = thread->getEventSummary(end_token)->timestamps->at(end_index);
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
    auto inclusive_duration = end_ts - start_ts;
    auto exclusive_duration = inclusive_duration;
    // Then we need to compute the exclusive duration
    for (size_t i = start_index; i < end_index; i ++) {
        auto token = sequence->tokens[i];
        if (token.type == TypeEvent) {
            continue;
        }
        if (token.type == TypeSequence) {
            auto* s = thread->getSequence(token);
            exclusive_duration -= s->durations->at(curIndexSeq[i]);
            continue;
        }
        if (token.type == TypeLoop) {
            auto* l = thread->getLoop(token);
            auto* s = thread->getSequence(l->repeated_token);
            for (size_t j = 0; j < l->nb_iterations; j ++) {
                exclusive_duration -= s->durations->at(curIndexSeq[i] - j);
            }
            continue;
        }
    }
    pallas_assert_inferior_equal(exclusive_duration,  inclusive_duration);

    return {inclusive_duration, inclusive_duration};
}
}  // namespace pallas

/* C Callbacks */
pallas::ThreadWriter* pallas_thread_writer_new(pallas::Archive* archive, pallas::ThreadId thread_id) {
    return new pallas::ThreadWriter(*archive, thread_id);
}

extern void pallas_global_archive_close(pallas::GlobalArchive* archive) {
    archive->close();
};

extern void pallas_thread_writer_close(pallas::ThreadWriter* thread_writer) {
    thread_writer->threadClose();
};

extern void pallas_archive_close(PALLAS(Archive) * archive) {
    archive->close();
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
