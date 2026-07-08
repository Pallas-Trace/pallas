/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_read.h"

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"

/** Default Callstack Size. */
#define DEFAULT_CALLSTACK_DEPTH 64

namespace pallas {

CallstackFrame::CallstackFrame() {
    this->current_timestamp = 0;
    this->frame_index = 0;
}

CallstackFrame::~CallstackFrame() = default;

Cursor::Cursor() {
    current_frame_index = 0;
    callstack_capacity = DEFAULT_CALLSTACK_DEPTH;
    callstack = new CallstackFrame[callstack_capacity]();
    currentFrame = callstack;
    read_ended = false;
}

Cursor::~Cursor() {
    delete[] callstack;
}

Cursor::Cursor(const Cursor& other) {
    current_frame_index = other.current_frame_index;
    callstack_capacity = other.callstack_capacity;
    callstack = new CallstackFrame[callstack_capacity]();
    DOFOR(i, current_frame_index) {
        callstack[i].tokenCount = other.callstack[i].tokenCount;
        callstack[i].frame_index = other.callstack[i].frame_index;
        callstack[i].callstack_iterable = other.callstack[i].callstack_iterable;
        callstack[i].current_timestamp = other.callstack[i].current_timestamp;
    }
    currentFrame = &callstack[current_frame_index];
    read_ended = other.read_ended;
}
Cursor& Cursor::operator=(const Cursor& other) {
    current_frame_index = other.current_frame_index;
    callstack_capacity = other.callstack_capacity;
    callstack = new CallstackFrame[callstack_capacity]();
    DOFOR(i, current_frame_index) {
        callstack[i].tokenCount = other.callstack[i].tokenCount;
        callstack[i].frame_index = other.callstack[i].frame_index;
        callstack[i].callstack_iterable = other.callstack[i].callstack_iterable;
        callstack[i].current_timestamp = other.callstack[i].current_timestamp;
    }
    currentFrame = &callstack[current_frame_index];
    read_ended = other.read_ended;
    return *this;
}

ThreadReader::ThreadReader(Archive* archive, ThreadId threadId, int read_flags) {
    // Setup the basic
    this->archive = archive;
    this->pallas_read_flag = read_flags;

    pallas_assert(threadId != PALLAS_THREAD_ID_INVALID);
    this->thread_trace = archive->get_thread(threadId);
    pallas_assert(this->thread_trace != nullptr);

    Token root_token = Token(TypeSequence, this->thread_trace->sequence_root);

    if (debugLevel >= DebugLevel::Verbose) {
        pallas_log(DebugLevel::Verbose, "init callstack for thread %d\n", threadId);
        pallas_log(DebugLevel::Verbose, "The trace contains:\n");
        this->thread_trace->printSequence(root_token);
    }

    if (this->thread_trace->getSequence(root_token)->size() == 0) {
        pallas_warn("Thread %s is empty\n", this->thread_trace->getName());
    } else {
        // And initialize the callstack
        // ie set the cursor on the first event
        this->currentState.current_frame_index = 0;
        this->currentState.currentFrame = &currentState.callstack[0];
        this->currentState.currentFrame->callstack_iterable = root_token;
        this->currentState.currentFrame->current_timestamp = this->thread_trace->first_timestamp;
    }
}

const Token& ThreadReader::get_frame_in_callstack(int frame_number) const {
    if (frame_number < 0 || frame_number >= currentState.callstack_capacity) {
        pallas_error("Frame number is too high or negative: %d\n", frame_number);
    }
    return currentState.callstack[frame_number].callstack_iterable;
}

const Token& ThreadReader::get_token_in_callstack(int frame_number) const {
    if (frame_number < 0 || frame_number >= currentState.callstack_capacity) {
        pallas_error("Frame number is too high or negative: %d\n", frame_number);
    }
    auto sequence = get_frame_in_callstack(frame_number);
    pallas_assert(sequence.isIterable());
    return thread_trace->getToken(sequence, currentState.callstack[frame_number].frame_index);
}

pallas_timestamp_t ThreadReader::get_current_timestamp() const {
    return currentState.currentFrame->current_timestamp;
}

void ThreadReader::print_current_token() const {
    std::cout << thread_trace->getTokenString(poll_current_token()) << std::endl;
}
const Token& ThreadReader::get_current_iterable() const {
    return currentState.currentFrame->callstack_iterable;
}
void ThreadReader::print_current_sequence() const {
    thread_trace->printSequence(get_current_iterable());
}

void ThreadReader::print_callstack() const {
    printf("# Callstack (depth: %d) ------------\n", currentState.current_frame_index + 1);
    for (int i = 0; i < currentState.current_frame_index + 1; i++) {
        auto current_sequence_id = get_frame_in_callstack(i);
        auto current_token = get_token_in_callstack(i);

        printf("%.*s[%d] ", i * 2, "                       ", i);
        std::cout << thread_trace->getTokenString(current_sequence_id) << std::endl;

        if (current_sequence_id.type == TypeLoop) {
            auto* loop = thread_trace->getLoop(current_sequence_id);
            printf(" iter %d/%d", currentState.callstack[i].frame_index, loop->nb_iterations);
            pallas_assert(currentState.callstack[i].frame_index < currentState.callstack_capacity);
        } else if (current_sequence_id.type == TypeSequence) {
            auto* sequence = thread_trace->getSequence(current_sequence_id);
            printf(" pos %d/%lu", currentState.callstack[i].frame_index, sequence->size());
            pallas_assert(currentState.callstack[i].frame_index < currentState.callstack_capacity);
        }

        std::cout << "\t-> " << thread_trace->getTokenString(current_token) << std::endl;
    }
}
Event* ThreadReader::get_event(Token event) const {
    pallas_assert(event.type == TypeEvent);
    return thread_trace->getEvent(event);
    pallas_error("Given event (%d) was invalid\n", event.id);
}
pallas_timestamp_t ThreadReader::get_event_timestamp(Token event, int occurrence_id) const {
    pallas_assert(event.type == TypeEvent);
    auto summary = get_event(event);
    if (0 <= occurrence_id && occurrence_id < summary->nb_occurrences) {
        return summary->timestamps->at(occurrence_id);
    }
    pallas_error("Given occurrence_id (%d) was invalid for event %d\n", occurrence_id, event.id);
}
bool ThreadReader::is_end_of_sequence(int current_index, Token sequence_id) const {
    if (sequence_id.type == TypeSequence) {
        auto* sequence = thread_trace->getSequence(sequence_id);
        return current_index + 1 >= sequence->size();
        // We are in a sequence and index is beyond the end of the sequence
    }
    pallas_error("The given sequence_id was the wrong type: %d\n", sequence_id.type);
}
bool ThreadReader::is_end_of_loop(int current_index, Token loop_id) const {
    if (loop_id.type == TypeLoop) {
        auto* loop = thread_trace->getLoop(loop_id);
        pallas_assert(current_index < loop->nb_iterations);
        return current_index + 1 >= loop->nb_iterations;
        // We are in a loop and index is beyond the number of iterations
    }
    pallas_error("The given loop_id was the wrong type: %d\n", loop_id.type);
}
bool ThreadReader::is_end_of_block(int index, Token iterable_token) const {
    switch (iterable_token.type) {
    case TypeSequence:
        return is_end_of_sequence(index, iterable_token);
    case TypeLoop:
        return is_end_of_loop(index, iterable_token);
    case TypeEvent:
        return false;
    case TypeInvalid:
        pallas_error("Current frame is invalid");
    }
    return false;
}
bool ThreadReader::is_end_of_current_block() const {
    pallas_assert(currentState.current_frame_index >= 0);

    int current_index = currentState.currentFrame->frame_index;
    auto current_iterable_token = currentState.currentFrame->callstack_iterable;

    return is_end_of_block(current_index, current_iterable_token);
}
bool ThreadReader::is_end_of_trace() const {
    return currentState.read_ended;
}

pallas_duration_t ThreadReader::get_loop_duration(Token loop_id) const {
    pallas_assert(loop_id.type == TypeLoop);
    const auto* loop = thread_trace->getLoop(loop_id);
    const auto* sequence = thread_trace->getSequence(loop->repeated_token);

    const Token sequence_id = loop->repeated_token;

    size_t offset;
    if (get_current_iterable() != loop_id)
        offset = currentState.currentFrame->tokenCount.get_value(sequence_id);
    else
        offset = currentState.callstack[currentState.current_frame_index - 1].tokenCount.get_value(sequence_id);
    const size_t nIterations = loop->nb_iterations;
    return sequence->timestamps->at(offset + nIterations - 1) - sequence->timestamps->at(offset) + sequence->durations->at(offset + nIterations - 1);
}

EventOccurrence ThreadReader::get_event_occurrence(Token event_id, size_t occurrence_id) const {
    auto eventOccurrence = EventOccurrence();
    auto* es = get_event(event_id);
    eventOccurrence.event = &thread_trace->getEvent(event_id)->data;

    eventOccurrence.timestamp = es->timestamps->at(occurrence_id);
    eventOccurrence.attributes = get_event_attribute_list(event_id, occurrence_id);
    return eventOccurrence;
}

SequenceOccurrence ThreadReader::get_sequence_occurrence(Token sequence_id, size_t occurrence_id) const {
    auto sequenceOccurrence = SequenceOccurrence();
    sequenceOccurrence.sequence = thread_trace->getSequence(sequence_id);

    sequenceOccurrence.timestamp = sequenceOccurrence.sequence->timestamps->at(occurrence_id);
    sequenceOccurrence.duration = sequenceOccurrence.sequence->durations->at(occurrence_id);
    sequenceOccurrence.full_sequence = nullptr;

    return sequenceOccurrence;
};

LoopOccurrence ThreadReader::get_loop_occurrence(Token loop_id, size_t occurrence_id) const {
    auto loopOccurrence = LoopOccurrence();
    loopOccurrence.loop = thread_trace->getLoop(loop_id);
    loopOccurrence.nb_iterations = loopOccurrence.loop->nb_iterations;
    loopOccurrence.full_loop = nullptr;
    loopOccurrence.timestamp = currentState.currentFrame->current_timestamp;
    loopOccurrence.duration = get_loop_duration(loop_id);
    return loopOccurrence;
}
size_t ThreadReader::get_current_token_count(Token t) const {
    return currentState.currentFrame->tokenCount[t];
}

AttributeList* ThreadReader::get_event_attribute_list(Token event_id, size_t occurrence_id) const {
    auto* summary = get_event(event_id);
    if (summary->attribute_buffer == nullptr)
        return nullptr;

    byte* read_pos = summary->attribute_buffer;
    AttributeList *attribute_list = (pallas::AttributeList*)read_pos;
    while (attribute_list->index != occurrence_id) { /* move to the next attribute until we reach the needed index */
        read_pos += attribute_list->struct_size;
        if (read_pos > summary->attribute_buffer + summary->attribute_buffer_size) {
            return nullptr;
        }
        attribute_list = (pallas::AttributeList*)read_pos;
    }
    if (attribute_list->index > occurrence_id) {
        pallas_error("Error fetching attribute %zu. We went too far (cur position: %d) !\n", occurrence_id, attribute_list->index);
    }
    return attribute_list;
}

void ThreadReader::guess_sequences_names(std::map<pallas::Sequence*, std::string>& names) const {
    // Let's call the main sequence "main"
    names[&thread_trace->sequences[thread_trace->sequence_id_map[thread_trace->sequence_root]]] = "main";

    for (int i = 1; i < thread_trace->nb_sequences; i++) {
        pallas::Sequence* s = &thread_trace->sequences[i];

        if (names.count(s) == 0) {
            // The sequence is not named yet

            bool name_found = false;
            if (s->size() <= 4) {
                // for small (enter/leave function) sequence, use the name of the function
                pallas::Token t_start = s->tokens[0];
                if (t_start.type == pallas::TypeEvent) {
                    EventData* data = &thread_trace->getEvent(t_start)->data;
                    if (data->record == pallas::PALLAS_EVENT_ENTER) {
                        const char* event_name = thread_trace->getRegionStringFromEvent(data);
                        // TODO if that's an MPI call (eg MPI_Send, MPI_Allreduce, ...)
                        //      we may want to get the function parameters (eg. dest, tag, ...)
                        names[s] = std::string(event_name);
                        name_found = true;
                    } else if (data->record == pallas::PALLAS_EVENT_THREAD_BEGIN) {
                        names[s] = "main";
                        name_found = true;
                    }
                }
            }

            if (!name_found) {
                // is it a loop ?
                for (int j = 0; j < thread_trace->nb_loops; j++) {
                    pallas::Loop& l = thread_trace->loops[j];
                    if (thread_trace->getSequence(l.repeated_token) == s) {
                        char buff[128];
                        snprintf(buff, sizeof(buff), "Loop_%d", l.self_id.id);
                        names[s] = std::string(buff);
                        name_found = true;
                        break;
                    }
                }
            }

            if (!name_found) {
                // probably a complex/long sequence. Just name it randomly
                char buff[128];
                snprintf(buff, sizeof(buff), "Sequence_%d", s->id.id);
                names[s] = std::string(buff);
            }
        }
    }
}

//******************* EXPLORATION FUNCTIONS ********************

const Token& ThreadReader::poll_current_token() const {
    return get_token_in_callstack(currentState.current_frame_index);
}

Token ThreadReader::poll_next_token(int flags) const {
    if (currentState.current_frame_index < 0)
        // return an invalid token
        return Token();

    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;

    // NOTE:
    // are these the same field, redundancy?
    int current_frame = currentState.current_frame_index;
    int current_index = currentState.currentFrame->frame_index;
    auto current_iterable_token = currentState.currentFrame->callstack_iterable;
    pallas_assert(current_iterable_token.isIterable());

    if (const Token current_token = poll_current_token(); current_token.isIterable()) {
        if (current_token.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
            return thread_trace->getSequence(current_token)->tokens.at(0);
        }
        if (current_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
            return thread_trace->getLoop(current_token)->repeated_token;
        }
    }
    while (is_end_of_block(current_index, current_iterable_token)) {
        if (current_frame == 0)
            return Token();
        // NOTE:
        // merge logic
        if (current_iterable_token.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
            current_frame--;
            current_index = currentState.currentFrame->frame_index;
            current_iterable_token = currentState.currentFrame->callstack_iterable;
        } else if (current_iterable_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
            current_frame--;
            current_index = currentState.currentFrame->frame_index;
            current_iterable_token = currentState.currentFrame->callstack_iterable;
        } else {
            return Token();
        }
    }
    return thread_trace->getToken(current_iterable_token, current_index + 1);
}

Token ThreadReader::poll_previous_token(int flags) const {
    if (currentState.current_frame_index < 0)
        return Token();

    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;

    int current_frame = currentState.current_frame_index;
    int current_index = currentState.currentFrame->frame_index;
    auto current_iterable_token = currentState.currentFrame->callstack_iterable;
    pallas_assert(current_iterable_token.isIterable());

    while (current_index == 0) {
        if (current_frame == 0)
            return Token();
        if (current_iterable_token.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
            current_frame--;
            current_index = currentState.currentFrame->frame_index;
            current_iterable_token = currentState.currentFrame->callstack_iterable;
        } else if (current_iterable_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
            current_frame--;
            current_index = currentState.currentFrame->frame_index;
            current_iterable_token = currentState.currentFrame->callstack_iterable;
        } else {
            return Token();
        }
    }
    Token result = thread_trace->getToken(current_iterable_token, current_index - 1);
    while (result.isIterable()) {
        if (result.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
            result = thread_trace->getSequence(result)->tokens.at(thread_trace->getSequence(result)->tokens.size() - 1);
        } else if (result.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
            result = thread_trace->getLoop(result)->repeated_token;
        } else if (result.type == TypeSequence || result.type == TypeLoop) {
            break;
        }
    }
    return result;
}

bool ThreadReader::move_to_next_token(int flags) {
    // Check if we've reached the end of the trace
    if (is_end_of_trace()) {
        pallas_log(DebugLevel::Debug, "End of trace %d!\n", __LINE__);
        return false;
    }

    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;

    pallas_assert(get_current_iterable().isIterable());

   // If we can enter a block, then we enter
    if (enter_if_start_of_block(flags)) {
        return true;
    }

    // Exit every block we can
    bool exited = exit_if_end_of_block(flags);
    while (exit_if_end_of_block(flags)) {
    }

    if (is_end_of_current_block()) {
        return false;
    }

    auto previous_token = this->poll_current_token();
    auto& currentTokenCount = currentState.currentFrame->tokenCount;

    // Update token count according to current token
    if (previous_token.type == TypeSequence) {
        auto seq = thread_trace->getSequence(previous_token);
        currentTokenCount += seq->getTokenCountReading(thread_trace);
    }
    if (previous_token.type == TypeLoop) {
        auto loop = thread_trace->getLoop(previous_token);
        auto loop_sequence = thread_trace->getSequence(loop->repeated_token);
        auto loopCount = loop->nb_iterations;
        for (size_t i = 0; i < loopCount; i++) {
            currentTokenCount += loop_sequence->getTokenCountReading(thread_trace);
            currentTokenCount[loop->repeated_token] += 1;
        }
    }

    currentTokenCount[previous_token]++;
    currentState.currentFrame->frame_index++;
    auto current_token = poll_current_token();

    /* Update the current timestamp. */
    pallas_timestamp_t& new_timestamp = currentState.currentFrame->current_timestamp;
    switch (current_token.type) {
    case TypeEvent:
        new_timestamp = get_event(current_token)->timestamps->at(currentState.currentFrame->tokenCount[current_token]);
        break;
    case TypeLoop: {
        auto loop = thread_trace->getLoop(current_token);
        auto loop_sequence = thread_trace->getSequence(loop->repeated_token);
        new_timestamp = loop_sequence->timestamps->at(currentState.currentFrame->tokenCount[loop->repeated_token]);
        break;
    }
    case TypeSequence: {
        auto seq = thread_trace->getSequence(current_token);
        currentState.currentFrame->current_timestamp = seq->timestamps->at(currentState.currentFrame->tokenCount[current_token]);
        break;
    }

    case TypeInvalid:
        pallas_error("Token is Invalid");
    }
    return true;
}

bool ThreadReader::move_to_next_token_in_block() {
    return move_to_next_token(PALLAS_READ_FLAG_NO_UNROLL);
}

bool ThreadReader::move_to_previous_token(int flags) {
    // Check if we've reached the beginning of the trace
    if (currentState.current_frame_index < 0) {
        pallas_log(DebugLevel::Debug, "End of trace %d!\n", __LINE__);
        return false;
    }

    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;

    pallas_assert(get_current_iterable().isIterable());

    if (currentState.currentFrame->frame_index == 0) {
        if (currentState.current_frame_index <= 1) {
            return false;
        }
        Token current_iterable_token = get_current_iterable();
        if (current_iterable_token.type == TypeSequence && !(flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE)) {
            return false;
        }
        if (current_iterable_token.type == TypeLoop && !(flags & PALLAS_READ_FLAG_UNROLL_LOOP)) {
            return false;
        }
        leave_block();
        return true;
    }

    /* Get the previous token in the current sequence. */
    auto previous_token = poll_previous_token(PALLAS_READ_FLAG_NO_UNROLL);

    auto& currentTokenCount = currentState.currentFrame->tokenCount;
    currentState.currentFrame->frame_index --;
    if (previous_token.type == TypeEvent) {
        currentState.currentFrame->current_timestamp = get_event_timestamp(previous_token, currentTokenCount[previous_token]);
    }
    if (previous_token.type == TypeSequence) {
        pallas_error("Not implemented yet");
        auto* s = thread_trace->getSequence(previous_token);
        /* TODO This following line should only be done when we don't unroll the sequences. */
        currentTokenCount -= s->getTokenCountReading(thread_trace);
        currentState.currentFrame->current_timestamp = s->timestamps->at(currentTokenCount[previous_token]);
        if (flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE ) {
            /* This isn't as straightforward as you may think
             * Because you need to do
             * while (the last token is a Loop/Sequence)
             *    Get inside of it
            `*/
        }
    }
    if (previous_token.type == TypeLoop) {
        // TODO Implement this
        pallas_error("Not implemented yet");
    }
    currentTokenCount[previous_token] --;
    return true;

}
bool ThreadReader::move_to_previous_token_in_block() {
    return move_to_previous_token(PALLAS_READ_FLAG_NO_UNROLL);
}

Token ThreadReader::get_next_token(int flags) {
    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;
    if (!move_to_next_token(flags))
        return Token();
    if (is_end_of_trace())
        return Token();
    return poll_current_token();
}
Token ThreadReader::get_previous_token(int flags) {
    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;
    if (!move_to_previous_token(flags))
        return Token();
    return poll_current_token();
}

void ThreadReader::enter_block() {
    auto new_block = poll_current_token();
    pallas_assert(new_block.isIterable());
    if (debugLevel >= DebugLevel::Debug) {
        pallas_log(DebugLevel::Debug, "[%d] Enter Block ", currentState.current_frame_index);
        print_current_token();
        printf("\n");
    }

    currentState.current_frame_index++;
    if (currentState.current_frame_index >= currentState.callstack_capacity) {
        pallas_log(DebugLevel::Debug, "Doubling size of callstack buffer");
        currentState.callstack_capacity *= 2;
        CallstackFrame *new_buffer = new CallstackFrame[currentState.callstack_capacity]();
        DOFOR(i, currentState.current_frame_index) {
            new_buffer[i] = currentState.callstack[i];
        }
        delete[] currentState.callstack;
        currentState.callstack = new_buffer;
    }
    currentState.currentFrame++;
    currentState.currentFrame->frame_index = 0;
    currentState.currentFrame->current_timestamp = currentState.callstack[currentState.current_frame_index - 1].current_timestamp;
    currentState.currentFrame->callstack_iterable = new_block;
    currentState.currentFrame->tokenCount = (currentState.currentFrame - 1)->tokenCount;
#ifdef DEBUG
    if (new_block.type == TypeSequence) {
        auto current_timestamp = currentState.currentFrame->current_timestamp;
        auto seq = thread_trace->getSequence(new_block);
        auto theorical_timestamp = seq->timestamps->at(currentState.currentFrame->tokenCount[new_block]);
        if (theorical_timestamp != current_timestamp) {
            int a = 1;
        }
        // pallas_assert(theorical_timestamp == current_timestamp);
    }
#endif
}

void ThreadReader::leave_block() {
    if (debugLevel >= DebugLevel::Debug) {
        pallas_log(DebugLevel::Debug, "[%d] Leave \n", currentState.current_frame_index);
    }

    pallas_assert(currentState.current_frame_index > 0);

    currentState.current_frame_index--;
    currentState.currentFrame--;

    if (debugLevel >= DebugLevel::Debug && currentState.current_frame_index >= 0) {
        pallas_assert(get_current_iterable().isIterable());
    }

    if (currentState.current_frame_index == 0) {
        currentState.read_ended = true;
    }
}

bool ThreadReader::exit_if_end_of_block(int flags) {
    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;

    if (currentState.current_frame_index == 0)
        return false;
    int current_index = currentState.currentFrame->frame_index;
    auto current_iterable_token = currentState.currentFrame->callstack_iterable;
    if (current_iterable_token.type == TypeSequence) {
        if (is_end_of_sequence(current_index, current_iterable_token) && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
            /* We've reached the end of a sequence. Leave the block. */
            leave_block();
            return true;
        }
    } else {
        if (is_end_of_loop(current_index, current_iterable_token) && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
            /* We've reached the end of the loop. Leave the block. */
            leave_block();
            return true;
        }
    }
    return false;
}
bool ThreadReader::enter_if_start_of_block(int flags) {
    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;

    auto current_token = poll_current_token();
    if (!current_token.isIterable())
        return false;
    if (current_token.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
        enter_block();
        return true;
    }
    if (current_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
        enter_block();
        return true;
    }
    return false;
}

Cursor ThreadReader::create_checkpoint() const {
    return Cursor(this->currentState);
}
void ThreadReader::load_checkpoint(Cursor* checkpoint) {
    currentState = *checkpoint;
}

ThreadReader::~ThreadReader() {
}

ThreadReader::ThreadReader(const ThreadReader& other) = default;

ThreadReader::ThreadReader(ThreadReader&& other) noexcept {
    archive = other.archive;
    thread_trace = other.thread_trace;
    currentState = other.currentState;
    pallas_read_flag = other.pallas_read_flag;
    // Set other to 0 for everything
    other.archive = nullptr;
    other.thread_trace = nullptr;
    other.currentState = Cursor();
    other.pallas_read_flag = 0;
}

ThreadReader& ThreadReader::operator=(const ThreadReader& other) {
    archive = other.archive;
    thread_trace = other.thread_trace;
    currentState = other.currentState;
    pallas_read_flag = other.pallas_read_flag;
    return *this;
}

ThreadReader& ThreadReader::operator=(ThreadReader&& other) noexcept {
    archive = other.archive;
    thread_trace = other.thread_trace;
    currentState = other.currentState;
    pallas_read_flag = other.pallas_read_flag;
    // Set other to 0 for everything
    other.archive = nullptr;
    other.thread_trace = nullptr;
    other.currentState = Cursor();
    other.pallas_read_flag = 0;
    return *this;
}


MultiThreadReader::MultiThreadReader(std::vector<Thread *> threads) {
    this->n_threads = threads.size();
    this->readers = new ThreadReader[this->n_threads];
    this->current_reader = nullptr;

    for (size_t i = 0; i < this->n_threads; i++) {
        Thread *thread = threads[i];
        this->readers[i] = ThreadReader(thread->archive, thread->id, PALLAS_READ_FLAG_UNROLL_ALL);
        while (!this->readers[i].is_end_of_trace() && this->readers[i].poll_current_token().type != TypeEvent) {
            this->readers[i].move_to_next_token();
        }
    }
}

MultiThreadReader::MultiThreadReader(GlobalArchive &trace) {
    auto threads = trace.getThreadList();
    n_threads = threads.size();
    readers = new ThreadReader[n_threads];
    current_reader = &readers[0];

    for (size_t i = 0; i < n_threads; i++) {
        Thread *thread = threads[i];
        auto* r = new (&readers[i]) ThreadReader(thread->archive, thread->id, PALLAS_READ_FLAG_UNROLL_ALL);
    }
    update_minimum_reader();
}

MultiThreadReader::~MultiThreadReader() {
    delete[] this->readers;
}

Token MultiThreadReader::poll_current_token() const {
    if (this->current_reader == nullptr) {
        return INVALID_TOKEN;
    }
    return this->current_reader->poll_current_token();
}

bool MultiThreadReader::update_minimum_reader() {
    pallas_timestamp_t min_timestamp = std::numeric_limits<unsigned long>::max();
    for (size_t i = 0; i < n_threads; i ++) {
        if (!readers[i].is_end_of_trace() && readers[i].get_current_timestamp() < min_timestamp) {
            current_reader = &readers[i];
            min_timestamp = readers[i].get_current_timestamp();
        }
    }
    if (min_timestamp == std::numeric_limits<unsigned long>::max()) {
        return false;
    }
    return true;
}

bool MultiThreadReader::move_to_next_token() {
    if (this->current_reader != nullptr) {
        this->current_reader->move_to_next_token();
    }
    return update_minimum_reader();
}

Token MultiThreadReader::get_next_token() {
    if (this->move_to_next_token()) {
        return this->poll_current_token();
    }
    return INVALID_TOKEN;
}


/* C bindings */

extern ThreadReader pallas_create_thread_reader(Archive* archive, ThreadId threadId, int options) {
    return {archive, threadId, options};
}
extern void pallas_print_current_token(ThreadReader* thread_reader) {
    thread_reader->print_current_token();
}
extern Token pallas_get_current_iterable(ThreadReader* thread_reader) {
    return thread_reader->get_current_iterable();
}
extern void pallas_print_current_sequence(ThreadReader* thread_reader) {
    thread_reader->print_current_sequence();
}
extern void pallas_print_callstack(ThreadReader* thread_reader) {
    thread_reader->print_callstack();
}
extern Event* pallas_get_thread_reader_event(ThreadReader* thread_reader, Token event) {
    return thread_reader->get_event(event);
}
extern pallas_timestamp_t pallas_get_event_timestamp(ThreadReader* thread_reader, Token event, int occurrence_id) {
    return thread_reader->get_event_timestamp(event, occurrence_id);
}
extern bool pallas_is_end_of_sequence(ThreadReader* thread_reader, int current_index, Token sequence_id) {
    return thread_reader->is_end_of_sequence(current_index, sequence_id);
}
extern bool pallas_is_end_of_loop(ThreadReader* thread_reader, int current_index, Token loop_id) {
    return thread_reader->is_end_of_loop(current_index, loop_id);
}
extern bool pallas_is_end_of_current_block(ThreadReader* thread_reader) {
    return thread_reader->is_end_of_current_block();
}
extern bool pallas_is_end_of_trace(ThreadReader* thread_reader) {
    return thread_reader->is_end_of_trace();
}
extern size_t pallas_get_occurrence(ThreadReader *thread_reader, Token token) {
    return thread_reader->get_current_token_count(token);
}
extern EventOccurrence pallas_get_event_occurrence(ThreadReader* thread_reader, Token event_id, size_t occurrence_id) {
    return thread_reader->get_event_occurrence(event_id, occurrence_id);
}
extern SequenceOccurrence pallas_get_sequence_occurrence(ThreadReader* thread_reader, Token sequence_id, size_t occurrence_id) {
    return thread_reader->get_sequence_occurrence(sequence_id, occurrence_id);
}
extern LoopOccurrence pallas_get_loop_occurrence(ThreadReader* thread_reader, Token loop_id, size_t occurrence_id) {
    return thread_reader->get_loop_occurrence(loop_id, occurrence_id);
}
extern AttributeList* pallas_get_event_attribute_list(ThreadReader* thread_reader, Token event_id, size_t occurrence_id) {
    return thread_reader->get_event_attribute_list(event_id, occurrence_id);
}
extern Token pallas_poll_current_token(ThreadReader* thread_reader) {
    return thread_reader->poll_current_token();
}
extern Token pallas_poll_next_token(ThreadReader* thread_reader, int flags) {
    return thread_reader->poll_next_token(flags);
}
extern Token pallas_poll_previous_token(ThreadReader* thread_reader, int flags) {
    return thread_reader->poll_previous_token(flags);
}
extern bool pallas_move_to_next_token(ThreadReader* thread_reader, int flags) {
    return thread_reader->move_to_next_token(flags);
}
extern bool pallas_move_to_next_token_in_block(ThreadReader* thread_reader) {
    return pallas_move_to_next_token(thread_reader, PALLAS_READ_FLAG_NO_UNROLL);
}
extern bool pallas_move_to_previous_token(ThreadReader* thread_reader, int flags) {
    return thread_reader->move_to_previous_token(flags);
}
extern bool pallas_move_to_previous_token_in_block(ThreadReader* thread_reader) {
    return pallas_move_to_previous_token(thread_reader, PALLAS_READ_FLAG_NO_UNROLL);
}
extern Token pallas_get_next_token(ThreadReader* thread_reader, int flags) {
    return thread_reader->get_next_token(flags);
}
extern Token pallas_get_previous_token(ThreadReader* thread_reader, int flags) {
    return thread_reader->get_previous_token(flags);
}
extern void pallas_enter_block(ThreadReader* thread_reader) {
    thread_reader->enter_block();
}
extern void pallas_leave_block(ThreadReader* thread_reader) {
    thread_reader->leave_block();
}
extern bool pallas_exit_if_end_of_block(ThreadReader* thread_reader, int flags) {
    return thread_reader->exit_if_end_of_block(flags);
}
extern bool pallas_enter_if_start_of_block(ThreadReader* thread_reader, int flags) {
    return thread_reader->enter_if_start_of_block(flags);
}
extern Cursor pallas_create_checkpoint(ThreadReader* thread_reader) {
    return thread_reader->create_checkpoint();
}
extern void pallas_load_checkpoint(ThreadReader* thread_reader, Cursor* checkpoint) {
    thread_reader->load_checkpoint(checkpoint);
}

TokenOccurrence::~TokenOccurrence() {
    if (token == nullptr || occurrence == nullptr) {
        return;
    }
    if (token->type == TypeLoop) {
        auto& loopOccurrence = occurrence->loop_occurrence;
        if (loopOccurrence.full_loop) {
            delete[] loopOccurrence.full_loop;
        }
    }
    delete occurrence;
}

} /* namespace pallas */

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
