#include "pallas/pallas.h"
#include "pallas/utils/pallas_storage.h"
#include "pallas/linked_vector/pallas_subarray.h"

namespace pallas {

  namespace file_format {

    struct event_layout;
    struct sequence_layout;
    struct loop_layout;
    struct event_attributes;
    struct statistics_duration;
    struct subarray_duration;
    struct linked_vector_duration;
    struct statistics_timestamp;
    struct subarray_timestamp;
    struct linked_vector_timestamp;
    struct thread_summary_header;
    struct thread_summary;

    struct event_section_header {
      size_t nb_events = 0;

      off_t event_id_map_offset = -1;
      size_t event_id_map_size = -1;

      off_t event_attributes_offset = -1;
      size_t event_attributes_size = -1;

      off_t event_timestamps_offset = -1;
      size_t event_timestamps_size = -1;

      off_t event_list_offset = -1;
      size_t event_list_size = -1;
    };

    struct event_section {
      struct event_section_header header;

      /* An event_id map of size header.event_id_map_size
       * It is located at offset header.event_id_map_offset
       */
      uint32_t *event_id_map = nullptr;

      /* An array of event attributes.
       * Warning:
       *  - some events may have no attribute (if attributes_offset is -1)
       *  - the size of attributes varies
       */
      struct event_attributes* event_attributes = nullptr;

      /** An array of linkedVectors that contain timestamps
       *  There are header.nb_events linkedvectors
       * It is located at offset header.event_timestamps_offset
       */
      struct linked_vector_timestamp* event_timestamps = nullptr;

      /* An array of events. 
       * There are header.nb_events events of size sizeof(struct event_layout)
       * It is located at offset header.event_list_offset
       */
      struct event_layout *event_list = nullptr;
    };

    struct sequence_section_header {
      size_t nb_sequences;
  
      off_t sequence_id_map_offset = -1;
      size_t sequence_id_map_size = -1;

      off_t sequence_tokens_offset = -1;
      size_t sequence_tokens_size = -1;

      off_t sequence_durations_offset = -1;
      size_t sequence_durations_size = -1;

      off_t sequence_exclusive_durations_offset = -1;
      size_t sequence_exclusive_durations_size = -1;

      off_t sequence_timestamps_offset = -1;
      size_t sequence_timestamps_size = -1;

      off_t sequence_list_offset = -1;
      size_t sequence_list_size = -1;
    };

    struct sequence_section {
      struct sequence_section_header header;

      /* A sequence_id map of size header.sequence_id_mapsize
       * It is located at offset header.sequence_id_map_offset
       */
      uint32_t *sequence_id_map = nullptr;

      /** An array of sequence_tokens
       * There are header.nb_sequences entries
       * It is located at offset header.sequence_tokens_offset
       */
      struct sequence_tokens* sequence_tokens = nullptr;

      /** An array of linkedVectors that contain durations
       *  There are header.nb_sequences linkedvectors
       * It is located at offset header.sequence_duration_offset
       */
      struct linked_vector_duration* sequence_durations = nullptr;
            
      /** An array of linkedVectors that contain exclusive durations
       *  There are header.nb_sequences linkedvectors
       * It is located at offset header.sequence_exclusive_duration_offset
       */
      struct linked_vector_duration* sequence_exclusive_durations = nullptr;
            
      /** An array of linkedVectors that contain exclusive timestamps
       *  There are header.nb_sequences linkedvectors
       * It is located at offset header.sequence_timestamps_offset
       */
      struct linked_vector_timestamp* sequence_timestamps = nullptr;

      /* An array of sequences. 
       * There are header.nb_sequences sequences of size sizeof(struct sequence_layout)
       * It is located at offset header.sequence_list_offset
       */
      struct sequence_layout *sequence_list = nullptr;
    };

    struct loop_section_header {
      size_t nb_loops;
      off_t loop_list_offset = -1;
      size_t loop_list_size = -1;

      off_t loop_id_map_offset = -1;
      size_t loop_id_map_size = -1;
    };

    struct loop_section {
      struct loop_section_header header;

      /* A loop_id map of size header.loop_id_mapsize
       * It is located at offset header.loop_id_map_offset
       */
      uint32_t *loop_id_map = nullptr;

      /* An array of loop. 
       * There are header.nb_loop loops of size sizeof(struct loop_layout)
       * It is located at offset header.loop_list_offset
       */
      struct loop_layout *loop_list = nullptr;
    };
      

    struct thread_summary_header {
      ThreadId id;
      LocationGroupId archive_id;
      size_t nb_events;
      size_t nb_sequences;
      size_t nb_loops;
      TokenId sequence_root;
      pallas_timestamp_t first_timestamp;

      off_t event_section_offset;
      size_t event_section_size;

      off_t sequence_section_offset;
      size_t sequence_section_size;

      off_t loop_section_offset;
      size_t loop_section_size;
    };


    /**
     * The thread.summary file is structured as follows:
     * - a header describes the file content
     * - a event section describes event-related data
     * - a sequence section describes sequence-related data
     * - a loop section describes loop-related data
     */
    struct thread_summary {
      /* A header describes the file content. */
      struct thread_summary_header header;

      struct event_section event_section;
      struct sequence_section sequence_section;
      struct loop_section loop_section;

#ifdef __cplusplus
      /** Creates a thread_summary from a Thread */
      thread_summary(Thread* thread, const ParameterHandler* parameter_handler, bool load_thread);
#endif
    };



    struct event_layout {
      Token id;
      EventData data;
      size_t nb_occurrences;
      off_t attributes_offset;
      off_t timestamps_offset;
    };

    struct event_attributes {
      Token parent; // token of the event
      size_t size; // size of the buffer
      byte *buffer;
    };

    struct sequence_layout {
      Token id;
      enum SequenceType type;
      size_t size;
      off_t sequence_tokens_offset;
      off_t durations_offset;
      off_t exclusive_durations_offset;
      off_t timestamps_offset;
    };

    struct loop_layout {
      Token id;
      Token repeated_token;
      unsigned int nb_iterations;
      uint64_t nb_occurrences;
    };

    struct statistics_duration {
      int count;
      uint64_t min_duration;
      uint64_t max_duration;
      uint64_t mean_duration;
    };

    struct subarray_duration {
      int starting_index;
      int nb_values;
      StoragePolicy policy;
      size_t data_size; // size of the data in event_details.dat
      off_t data_offset; // offset of the data in event_details.dat 
      struct statistics_duration statistics;
    };

    struct linked_vector_duration {
      Token parent; // token of the event
      ValueDomain value_domain;
      //struct statistics_duration statistics;
      int nb_subarray;
      struct subarray_duration* subarray;
    };

    struct statistics_timestamp {
      uint64_t first_timestamp;
      uint64_t last_timestamp;
      int count;
    };        

    struct subarray_timestamp {
      int starting_index;
      int nb_values;
      StoragePolicy policy;
      size_t data_size; // size of the data in event_details.dat
      off_t data_offset; // offset of the data in event_details.dat
      struct statistics_timestamp statistics;
    };

    struct linked_vector_timestamp {
      Token parent; // token of the event
      ValueDomain value_domain;
      //todo: it would be usefull to have global statistics for a linked vector. Currently statistics are collected at the subarray granualiry
      //      struct statistics_timestamp staticstics;
      int nb_subarray;
      struct subarray_timestamp* subarray;
    };

    struct sequence_tokens {
      size_t nb_tokens;
      Token* tokens;
    };
        


    struct subarray_data {
      size_t size; // size of the buffer
      void* buffer;  // implementation-specific array that contains timestamps/durations
    };

    /**
     *  The events.details file is structured as follows
     */
    struct events_details {            
      // TODO: add a header ?
      struct subarray_data* subarrays;
    };

    /**
     *  The sequences.details file is structured as follows
     */
    struct sequences_details {
      // TODO: add a header ?
      struct subarray_data* subarrays;
    };


    thread_summary::thread_summary(Thread* thread, const ParameterHandler* parameter_handler, bool load_thread) {
      header.id = thread->id;
      header.archive_id = thread->archive->id;
      header.nb_events = thread->nb_events;
      header.nb_sequences = thread->nb_sequences;
      header.nb_loops =  thread->nb_loops;

      header.sequence_root = thread->sequence_root;
      header.first_timestamp = thread->first_timestamp;
    }

#if 0
    off_t current_offset = offsetof(thread_summary, event_list);

    header.event_list_offset = current_offset;
    header.event_list_size = sizeof(struct event_layout) * thread->nb_events;
    event_list = thread->events;
    current_offset += header.event_list_size;

    header.sequence_list_offset = current_offset;
    header.sequence_list_size = sizeof(struct sequence_layout) * thread->nb_sequences;
    sequence_list = thread->sequences;
    current_offset += header.sequence_list_size;

    header.loop_list_offset = current_offset;
    header.loop_list_size = sizeof(struct loop_layout) * thread->nb_loops;
    loop_list = thread->loops;
    current_offset += header.loop_list_size;

    header.event_id_map_offset = current_offset;
    header.event_id_map_size = sizeof(uint32_t) * thread->event_id_map.size();
    event_id_map = th->event_id_map.data();
    current_offset += header.event_id_map_size;

    header.sequence_id_map_offset = current_offset;
    header.sequence_id_map_size = sizeof(uint32_t) * thread->sequence_id_map.size();
    sequence_id_map = th->sequence_id_map.data();
    current_offset += header.sequence_id_map_size;

    header.loop_id_map_offset = current_offset;
    header.loop_id_map_size = sizeof(uint32_t) * thread->loop_id_map.size();
    loop_id_map = th->loop_id_map.data();
    current_offset += header.loop_id_map_size;

    header.event_attributes_offset = current_offset;
    // todo: iterate over events 
    size_t event_attributes_size;

    off_t event_timestamps_offset;
    size_t event_timestamps_size;

    off_t sequence_tokens_offset;
    size_t sequence_tokens_size;

    off_t sequence_durations_offset;
    size_t sequence_durations_size;

    off_t sequence_exclusive_durations_offset;
    size_t sequence_exclusive_durations_size;

    off_t sequence_timestamps_offset;
    size_t sequence_timestamps_size;
  }
#endif

  
  /** Write an id_map
   * Returns the offset after writing
   */
  static off_t store_id_map(File& thread_summary_file, 
    std::vector<uint32_t> id_map,
      off_t map_offset) {

    size_t map_size = id_map.size() * sizeof(uint32_t);
    size_t written_data = 0;
    if (map_size > 0) {
      written_data = map_size * sizeof(uint32_t);
      thread_summary_file.write(id_map.data(), written_data, 1, map_offset);
    }
    off_t current_offset = map_offset + written_data;
    pallas_assert(thread_summary_file.offset() == current_offset);
    return current_offset;
  }

  static off_t store_event_attributes(File& thread_summary_file,
				      Thread* thread,
				      struct event_layout* events,
				      off_t attributes_offset) {

    for(int i = 0; i<thread->nb_events; i++) {
      struct Event*e = &thread->events[i];

      struct event_attributes attr;

      // Copy the event attribute
      attr.parent = Token(pallas::TypeEvent, e->id);
      attr.size = e->attribute_pos;
      if (attr.size == 0) {
	attr.buffer = nullptr;
      } else {
	attr.buffer = e->attribute_buffer;	
      }

      // size of the whole event_attributes structure
      size_t attr_size = offsetof(event_attributes, buffer) + attr.size;

      events[i].attributes_offset = attributes_offset;

      // write the first part of the structure to disk
      thread_summary_file.write(&attr, offsetof(event_attributes, buffer), 1, attributes_offset);
      attributes_offset += offsetof(event_attributes, buffer);
      pallas_assert(thread_summary_file.offset() == attributes_offset);

      // write the buffer to disk
      if(attr.size > 0) {
	thread_summary_file.write(attr.buffer, attr.size, 1, attributes_offset);
	attributes_offset += attr.size;
	pallas_assert(thread_summary_file.offset() == attributes_offset);
      }
    }
    return attributes_offset;
  }

  off_t store_linked_vector(File& thread_summary_file,
                        File& details_file,
                        Token parent,
                        LinkedVectorBase* lv,
                        off_t lv_offset) {
    off_t current_offset = lv_offset;

    if(lv->domain() == ValueDomain::Timestamp) {
        struct linked_vector_timestamp linked_vector;
        linked_vector.parent = parent;
        linked_vector.value_domain = lv->domain();
        linked_vector.nb_subarray = lv->subarray_count();
        // write the first part of the linkedvector data structure
        thread_summary_file.write(&linked_vector, offsetof(linked_vector_timestamp, subarray), 1, current_offset);
        current_offset += offsetof(linked_vector_timestamp, subarray);
    } else {
        struct linked_vector_duration linked_vector;
        linked_vector.parent = parent;
        linked_vector.value_domain = lv->domain();
        linked_vector.nb_subarray = lv->subarray_count();
        // write the first part of the linkedvector data structure
        thread_summary_file.write(&linked_vector, offsetof(linked_vector_duration, subarray), 1, current_offset);
        current_offset += offsetof(linked_vector_duration, subarray);        
    }
    pallas_assert(thread_summary_file.offset() == current_offset);

    // Write the subarrays once at a time
    for(int i=0; i<lv->subarray_count(); i++) {
        SubArrayBase* subarray = lv->get_subarray(i);

        if(lv->domain() == ValueDomain::Timestamp) {
            struct subarray_timestamp sub;
	    
            sub.starting_index = subarray->starting_index();
            sub.nb_values = subarray->size();
            sub.policy = subarray->storage_policy();

	        subarray->write_details(&details_file, &sub.data_size, &sub.data_offset);

            sub.statistics.first_timestamp = subarray->subarray_stats().first_timestamp();
            sub.statistics.last_timestamp = subarray->subarray_stats().last_timestamp();
            sub.statistics.count = subarray->size();

	       thread_summary_file.write(&sub, sizeof(sub), 1, current_offset);
	        current_offset += sizeof(sub);
        } else {
            struct subarray_duration sub;
	    
            sub.starting_index = subarray->starting_index();
            sub.nb_values = subarray->size();
            sub.policy = subarray->storage_policy();

	        subarray->write_details(&details_file, &sub.data_size, &sub.data_offset);

            sub.statistics.min_duration = subarray->subarray_stats().min_duration();
            sub.statistics.max_duration = subarray->subarray_stats().max_duration();
            sub.statistics.mean_duration = subarray->subarray_stats().mean_duration();
            sub.statistics.count = subarray->size();

	        thread_summary_file.write(&sub, sizeof(sub), 1, current_offset);
	        current_offset += sizeof(sub);
        }
	    pallas_assert(thread_summary_file.offset() == current_offset);
    }
    pallas_assert(thread_summary_file.offset() == current_offset);

    return current_offset;
}

  static off_t store_event_timestamps(File& thread_summary_file,
                        File& event_details_file,
				      Thread* thread,
				      struct event_layout* events,
				      off_t timestamps_offset) {

    // Copy the events LinkedVectors
    off_t  current_offset = timestamps_offset;   

    for(int i = 0; i<thread->nb_events; i++) {
      pallas_assert(thread_summary_file.offset() == current_offset);

      struct event_layout* el = &events[i];
      struct Event* e = &thread->events[i];

      el->timestamps_offset = current_offset;

      current_offset = store_linked_vector(thread_summary_file, event_details_file, 
                                    el->id, e->timestamps, el->timestamps_offset);

      pallas_assert(thread_summary_file.offset() == current_offset);
    }
    return current_offset;
  }

  static off_t store_event_list(File& thread_summary_file,
				Thread* thread,
				struct event_layout* events,
				off_t list_offset) {
    // TODO: we could check if all the sequences fields are filled
    thread_summary_file.write(events, sizeof(struct event_layout), thread->nb_events, list_offset);
    off_t current_offset = list_offset + sizeof(struct event_layout) * thread->nb_events;
    pallas_assert(thread_summary_file.offset() == current_offset);
    return current_offset;

}
  
  /** Updates the thread_summary header and writes the event section to disk
   */
  static void storeEvents(struct thread_summary *ts,
			  File& thread_summary_file,
			  File& event_details_file,
			  Thread* thread,
			  const ParameterHandler* parameter_handler,
			  bool load_thread) {

    struct event_section* es = &ts->event_section;
    ts->header.event_section_offset = offsetof(thread_summary, event_section);
    off_t event_section_offset = ts->header.event_section_offset;

    es->header.nb_events = thread->nb_events;
    es->event_list = (struct event_layout*) malloc(sizeof(struct event_layout) * es->header.nb_events);

    // write the event_id_map to disk
    es->header.event_id_map_offset = event_section_offset + offsetof(event_section, event_id_map);
    off_t current_offset = store_id_map(thread_summary_file,
				  thread->event_id_map,
				  es->header.event_id_map_offset);
    es->header.event_id_map_size = current_offset - es->header.event_id_map_offset;
    pallas_assert(thread_summary_file.offset() == current_offset);

    // write the attributes to disk
    es->header.event_attributes_offset = es->header.event_id_map_offset + es->header.event_id_map_size;
    pallas_assert(current_offset ==  es->header.event_attributes_offset);
    current_offset = store_event_attributes(thread_summary_file,
					    thread,
					    es->event_list,
					    es->header.event_attributes_offset);
    es->header.event_attributes_size = current_offset - es->header.event_attributes_size;
    pallas_assert(thread_summary_file.offset() == current_offset);

    // write the timestamps to disk
    es->header.event_timestamps_offset = es->header.event_attributes_offset + es->header.event_attributes_size;
    pallas_assert(current_offset ==  es->header.event_timestamps_offset);
    current_offset = store_event_timestamps(thread_summary_file,
                        event_details_file,
                        thread,
                        es->event_list,
					    es->header.event_timestamps_offset);
    es->header.event_timestamps_size = current_offset - es->header.event_timestamps_offset;
    pallas_assert(thread_summary_file.offset() == current_offset);

    // write the events to disk
    es->header.event_list_offset = es->header.event_timestamps_offset + es->header.event_timestamps_size;
    pallas_assert(current_offset ==  es->header.event_list_offset);
    current_offset = store_event_list(thread_summary_file,
				      thread,
                      es->event_list,
                      es->header.event_list_offset);
    es->header.event_list_size = current_offset - es->header.event_list_offset;
    pallas_assert(thread_summary_file.offset() == current_offset);

    // Now, each event_layout should be complete and the header info related to events should be filled
    pallas_assert(es->header.event_list_offset > -1);
    pallas_assert(es->header.event_list_size >= -1);
    pallas_assert(es->header.event_id_map_offset >= -1);
    pallas_assert(es->header.event_id_map_size >= -1);
    pallas_assert(es->header.event_attributes_offset >= -1);
    pallas_assert(es->header.event_attributes_size >= -1);
    pallas_assert(es->header.event_timestamps_offset >= -1);
    pallas_assert(es->header.event_timestamps_size >= -1);

    // Now we can write the event header to disk
    ts->header.event_section_offset = event_section_offset;
    ts->header.event_section_size = current_offset;
    thread_summary_file.write(&es->header, sizeof(es->header), 1, event_section_offset);

    delete[] es->event_list;
  }

  off_t store_tokens(struct thread_summary *ts,
			    File& thread_summary_file,
                Thread* thread,
                off_t tokens_offset) {
    struct sequence_section* ss = &ts->sequence_section;    

    off_t current_offset = tokens_offset;
    for(int i=0; i<thread->nb_sequences; i++) {
      struct sequence_layout *sl = &ss->sequence_list[i];
      Sequence* s = &thread->sequences[i];
      struct sequence_tokens sequence_tokens;

      sequence_tokens.nb_tokens = sl->size;
      sequence_tokens.tokens = s->tokens.data();

      sl->sequence_tokens_offset = current_offset;
      size_t sequence_tokens_size = sizeof(sequence_tokens.nb_tokens) + sequence_tokens.nb_tokens*sizeof(sequence_tokens.tokens[0]);

      thread_summary_file.write(&sequence_tokens.nb_tokens, sizeof(sequence_tokens.nb_tokens), 1, current_offset);      
      thread_summary_file.write(sequence_tokens.tokens, sequence_tokens.nb_tokens*sizeof(sequence_tokens.tokens[0]), 1, current_offset + sizeof(sequence_tokens.nb_tokens));

      current_offset += sequence_tokens_size;
      pallas_assert(thread_summary_file.offset() == current_offset);
    }
    return current_offset;
  }




  off_t store_durations(struct thread_summary* ts,
                                    File& thread_summary_file,
                                    File& sequence_details_file,
                                    Thread* thread,
                                    off_t sequence_durations_offset) {
    struct sequence_section* ss = &ts->sequence_section;    

    off_t  current_offset = sequence_durations_offset;
    for(int i = 0; i<thread->nb_sequences; i++) {
        pallas_assert(thread_summary_file.offset() == current_offset);
        struct sequence_layout* sl = &ss->sequence_list[i];
        struct linked_vector_timestamp linked_vector;
        struct Sequence* s = &thread->sequences[i];

        sl->durations_offset = current_offset;
        
        current_offset = store_linked_vector(thread_summary_file, sequence_details_file, 
                                                sl->id, s->durations, sl->durations_offset);
        pallas_assert(thread_summary_file.offset() == current_offset);
    }
    return current_offset;
}

off_t store_exclusive_durations(struct thread_summary* ts,
                                    File& thread_summary_file,
                                    File& sequence_details_file,
                                    Thread* thread,
                                    off_t sequence_exclusive_durations_offset) {
    struct sequence_section* ss = &ts->sequence_section;
    
    off_t  current_offset = sequence_exclusive_durations_offset;
    for(int i = 0; i<thread->nb_sequences; i++) {
        pallas_assert(thread_summary_file.offset() == current_offset);
        struct sequence_layout* sl = &ss->sequence_list[i];
        struct linked_vector_timestamp linked_vector;
        struct Sequence* s = &thread->sequences[i];

        sl->exclusive_durations_offset = current_offset;
        
        current_offset = store_linked_vector(thread_summary_file, sequence_details_file, 
                                                sl->id, s->exclusive_durations, sl->exclusive_durations_offset);
        pallas_assert(thread_summary_file.offset() == current_offset);
    }

    return current_offset;
}

  static off_t store_sequence_timestamps(File& thread_summary_file,
                        File& sequence_details_file,
				      Thread* thread,
				      struct sequence_layout* sequences,
				      off_t timestamps_offset) {

    // Copy the events LinkedVectors
    off_t  current_offset = timestamps_offset;   

    for(int i = 0; i<thread->nb_events; i++) {
      pallas_assert(thread_summary_file.offset() == current_offset);

      struct sequence_layout* sl = &sequences[i];
      struct Sequence* s = &thread->sequences[i];

      sl->timestamps_offset = current_offset;

      current_offset = store_linked_vector(thread_summary_file, sequence_details_file, 
                                    sl->id, s->timestamps, sl->timestamps_offset);

      pallas_assert(thread_summary_file.offset() == current_offset);
    }
    return current_offset;
  }

  static off_t store_sequence_list(File& thread_summary_file,
				      Thread* thread,
				      struct sequence_layout* sequences,
				      off_t list_offset) {
    // TODO: we could check if all the sequences fields are filled

    thread_summary_file.write(sequences, sizeof(struct sequence_layout), thread->nb_sequences, list_offset);
    off_t current_offset = list_offset + sizeof(struct sequence_layout) * thread->nb_sequences;
    pallas_assert(thread_summary_file.offset() == current_offset);
    return current_offset;
}

/** Updates the thread_summary header and writes the sequence_section to disk
   */
  static void storeSequences(struct thread_summary *ts,
			    File& thread_summary_file,
			    File& sequence_details_file,
			    Thread* thread,
			    const ParameterHandler* parameter_handler,
			    bool load_thread) {
    
    struct sequence_section* ss = &ts->sequence_section;
    ts->header.sequence_section_offset = ts->header.event_section_size + ts->header.event_section_size;
    off_t sequence_section_offset = ts->header.sequence_section_offset;
    off_t current_offset = sequence_section_offset + offsetof(sequence_section, sequence_id_map);

    ss->header.nb_sequences = thread->nb_sequences;
    ss->sequence_list = new struct sequence_layout[thread->nb_sequences];

    // We can't write the sequence list right now because we need to compute a few offsets
    for(int i=0; i<thread->nb_sequences; i++) {
      struct sequence_layout *sl = &ss->sequence_list[i];
      Sequence* s = &thread->sequences[i];

      sl->id = s->id;
      sl->type = s->type;
      sl->size = s->size();

      sl->sequence_tokens_offset = -1;
      sl->durations_offset = -1;
      sl->exclusive_durations_offset = -1;
      sl->timestamps_offset = -1;
    }

    // Write the sequence_id_map
    ss->header.sequence_id_map_offset = ts->header.sequence_section_offset + offsetof(sequence_section, sequence_id_map);
    current_offset = store_id_map(thread_summary_file,
				  thread->sequence_id_map,
				  ss->header.sequence_id_map_offset);
    ss->header.sequence_id_map_size = current_offset - ss->header.sequence_id_map_offset;
    pallas_assert(thread_summary_file.offset() == current_offset);

    // Write the tokens
    ss->header.sequence_tokens_offset = ss->header.sequence_id_map_size + ss->header.sequence_id_map_offset;
    current_offset = store_tokens(ts, thread_summary_file, thread,
                                ss->header.sequence_tokens_offset);
    ss->header.sequence_tokens_size = current_offset - ss->header.sequence_tokens_offset;
    pallas_assert(thread_summary_file.offset() == current_offset);


    // Write the sequence durations
    ss->header.sequence_durations_offset = ss->header.sequence_tokens_offset + ss->header.sequence_tokens_size;
    current_offset = store_durations(ts,
                                    thread_summary_file,
                                    sequence_details_file,
                                    thread,
                                    ss->header.sequence_durations_offset);
    ss->header.sequence_durations_size = current_offset - ss->header.sequence_durations_offset;
    pallas_assert(thread_summary_file.offset() == current_offset);

    // Write the sequence exclusive durations
    ss->header.sequence_exclusive_durations_offset = ss->header.sequence_durations_offset + ss->header.sequence_durations_size;
    current_offset = store_exclusive_durations(ts,
                                    thread_summary_file,
                                    sequence_details_file,
                                    thread,
                                    ss->header.sequence_exclusive_durations_offset);
    ss->header.sequence_exclusive_durations_size = current_offset - ss->header.sequence_exclusive_durations_offset;
    pallas_assert(thread_summary_file.offset() == current_offset);

    // Write the sequence timestamps
    ss->header.sequence_timestamps_offset = ss->header.sequence_exclusive_durations_offset + ss->header.sequence_exclusive_durations_size;
    current_offset = store_sequence_timestamps(
                                    thread_summary_file,
                                    sequence_details_file,
                                    thread,
                                    ss->sequence_list,
                                    ss->header.sequence_timestamps_offset);
    ss->header.sequence_timestamps_size = current_offset - ss->header.sequence_timestamps_offset;
    pallas_assert(thread_summary_file.offset() == current_offset);

    ss->header.sequence_list_offset = ss->header.sequence_timestamps_offset + ss->header.sequence_timestamps_size;
    current_offset = store_sequence_list(thread_summary_file,
                                    thread,
                                    ss->sequence_list,
                                    ss->header.sequence_list_offset);
    ss->header.sequence_list_size = current_offset - ss->header.sequence_list_offset;
    pallas_assert(thread_summary_file.offset() == current_offset);

    // Now, each sequence_layout should be complete and the header info related to sequences should be filled
    pallas_assert(ss->header.sequence_id_map_offset > -1);
    pallas_assert(ss->header.sequence_id_map_size > -1);
    pallas_assert(ss->header.sequence_tokens_offset > -1);
    pallas_assert(ss->header.sequence_tokens_size > -1);
    pallas_assert(ss->header.sequence_durations_offset > -1);
    pallas_assert(ss->header.sequence_durations_size > -1);
    pallas_assert(ss->header.sequence_exclusive_durations_offset > -1);
    pallas_assert(ss->header.sequence_exclusive_durations_size > -1);
    pallas_assert(ss->header.sequence_timestamps_offset > -1);
    pallas_assert(ss->header.sequence_timestamps_size > -1);
    pallas_assert(ss->header.sequence_list_offset > -1);
    pallas_assert(ss->header.sequence_list_size > -1);
    

    // Now we can write the event header to disk
    ts->header.sequence_section_offset = sequence_section_offset;
    ts->header.sequence_section_size = current_offset;
    thread_summary_file.write(&ss->header, sizeof(ss->header), 1, sequence_section_offset);
 
    delete[] ss->sequence_list;
  }

  static off_t store_loop_list(File& thread_summary_file,
				      Thread* thread,
				      struct loop_layout* loops,
				      off_t list_offset) {
    // TODO: we could check if all the sequences fields are filled

    thread_summary_file.write(loops, sizeof(struct loop_layout), thread->nb_loops, list_offset);
    off_t current_offset = list_offset + sizeof(struct loop_layout) * thread->nb_loops;
    pallas_assert(thread_summary_file.offset() == current_offset);
    return current_offset;
}

  /** Updates the thread_summary header and writes the sequence_section to disk
   */
  static void storeLoops(struct thread_summary *ts,
			    File& thread_summary_file,
			    Thread* thread,
			    const ParameterHandler* parameter_handler,
			    bool load_thread) {

    ts->header.loop_section_offset = ts->header.sequence_section_size + ts->header.sequence_section_size;
    off_t loop_section_offset = ts->header.loop_section_offset;
    struct loop_section * ls = &ts->loop_section;
    struct loop_section_header *header = &ls->header;
    header->nb_loops = thread->nb_loops;
    ls->loop_list = new loop_layout[thread->nb_loops];

    for(int i = 0; i<thread->nb_loops; i++) {
        ls->loop_list[i].id = thread->loops[i].self_id;
        ls->loop_list[i].repeated_token = thread->loops[i].repeated_token;
        ls->loop_list[i].nb_iterations = thread->loops[i].nb_iterations;
        ls->loop_list[i].nb_occurrences = thread->loops[i].nb_occurrences;
    }

    // Write the loop_id_map
    header->loop_id_map_offset = ts->header.loop_section_offset + offsetof(loop_section, loop_id_map);
    off_t current_offset = store_id_map(thread_summary_file,
				  thread->loop_id_map,
				  header->loop_id_map_offset);
    header->loop_id_map_size = current_offset - header->loop_id_map_offset;
    pallas_assert(thread_summary_file.offset() == current_offset);

    // Write the loop list
    header->loop_list_offset = header->loop_id_map_offset + header->loop_id_map_size;
    current_offset = store_loop_list(thread_summary_file,
                                    thread,
                                    ls->loop_list,
                                    header->loop_list_offset);
    header->loop_list_size = current_offset - header->loop_list_offset;
    pallas_assert(thread_summary_file.offset() == current_offset);


    // Now, each loop_layout should be complete and the header info related to loop should be filled
    pallas_assert(header->loop_list_offset > -1);
    pallas_assert(header->loop_list_size > -1);
    pallas_assert(header->loop_id_map_offset > -1);
    pallas_assert(header->loop_id_map_size > -1); 

    // Now we can write the loop header to disk
    ts->header.loop_section_offset = loop_section_offset;
    ts->header.loop_section_size = current_offset;
    thread_summary_file.write(&ls->header, sizeof(ls->header), 1, loop_section_offset);

    delete[] ls->loop_list;
}

void storeThread(File& thread_summary_file,
		   File& event_details_file,
		   File& sequence_details_file,
		   Thread* thread,
		   const ParameterHandler* parameter_handler,
		   bool load_thread) {

    if (!thread_summary_file.is_open()) { thread_summary_file.open("w"); }

    pallas_log(pallas::DebugLevel::Verbose, "\tThread %u {.nb_events=%lu, .nb_sequences=%lu, .nb_loops=%lu}\n",
	       thread->id, thread->nb_events, thread->nb_sequences, thread->nb_loops);

    thread_summary_file.begin_block(__func__);
    struct thread_summary file_layout(thread, parameter_handler, load_thread);
    struct thread_summary_header *header = &file_layout.header;

    // warning: some fields are not yet set
    thread_summary_file.write(header, sizeof(struct thread_summary_header), 1, 0); // todo: move at the end

    storeEvents(&file_layout, thread_summary_file, event_details_file, thread, parameter_handler, load_thread);
    storeSequences(&file_layout, thread_summary_file, sequence_details_file, thread, parameter_handler, load_thread);
    storeLoops(&file_layout, thread_summary_file, thread, parameter_handler, load_thread);

#if 0
    thread_summary_file->write(file_format.sequence_list, header->sequence_list_size, 1, header->sequence_list_offset);
    thread_summary_file->write(file_format.loop_list, header->loop_list_size, 1, header->loop_list_offset);

    thread_summary_file->write(file_format.event_id_map, header->event_id_map_size, 1, header->event_id_map_offset);
    thread_summary_file->write(file_format.sequence_id_map,  header->sequence_id_map_size, 1, header->sequence_id_map_offset);
    thread_summary_file->write(file_format.loop_id_map, header->loop_id_map_size, 1, header->loop_id_map_offset);

    thread_summary_file->write(file_format.event_attributes, header->loop_list_size, 1, header->loop_list_offset);
    thread_summary_file->write(file_format.event_timestamps, event_timestamps_size, 1, event_timestamps_offset);
    thread_summary_file->write(file_format.sequence_tokens, sequence_tokens_size, 1, sequence_tokens_offset);
    thread_summary_file->write(file_format.sequence_durations, sequence_durations_size, 1, sequence_durations_offset);
    thread_summary_file->write(file_format.sequence_exclusive_durations, sequence_exclusive_durations_size, 1, sequence_exclusive_durations_offset);
    thread_summary_file->write(file_format.sequence_timestamps, sequence_timestamps_size, 1, sequence_timestamps_offset);
#endif

    thread_summary_file.end_block(__func__);
  }

};
};


/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
