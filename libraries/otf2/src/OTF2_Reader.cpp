#include <stdio.h>
#include <stdlib.h>

#include "pallas/pallas.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_storage.h"
#include "pallas/pallas_record.h"
#include "otf2/OTF2_Reader.h"
#include "otf2/otf2.h"

static void init_OTF2_GlobalDefReader(OTF2_Reader* reader, OTF2_GlobalDefReader* global_def_reader) {
  memset(global_def_reader, 0, sizeof(OTF2_GlobalDefReader));
  global_def_reader->archive = reader->archive;
}

static void init_OTF2_GlobalEvtReader(OTF2_Reader* reader, OTF2_GlobalEvtReader* global_evt_reader) {
  memset(global_evt_reader, 0, sizeof(OTF2_GlobalEvtReader));
  global_evt_reader->otf2_reader = reader;
}

OTF2_Reader* OTF2_Reader_Open(const char* anchorFilePath) {

  OTF2_Reader* reader = (OTF2_Reader*)malloc(sizeof(OTF2_Reader));
  memset(reader, 0, sizeof(OTF2_Reader));

  pallas::GlobalArchive* archive = pallas_open_trace(anchorFilePath);
  reader->archive = reinterpret_cast<struct GlobalArchive*>(archive);

  for (int i = 0; i < archive->nb_archives; i++) {
    reader->nb_locations += archive->archive_list[i]->nb_threads;
    int nb_threads = archive->archive_list[i]->nb_threads;
    for(int j = 0; j<nb_threads; j++) {
      pallas::Thread* t = archive->archive_list[i]->threads[i];
    }
  }
  reader->locations = new PALLAS(ThreadId)[reader->nb_locations];
  reader->evt_readers = (struct OTF2_EvtReader_struct*) calloc(reader->nb_locations, sizeof(struct OTF2_EvtReader_struct));
  reader->def_readers = (struct OTF2_DefReader_struct*) calloc(reader->nb_locations, sizeof(struct OTF2_DefReader_struct));
  reader->thread_readers = (pallas::ThreadReader**) calloc(reader->nb_locations, sizeof(pallas::ThreadReader*));

  for(int i = 0; i< reader->nb_locations; i++) {
    // reader->locations[i] =   archive->locations[i].id;
    reader->evt_readers[i].location = OTF2_UNDEFINED_LOCATION;
    reader->def_readers[i].location = OTF2_UNDEFINED_LOCATION;
  }


  reader->selected_locations = (int*) calloc(reader->nb_locations, sizeof(int));
  reader->global_def_reader = (OTF2_GlobalDefReader*)malloc(sizeof(OTF2_GlobalDefReader));
  init_OTF2_GlobalDefReader(reader, reader->global_def_reader);
  reader->global_evt_reader = (OTF2_GlobalEvtReader*)malloc(sizeof(OTF2_GlobalEvtReader));
  init_OTF2_GlobalEvtReader(reader, reader->global_evt_reader);
  
  return reader;
}

OTF2_ErrorCode OTF2_Reader_Close(OTF2_Reader* reader) {
  // TODO: free memory
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_SetHint(OTF2_Reader* reader, OTF2_Hint hint, void* value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_SetCollectiveCallbacks(OTF2_Reader* reader,
                                                  const OTF2_CollectiveCallbacks* collectiveCallbacks,
                                                  void* collectiveData,
                                                  OTF2_CollectiveContext* globalCommContext,
                                                  OTF2_CollectiveContext* localCommContext) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_SetSerialCollectiveCallbacks(OTF2_Reader* reader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_SetLockingCallbacks(OTF2_Reader* reader,
                                               const OTF2_LockingCallbacks* lockingCallbacks,
                                               void* lockingData) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_RegisterEvtCallbacks(OTF2_Reader* reader,
                                                OTF2_EvtReader* evtReader,
                                                const OTF2_EvtReaderCallbacks* callbacks,
                                                void* userData) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_RegisterGlobalEvtCallbacks(OTF2_Reader* reader,
                                                      OTF2_GlobalEvtReader* evtReader,
                                                      const OTF2_GlobalEvtReaderCallbacks* callbacks,
                                                      void* userData) {
  memcpy(&evtReader->callbacks, callbacks, sizeof(OTF2_GlobalEvtReaderCallbacks));
  evtReader->user_data = userData;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_RegisterDefCallbacks(OTF2_Reader* reader,
                                                OTF2_DefReader* defReader,
                                                const OTF2_DefReaderCallbacks* callbacks,
                                                void* userData) {
  memcpy(&defReader->callbacks, callbacks, sizeof(OTF2_DefReaderCallbacks));
  defReader->user_data = userData;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_RegisterGlobalDefCallbacks(OTF2_Reader* reader,
                                                      OTF2_GlobalDefReader* defReader,
                                                      const OTF2_GlobalDefReaderCallbacks* callbacks,
                                                      void* userData) {
  memcpy(&defReader->callbacks, callbacks, sizeof(OTF2_GlobalDefReaderCallbacks));
  defReader->user_data = userData;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_RegisterSnapCallbacks(OTF2_Reader* reader,
                                                 OTF2_SnapReader* snapReader,
                                                 const OTF2_SnapReaderCallbacks* callbacks,
                                                 void* userData) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_RegisterGlobalSnapCallbacks(OTF2_Reader* reader,
                                                       OTF2_GlobalSnapReader* evtReader,
                                                       const OTF2_GlobalSnapReaderCallbacks* callbacks,
                                                       void* userData) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_RegisterMarkerCallbacks(OTF2_Reader* reader,
                                                   OTF2_MarkerReader* markerReader,
                                                   const OTF2_MarkerReaderCallbacks* callbacks,
                                                   void* userData) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadLocalEvents(OTF2_Reader* reader,
                                           OTF2_EvtReader* evtReader,
                                           uint64_t eventsToRead,
                                           uint64_t* eventsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadAllLocalEvents(OTF2_Reader* reader, OTF2_EvtReader* evtReader, uint64_t* eventsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadLocalEventsBackward(OTF2_Reader* reader,
                                                   OTF2_EvtReader* evtReader,
                                                   uint64_t eventsToRead,
                                                   uint64_t* eventsRead) {
  NOT_IMPLEMENTED;
}

/* return the threadRead that contains the next event to process (ie. the one with the minimum
 * timestamp)
 */
pallas::ThreadReader* _get_next_global_event(OTF2_Reader* reader, OTF2_GlobalEvtReader* evtReader) {
  pallas::ThreadReader* retval = nullptr;

  pallas_timestamp_t min_timestamp = ULONG_MAX;

  for(int i = 0; i<reader->nb_locations; i++) {
    if(reader->thread_readers[i]) {
      pallas::ThreadReader *tr = reader->thread_readers[i];

      if(tr->isEndOfTrace())
	continue;
      if(tr->currentState.currentFrame->current_timestamp < min_timestamp) {
	min_timestamp = tr->currentState.currentFrame->current_timestamp;
	retval = tr;
      }
    }
  }  
  
  return retval;
}

OTF2_ErrorCode OTF2_Reader_ReadGlobalEvent(OTF2_Reader* reader, OTF2_GlobalEvtReader* evtReader) {
  pallas::ThreadReader* thread_reader = _get_next_global_event(reader, evtReader);

  auto token = thread_reader->pollCurToken();
  if (token.type == pallas::TypeEvent) {
    const pallas::EventOccurence e = thread_reader->getEventOccurence(token, thread_reader->currentState.currentFrame->tokenCount[token]);

    pallas::Record event_type = e.event->record;
    switch(event_type) {
    case pallas::PALLAS_EVENT_ENTER:
      if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_Enter_callback) {
	AttributeList* attribute_list;
	pallas_timestamp_t time;
	pallas::RegionRef region_ref;

	pallas_read_enter(thread_reader,
			  (PALLAS(AttributeList)**)&attribute_list,
			  &time,
			  &region_ref);
	
	evtReader->callbacks.OTF2_GlobalEvtReaderCallback_Enter_callback(thread_reader->thread_trace->id,
									 time,
									 evtReader->user_data,
									 attribute_list,
									 region_ref);
	
      }
      break;
    case pallas::PALLAS_EVENT_LEAVE:
      if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_Leave_callback) {
	AttributeList* attribute_list;
	pallas_timestamp_t time;
	pallas::RegionRef region_ref;

	pallas_read_leave(thread_reader,
			  (PALLAS(AttributeList)**) &attribute_list,
			  &time,
			  &region_ref);
	
	evtReader->callbacks.OTF2_GlobalEvtReaderCallback_Leave_callback(thread_reader->thread_trace->id,
									 time,
									 evtReader->user_data,
									 attribute_list,
									 region_ref);
	break;
      case pallas::PALLAS_EVENT_MPI_SEND:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiSend_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint32_t receiver;
	  uint32_t communicator;
	  uint32_t msgTag;
	  uint64_t msgLength;
	  pallas_read_mpi_send(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time,
			       & receiver,
			       & communicator,
			       & msgTag,
			       & msgLength);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiSend_callback(thread_reader->thread_trace->id,
									     time,
									     evtReader->user_data,
									     attribute_list,
									     receiver,
									     communicator,
									     msgTag,
									     msgLength);
	}	
	break;

      case pallas::PALLAS_EVENT_MPI_ISEND:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiIsend_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint32_t receiver;
	  uint32_t communicator;
	  uint32_t msgTag;
	  uint64_t msgLength;
	  uint64_t requestID;
	  pallas_read_mpi_isend(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time,
				&receiver,
				& communicator,
				& msgTag,
				& msgLength,
				& requestID);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiIsend_callback(thread_reader->thread_trace->id,
									      time,
									      evtReader->user_data,
									      attribute_list,
									      receiver,
									      communicator,
									      msgTag,
									      msgLength,
									      requestID);
	}	
	break;

      case pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiIsendComplete_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint64_t requestID;
	  pallas_read_mpi_isend_complete(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time, &requestID);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiIsendComplete_callback(thread_reader->thread_trace->id,
										      time,
										      evtReader->user_data,
										      attribute_list,
										      requestID);
	}	
	break;

      case pallas::PALLAS_EVENT_MPI_IRECV_REQUEST:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiIrecvRequest_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint64_t requestID;
	  pallas_read_mpi_irecv_request(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time, &requestID);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiIrecvRequest_callback(thread_reader->thread_trace->id,
										     time,
										     evtReader->user_data,
										     attribute_list,
										     requestID);
	}	
	break;

      case pallas::PALLAS_EVENT_MPI_RECV:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiRecv_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint32_t sender;
	  uint32_t communicator;
	  uint32_t msgTag;
	  uint64_t msgLength;
	  pallas_read_mpi_recv(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time,
			       & sender,
			       & communicator,
			       & msgTag,
			       & msgLength);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiRecv_callback(thread_reader->thread_trace->id,
									     time,
									     evtReader->user_data,
									     attribute_list,
									     sender,
									     communicator,
									     msgTag,
									      msgLength);
	}	
	break;

      case pallas::PALLAS_EVENT_MPI_IRECV:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiIrecv_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint32_t sender;
	  uint32_t communicator;
	  uint32_t msgTag;
	  uint64_t msgLength;
	  uint64_t requestID;
	  pallas_read_mpi_irecv(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time,
				& sender,
				& communicator,
				& msgTag,
				& msgLength,
				& requestID);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiIrecv_callback(thread_reader->thread_trace->id,
									      time,
									      evtReader->user_data,
									      attribute_list,
									      sender,
									      communicator,
									      msgTag,
									      msgLength,
									      requestID);
	}	
	break;

      case pallas::PALLAS_EVENT_MPI_REQUEST_TEST:
	// not implemented
	break;

      case pallas::PALLAS_EVENT_MPI_REQUEST_CANCELLED:
	// not implemented
	break;

      case pallas::PALLAS_EVENT_MPI_COLLECTIVE_BEGIN:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiCollectiveBegin_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_mpi_collective_begin(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiCollectiveBegin_callback(thread_reader->thread_trace->id,
											time,
											evtReader->user_data,
											attribute_list);
	}	
	break;

      case pallas::PALLAS_EVENT_MPI_COLLECTIVE_END:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiCollectiveEnd_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint32_t collectiveOp;
	  uint32_t communicator;
	  uint32_t root;
	  uint64_t sizeSent;
	  uint64_t sizeReceived;
	  pallas_read_mpi_collective_end(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time,
					 & collectiveOp,
					 & communicator,
					 & root,
					 & sizeSent,
					 & sizeReceived);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_MpiCollectiveEnd_callback(thread_reader->thread_trace->id,
										      time,
										      evtReader->user_data,
										      attribute_list,
										      collectiveOp,
										      communicator,
										      root,
										      sizeSent,
										      sizeReceived);
	}	
	break;

      case pallas::PALLAS_EVENT_OMP_FORK:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpFork_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint32_t numberOfRequestedThreads;
	  pallas_read_omp_fork(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time, &numberOfRequestedThreads);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpFork_callback(thread_reader->thread_trace->id,
									     time,
									     evtReader->user_data,
									     attribute_list,
									     numberOfRequestedThreads);
	}	
	break;

      case pallas::PALLAS_EVENT_OMP_JOIN:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpJoin_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_omp_join(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpJoin_callback(thread_reader->thread_trace->id,
								       time,
								       evtReader->user_data,
								       attribute_list);
	}	
	break;

      case pallas::PALLAS_EVENT_OMP_ACQUIRE_LOCK:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpAcquireLock_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint32_t lockID;
	  uint32_t acquisitionOrder;
	  pallas_read_omp_acquire_lock(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time, &lockID, &acquisitionOrder);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpAcquireLock_callback(thread_reader->thread_trace->id,
										    time,
										    evtReader->user_data,
										    attribute_list,
										    lockID,
										    acquisitionOrder);
	}	
	break;

      case pallas::PALLAS_EVENT_OMP_RELEASE_LOCK:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpReleaseLock_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint32_t lockID;
	  uint32_t acquisitionOrder;
	  pallas_read_omp_release_lock(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time,
				       &lockID, &acquisitionOrder);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpReleaseLock_callback(thread_reader->thread_trace->id,
										  time,
										  evtReader->user_data,
										  attribute_list,
										  lockID,
										    acquisitionOrder);
	}	
	break;

      case pallas::PALLAS_EVENT_OMP_TASK_CREATE:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpTaskCreate_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint64_t taskID;
	  pallas_read_omp_task_create(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time, &taskID);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpTaskCreate_callback(thread_reader->thread_trace->id,
										   time,
										   evtReader->user_data,
										   attribute_list,
										   taskID);
	}	
	break;

      case pallas::PALLAS_EVENT_OMP_TASK_SWITCH:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpTaskSwitch_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint64_t taskID;
	  pallas_read_omp_task_switch(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time, &taskID);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpTaskSwitch_callback(thread_reader->thread_trace->id,
										   time,
										   evtReader->user_data,
										   attribute_list,
										   taskID);
	}	
	break;

      case pallas::PALLAS_EVENT_OMP_TASK_COMPLETE:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpTaskComplete_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint64_t taskID;
	  pallas_read_omp_task_complete(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time, &taskID);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_OmpTaskComplete_callback(thread_reader->thread_trace->id,
										     time,
										     evtReader->user_data,
										     attribute_list,
										     taskID);
	}	
	break;

      case pallas::PALLAS_EVENT_METRIC:
	// not implemented 
	break;

      case pallas::PALLAS_EVENT_PARAMETER_STRING:
	// not implemented 
	break;

      case pallas::PALLAS_EVENT_PARAMETER_INT:
	// not implemented 
	break;

      case pallas::PALLAS_EVENT_PARAMETER_UNSIGNED_INT:
	// not implemented 
	break;

      case pallas::PALLAS_EVENT_THREAD_FORK:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadFork_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint32_t numberOfRequestedThreads;
	  pallas_read_thread_fork(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time, &numberOfRequestedThreads);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadFork_callback(thread_reader->thread_trace->id,
										time,
										evtReader->user_data,
										attribute_list,
										OTF2_PARADIGM_UNKNOWN,
										numberOfRequestedThreads);
	}	
	break;

      case pallas::PALLAS_EVENT_THREAD_JOIN:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadJoin_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_thread_join(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadJoin_callback(thread_reader->thread_trace->id,
										time,
										evtReader->user_data,
										attribute_list,
										OTF2_PARADIGM_UNKNOWN);
	}	
	break;

      case pallas::PALLAS_EVENT_THREAD_TEAM_BEGIN:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadTeamBegin_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_thread_team_begin(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadTeamBegin_callback(thread_reader->thread_trace->id,
										     time,
										     evtReader->user_data,
										     attribute_list,
										     OTF2_UNDEFINED_COMM);
	}	
	break;

      case pallas::PALLAS_EVENT_THREAD_TEAM_END:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadTeamEnd_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_thread_team_end(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadTeamEnd_callback(thread_reader->thread_trace->id,
										   time,
										   evtReader->user_data,
										   attribute_list,
										   OTF2_UNDEFINED_COMM);
	}	
	break;

      case pallas::PALLAS_EVENT_THREAD_ACQUIRE_LOCK:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadAcquireLock_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint32_t lockID;
	  uint32_t acquisitionOrder;
	  pallas_read_thread_acquire_lock(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time,
					  &lockID, &acquisitionOrder);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadAcquireLock_callback(thread_reader->thread_trace->id,
										       time,
										       evtReader->user_data,
										       attribute_list,
										       OTF2_PARADIGM_UNKNOWN,
										       lockID,
										       acquisitionOrder);
	}	
	break;

      case pallas::PALLAS_EVENT_THREAD_RELEASE_LOCK:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadReleaseLock_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  uint32_t lockID;
	  uint32_t acquisitionOrder;
	  pallas_read_thread_release_lock(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time,
					  &lockID, & acquisitionOrder);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadReleaseLock_callback(thread_reader->thread_trace->id,
										       time,
										       evtReader->user_data,
										       attribute_list,
										       OTF2_PARADIGM_UNKNOWN,
										       lockID,
										       acquisitionOrder);
	}	
	break;

      case pallas::PALLAS_EVENT_THREAD_TASK_CREATE:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadTaskCreate_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_thread_task_create(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadTaskCreate_callback(thread_reader->thread_trace->id,
										      time,
										      evtReader->user_data,
										      attribute_list,
										      OTF2_UNDEFINED_COMM,
										      0,// TODO:  creatingThread
										      0);
	}	
	break;

      case pallas::PALLAS_EVENT_THREAD_TASK_SWITCH:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadTaskSwitch_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_thread_task_switch(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadTaskSwitch_callback(thread_reader->thread_trace->id,
										      time,
										      evtReader->user_data,
										      attribute_list,
										      OTF2_UNDEFINED_COMM,
										      0,// TODO:  creatingThread
										      0);
	}	
	break;

      case pallas::PALLAS_EVENT_THREAD_TASK_COMPLETE:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadTaskComplete_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_thread_task_complete(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadTaskComplete_callback(thread_reader->thread_trace->id,
											time,
											evtReader->user_data,
											attribute_list,
											OTF2_UNDEFINED_COMM,
										      0,// TODO:  creatingThread
										      0);
	}	
	break;

      case pallas::PALLAS_EVENT_THREAD_CREATE:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadCreate_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_generic(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadCreate_callback(thread_reader->thread_trace->id,
										  time,
										  evtReader->user_data,
										  attribute_list,
										  OTF2_UNDEFINED_COMM,
										  0);
	}	
	break;

      case pallas::PALLAS_EVENT_THREAD_BEGIN:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadBegin_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_thread_begin(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadBegin_callback(thread_reader->thread_trace->id,
										 time,
										 evtReader->user_data,
										 attribute_list,
										 OTF2_UNDEFINED_COMM,
										 0); // TODO: sequenceCount
	}
      }
      break;

    case pallas::PALLAS_EVENT_THREAD_WAIT:
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadWait_callback) {
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_generic(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadWait_callback(thread_reader->thread_trace->id,
										time,
										evtReader->user_data,
										attribute_list,
										OTF2_UNDEFINED_COMM,
										0);
	}	
	break;

    case pallas::PALLAS_EVENT_THREAD_END:
      if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadEnd_callback) {
	AttributeList* attribute_list;
	pallas_timestamp_t time;
	pallas_read_thread_end(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	evtReader->callbacks.OTF2_GlobalEvtReaderCallback_ThreadEnd_callback(thread_reader->thread_trace->id,
									     time,
									     evtReader->user_data,
									     attribute_list,
									       OTF2_UNDEFINED_COMM,
										 0 );
	}	
	break;

    case pallas::PALLAS_EVENT_IO_CREATE_HANDLE:
      // Not implemented
      break;

    case pallas::PALLAS_EVENT_IO_DESTROY_HANDLE:
      // Not implemented
      break;

    case pallas::PALLAS_EVENT_IO_SEEK:
      // Not implemented
      break;

    case pallas::PALLAS_EVENT_IO_CHANGE_STATUS_FLAGS:
	break;

    case pallas::PALLAS_EVENT_IO_DELETE_FILE:
	break;

    case pallas::PALLAS_EVENT_IO_OPERATION_BEGIN:
	break;

    case pallas::PALLAS_EVENT_IO_DUPLICATE_HANDLE:
	break;

    case pallas::PALLAS_EVENT_IO_OPERATION_TEST:
	break;

    case pallas::PALLAS_EVENT_IO_OPERATION_ISSUED:
	break;

    case pallas::PALLAS_EVENT_IO_OPERATION_COMPLETE:
	break;

    case pallas::PALLAS_EVENT_IO_OPERATION_CANCELLED:
	break;

    case pallas::PALLAS_EVENT_IO_ACQUIRE_LOCK:
	break;

    case pallas::PALLAS_EVENT_IO_RELEASE_LOCK:
	break;

    case pallas::PALLAS_EVENT_IO_TRY_LOCK:
	break;

    case pallas::PALLAS_EVENT_PROGRAM_BEGIN:
	break;

    case pallas::PALLAS_EVENT_PROGRAM_END:
	break;

    case pallas::PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_REQUEST:
	break;

    case pallas::PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_COMPLETE:
	break;

    case pallas::PALLAS_EVENT_COMM_CREATE:
	break;

    case pallas::PALLAS_EVENT_COMM_DESTROY:
	break;

    case pallas::PALLAS_EVENT_GENERIC:

#define STRINGIFY(str) #str
	if(evtReader->callbacks.OTF2_GlobalEvtReaderCallback_Unknown_callback) {
	  pallas_warn("Unsupported event: %s\n", STRINGIFY(PALLAS_EVENT_GENERIC) );
	  AttributeList* attribute_list;
	  pallas_timestamp_t time;
	  pallas_read_generic(thread_reader, (PALLAS(AttributeList)**) &attribute_list, &time);
	  evtReader->callbacks.OTF2_GlobalEvtReaderCallback_Unknown_callback(thread_reader->thread_trace->id,
									     time,
									     evtReader->user_data,
									     attribute_list);
	}	
	break;	
    default:
      printf("Unsupported event type %d\n", event_type);
    }

  } // todo: else ? 

  if (! thread_reader->getNextToken(PALLAS_READ_FLAG_UNROLL_ALL).isValid()) {
    pallas_assert(thread_reader->isEndOfTrace());
  }

  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_HasGlobalEvent(OTF2_Reader* reader, OTF2_GlobalEvtReader* evtReader, int* flag) {
  pallas::ThreadReader*thread_reader = _get_next_global_event(reader, evtReader);
  *flag = (thread_reader!=nullptr);

  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_ReadGlobalEvents(OTF2_Reader* reader,
                                            OTF2_GlobalEvtReader* evtReader,
                                            uint64_t eventsToRead,
                                            uint64_t* eventsRead) {
  // TODO: implement
  int i = 0;
  for(i = 0; i < eventsToRead; i++) {
    int flag;
    OTF2_Reader_HasGlobalEvent(reader, evtReader,  &flag);
    if(!flag)			// no more event to process
      break;

    OTF2_Reader_ReadGlobalEvent(reader, evtReader);
  }

  *eventsRead = i;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_ReadAllGlobalEvents(OTF2_Reader* reader,
                                               OTF2_GlobalEvtReader* evtReader,
                                               uint64_t* eventsRead) {
  return OTF2_Reader_ReadGlobalEvents(reader, evtReader, UINT64_MAX, eventsRead) ;
}

OTF2_ErrorCode OTF2_Reader_ReadLocalDefinitions(OTF2_Reader* reader,
                                                OTF2_DefReader* defReader,
                                                uint64_t definitionsToRead,
                                                uint64_t* definitionsRead) {
  PALLAS(Archive)* archive = (PALLAS(Archive)*)defReader->thread_reader->archive;
  int nb_definition_read = 0;

#define CHECK_OTF2_CALLBACK_SUCCESS(_f_) do {			\
    if(nb_definition_read >= definitionsToRead) goto out;	\
    if(_f_ != OTF2_CALLBACK_SUCCESS) goto out;			\
    nb_definition_read++;					\
  } while(0)

  
  // TODO: implement local definition
  // out:
  *definitionsRead = nb_definition_read;
  pallas_assert(*definitionsRead <= definitionsToRead);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_ReadAllLocalDefinitions(OTF2_Reader* reader,
                                                   OTF2_DefReader* defReader,
                                                   uint64_t* definitionsRead) {
  return OTF2_Reader_ReadLocalDefinitions(reader, defReader, UINT64_MAX, definitionsRead);
}

OTF2_ErrorCode OTF2_Reader_ReadGlobalDefinitions(OTF2_Reader* reader,
                                                 OTF2_GlobalDefReader* defReader,
                                                 uint64_t definitionsToRead,
                                                 uint64_t* definitionsRead) {
  PALLAS(GlobalArchive)* archive = (PALLAS(GlobalArchive)*)reader->archive;
  int nb_definition_read = 0;

#define CHECK_OTF2_CALLBACK_SUCCESS(_f_) do {			  \
    if(nb_definition_read >= definitionsToRead) goto out; \
    if(_f_ != OTF2_CALLBACK_SUCCESS) goto out;		\
    nb_definition_read++;			\
  } while(0)

  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_String_callback) {
    for ( const auto &string : archive->definitions.strings ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_String_callback
				  (defReader->user_data,
				   string.first,
				   string.second.str));
    }
  }

  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_Region_callback) {
    for ( const auto &region : archive->definitions.regions ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_Region_callback
				  (defReader->user_data,
				   region.first,
				   region.second.string_ref, // name	
				   OTF2_UNDEFINED_STRING, // canonicalName
				   OTF2_UNDEFINED_STRING, // description
				   OTF2_REGION_ROLE_UNKNOWN, // regionRole
				   OTF2_PARADIGM_UNKNOWN,    // paradigm
				   OTF2_REGION_FLAG_NONE,    // regionFlags
				   OTF2_UNDEFINED_STRING, // sourceFile
				   0,			  // beginLineNumber
				   0));			  // endLineNumber
    }
  }

  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_Attribute_callback) {
    for ( const auto &attr : archive->definitions.attributes ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_Attribute_callback
				  (defReader->user_data,
				   attr.first,
				   attr.second.name,
				   attr.second.description,
				   attr.second.type));
    }
  }

  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_Group_callback) {
    for ( const auto &attr : archive->definitions.groups ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_Group_callback
				  (defReader->user_data,
				   attr.first,
				   attr.second.name,
				   OTF2_GROUP_TYPE_UNKNOWN, // groupType
				   OTF2_PARADIGM_UNKNOWN, // paradigm
				   0, // groupFlags
				   attr.second.numberOfMembers,
				   attr.second.members
				   )
				   );
    }
  }


  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_Comm_callback) {
    for ( const auto &attr : archive->definitions.comms ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_Comm_callback
				  (defReader->user_data,
				   attr.first,
				   attr.second.name,
				   attr.second.group,
				   attr.second.parent,
				   0)); // flags
    }
  }

  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_LocationGroup_callback) {
    for ( const auto &lg : archive->location_groups ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_LocationGroup_callback
				  (defReader->user_data,
				   lg.id, // self
				   lg.name, // name
				   OTF2_LOCATION_GROUP_TYPE_UNKNOWN, // locationGroupType
				   OTF2_UNDEFINED_SYSTEM_TREE_NODE, // systemTreeParent
				   lg.parent // creatingLocationGroup
				   ));
    }
  }

  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_Location_callback) {
    /*for ( const auto &loc : archive->locations ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_Location_callback
				  (defReader->user_data,
				   loc.id, // self
				   loc.name, // name
				   OTF2_LOCATION_TYPE_UNKNOWN, // locationType
				   0, // numberOfEvents
				   loc.parent // locationGroup
				   ));
    }
    */
  }

  // todo: handle the other definitions

 out:
  *definitionsRead = nb_definition_read;
  pallas_assert(*definitionsRead <= definitionsToRead);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_ReadAllGlobalDefinitions(OTF2_Reader* reader,
                                                    OTF2_GlobalDefReader* defReader,
                                                    uint64_t* definitionsRead) {
  return OTF2_Reader_ReadGlobalDefinitions(reader, defReader, UINT64_MAX, definitionsRead);
}

OTF2_ErrorCode OTF2_Reader_ReadLocalSnapshots(OTF2_Reader* reader,
                                              OTF2_SnapReader* snapReader,
                                              uint64_t recordsToRead,
                                              uint64_t* recordsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadAllLocalSnapshots(OTF2_Reader* reader,
                                                 OTF2_SnapReader* snapReader,
                                                 uint64_t* recordsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadGlobalSnapshots(OTF2_Reader* reader,
                                               OTF2_GlobalSnapReader* snapReader,
                                               uint64_t recordsToRead,
                                               uint64_t* recordsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadAllGlobalSnapshots(OTF2_Reader* reader,
                                                  OTF2_GlobalSnapReader* snapReader,
                                                  uint64_t* recordsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadMarkers(OTF2_Reader* reader,
                                       OTF2_MarkerReader* markerReader,
                                       uint64_t markersToRead,
                                       uint64_t* markersRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadAllMarkers(OTF2_Reader* reader, OTF2_MarkerReader* markerReader, uint64_t* markersRead) {
  NOT_IMPLEMENTED;
}

OTF2_EvtReader* OTF2_Reader_GetEvtReader(OTF2_Reader* reader, OTF2_LocationRef location) {
  for(int i = 0; i<reader->nb_locations; i++) {
    if(reader->locations[i] == location &&
       reader->selected_locations[i]) {

      if(! reader->thread_readers[i]) {
	PALLAS(GlobalArchive)* global_archive = (PALLAS(GlobalArchive)*)reader->archive;
	PALLAS(Archive)* thread_archive = global_archive->getArchiveFromLocation(location);
	reader->thread_readers[i] = new	pallas::ThreadReader(thread_archive, location, 0);
      }

      if(reader->evt_readers[i].location == OTF2_UNDEFINED_LOCATION) {
	// This evt readers has not been open yet.
	reader->evt_readers[i].location = location;
	reader->evt_readers[i].thread_reader = reader->thread_readers[i];
      }

      return &reader->evt_readers[i];
    }
  }
  return NULL;
}

OTF2_GlobalEvtReader* OTF2_Reader_GetGlobalEvtReader(OTF2_Reader* reader) {

  return reader->global_evt_reader;
}

OTF2_DefReader* OTF2_Reader_GetDefReader(OTF2_Reader* reader, OTF2_LocationRef location) {
  for(int i = 0; i<reader->nb_locations; i++) {
    if(reader->locations[i] == location &&
       reader->selected_locations[i]) {

      if(! reader->thread_readers[i]) {
	PALLAS(GlobalArchive)* global_archive = (PALLAS(GlobalArchive)*)reader->archive;
	PALLAS(Archive)* thread_archive = global_archive->getArchiveFromLocation(location);
	reader->thread_readers[i] = new	pallas::ThreadReader(thread_archive, location, 0);
      }

      if(reader->def_readers[i].location == OTF2_UNDEFINED_LOCATION) {
	// This def readers has not been open yet.
	memset(&reader->def_readers[i], 0, sizeof(struct OTF2_DefReader_struct));
	reader->def_readers[i].location = location;
	reader->def_readers[i].thread_reader = reader->thread_readers[i];
      }

      return &reader->def_readers[i];
    }
  }
  return NULL;
}

OTF2_GlobalDefReader* OTF2_Reader_GetGlobalDefReader(OTF2_Reader* reader) {

  return reader->global_def_reader;
}

OTF2_SnapReader* OTF2_Reader_GetSnapReader(OTF2_Reader* reader, OTF2_LocationRef location) {
  NOT_IMPLEMENTED;
}

OTF2_GlobalSnapReader* OTF2_Reader_GetGlobalSnapReader(OTF2_Reader* reader) {
  NOT_IMPLEMENTED;
}

OTF2_ThumbReader* OTF2_Reader_GetThumbReader(OTF2_Reader* reader, uint32_t number) {
  NOT_IMPLEMENTED;
}

OTF2_MarkerReader* OTF2_Reader_GetMarkerReader(OTF2_Reader* reader) {
  NOT_IMPLEMENTED;
}

OTF2_MarkerWriter* OTF2_Reader_GetMarkerWriter(OTF2_Reader* reader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseEvtReader(OTF2_Reader* reader, OTF2_EvtReader* evtReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseGlobalEvtReader(OTF2_Reader* reader, OTF2_GlobalEvtReader* globalEvtReader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_CloseDefReader(OTF2_Reader* reader, OTF2_DefReader* defReader) {
  // TODO: free/unitinialize def_reader?
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_CloseGlobalDefReader(OTF2_Reader* reader, OTF2_GlobalDefReader* globalDefReader) {
  // TODO: free reader->global_def_reader?
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_CloseSnapReader(OTF2_Reader* reader, OTF2_SnapReader* snapReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseGlobalSnapReader(OTF2_Reader* reader, OTF2_GlobalSnapReader* globalSnapReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseThumbReader(OTF2_Reader* reader, OTF2_ThumbReader* thumbReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseMarkerReader(OTF2_Reader* reader, OTF2_MarkerReader* markerReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseMarkerWriter(OTF2_Reader* reader, OTF2_MarkerWriter* markerWriter) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetVersion(OTF2_Reader* reader, uint8_t* major, uint8_t* minor, uint8_t* bugfix) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetChunkSize(OTF2_Reader* reader,
                                        uint64_t* chunkSizeEvents,
                                        uint64_t* chunkSizeDefinitions) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetFileSubstrate(OTF2_Reader* reader, OTF2_FileSubstrate* substrate) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetCompression(OTF2_Reader* reader, OTF2_Compression* compression) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetNumberOfLocations(OTF2_Reader* reader, uint64_t* numberOfLocations) {
  *numberOfLocations = reader->nb_locations;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_GetNumberOfGlobalDefinitions(OTF2_Reader* reader, uint64_t* numberOfDefinitions) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetMachineName(OTF2_Reader* reader, char** machineName) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetCreator(OTF2_Reader* reader, char** creator) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetDescription(OTF2_Reader* reader, char** description) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetPropertyNames(OTF2_Reader* reader, uint32_t* numberOfProperties, char*** names) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetProperty(OTF2_Reader* reader, const char* name, char** value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetBoolProperty(OTF2_Reader* reader, const char* name, bool* value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetTraceId(OTF2_Reader* reader, uint64_t* id) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetNumberOfSnapshots(OTF2_Reader* reader, uint32_t* number) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetNumberOfThumbnails(OTF2_Reader* reader, uint32_t* number) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_OpenEvtFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_CloseEvtFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_OpenDefFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_CloseDefFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_OpenSnapFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_CloseSnapFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_SelectLocation(OTF2_Reader* reader, OTF2_LocationRef location) {
  for(int i=0; i<reader->nb_locations; i++) {
    if(reader->locations[i] == location) {
      reader->selected_locations[i] = 1;
    }
  }

  return OTF2_SUCCESS;
}

/** @brief Determines if this reader is the primary archive which handles the
 *  global archive state (anchor file, global definitions, marker, ...).
 *
 *  @param reader       Reader handle.
 *  @param[out] result  Storage for the result.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Reader_IsPrimary(OTF2_Reader* reader, bool* result) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_IsMaster(OTF2_Reader* reader, bool* result) {
  NOT_IMPLEMENTED;
}
