/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include "otf2/otf2.h"
#include "pallas/pallas_log.h"

#define check_status( status, ...) if(status != OTF2_SUCCESS) {	\
    fprintf(stderr,  __VA_ARGS__);				\
    abort();							\
  }

#define check_pointer( ptr, ...) if(ptr == NULL) {		\
    fprintf(stderr,  __VA_ARGS__);				\
    abort();							\
  }

int show_definitions = 0;

#define OTF2_PRINT_DEF(format, ...) do {if(show_definitions) printf(format, __VA_ARGS__); } while(0)

static int otf2_EVENT_COLUMN_WIDTH=20;

#define STRING_MAX_LEN 1024
#define SMALL_BUFFER 128

struct otf2_string {
  OTF2_StringRef ref;
  char str[STRING_MAX_LEN];
};

struct otf2_region {
  OTF2_RegionRef ref;
  OTF2_StringRef name;
  struct otf2_string* name_str;
};

struct otf2_group {
  OTF2_GroupRef ref;    /**< Id of that Group.*/
  OTF2_StringRef name;
  uint32_t numberOfMembers;
  uint32_t* members;
  struct otf2_string* name_str;
};

struct otf2_comm {
  OTF2_CommRef ref;     /**< Id of that Comm.*/
  OTF2_StringRef name;
  OTF2_GroupRef group;
  OTF2_CommRef parent;
  struct otf2_string* name_str;
};

struct otf2_print_user_data {
  size_t nb_locations;
  uint64_t* locations;

  struct otf2_string *strings;
  int nb_strings;

  struct otf2_region *regions;
  int nb_regions;

  struct otf2_group *groups;
  int nb_groups;

  struct otf2_comm *comms;
  int nb_comms;

};

struct otf2_string* get_string_t(struct otf2_print_user_data *user_data, OTF2_StringRef string) {
  for(int i = 0; i<user_data->nb_strings; i++) {
    if(user_data->strings[i].ref == string)
      return &user_data->strings[i];
  }
  return NULL;
}

char* get_string(struct otf2_print_user_data *user_data, OTF2_StringRef string) {
  struct otf2_string* str = get_string_t(user_data, string);
  if(str) return str->str;
  return NULL;
}

struct otf2_region* get_region_t(struct otf2_print_user_data *user_data, OTF2_RegionRef region) {
  for(int i = 0; i<user_data->nb_regions; i++) {
    if(user_data->regions[i].ref == region) {
      if(! user_data->regions[i].name_str)
	user_data->regions[i].name_str = get_string_t(user_data, user_data->regions[i].name);
      return &user_data->regions[i];
    }
  }
  return NULL;
}

char* get_region_str(struct otf2_print_user_data *user_data, OTF2_RegionRef region) {
  struct otf2_region* r = get_region_t(user_data, region);
  if(r) {
    return r->name_str->str;
    char* buf = malloc(SMALL_BUFFER);
    snprintf(buf, SMALL_BUFFER, "Region_%d", region);
    return buf;			/* warning: memory leak ! */
  } else {
    char* buf = malloc(SMALL_BUFFER);
    snprintf(buf, SMALL_BUFFER, "Unknown_Region_%d", region);
    return buf; 		/* warning: memory leak ! */
  }
}

struct otf2_group* get_group_t(struct otf2_print_user_data *user_data, OTF2_GroupRef group) {
  for(int i = 0; i<user_data->nb_groups; i++) {
    if(user_data->groups[i].ref == group) {
      if(! user_data->groups[i].name_str)
	user_data->groups[i].name_str = get_string_t(user_data, user_data->groups[i].name);
      return &user_data->groups[i];
    }
  }
  return NULL;
}

char* get_group_str(struct otf2_print_user_data *user_data, OTF2_GroupRef group) {
  struct otf2_group* r = get_group_t(user_data, group);
  if(r) {
    return r->name_str->str;
    char* buf = malloc(SMALL_BUFFER);
    snprintf(buf, SMALL_BUFFER, "Group_%d", group);
    return buf;			/* warning: memory leak ! */
  } else {
    char* buf = malloc(SMALL_BUFFER);
    snprintf(buf, SMALL_BUFFER, "Unknown_Group_%d", group);
    return buf; 		/* warning: memory leak ! */
  }
}

struct otf2_comm* get_comm_t(struct otf2_print_user_data *user_data, OTF2_CommRef comm) {
  for(int i = 0; i<user_data->nb_comms; i++) {
    if(user_data->comms[i].ref == comm) {
      if(! user_data->comms[i].name_str)
	user_data->comms[i].name_str = get_string_t(user_data, user_data->comms[i].name);
      return &user_data->comms[i];
    }
  }
  return NULL;
}

char* get_comm_str(struct otf2_print_user_data *user_data, OTF2_CommRef comm) {
  struct otf2_comm* r = get_comm_t(user_data, comm);
  if(r) {
    return r->name_str->str;
    char* buf = malloc(SMALL_BUFFER);
    snprintf(buf, SMALL_BUFFER, "Comm_%d", comm);
    return buf;			/* warning: memory leak ! */
  } else {
    char* buf = malloc(SMALL_BUFFER);
    snprintf(buf, SMALL_BUFFER, "Unknown_Comm_%d", comm);
    return buf; 		/* warning: memory leak ! */
  }
}

OTF2_CallbackCode print_global_def_string(void *userData, OTF2_StringRef self, const char *string) {
  OTF2_PRINT_DEF("Global_def_string(userData=%p, self=%d, string='%s');\n", userData, self, string);
  struct otf2_print_user_data* user_data = userData;

  struct otf2_string* s = get_string_t(user_data, self);
  pallas_assert(s == NULL);
  int index = user_data->nb_strings++;
  user_data->strings = realloc(user_data->strings, sizeof(struct otf2_string) * user_data->nb_strings);
  s = &user_data->strings[index];
  s->ref = self;
  strncpy(s->str, string, STRING_MAX_LEN-1);

  return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode print_global_def_region(void *userData,
					  OTF2_RegionRef self,
					  OTF2_StringRef name,
					  OTF2_StringRef canonicalName,
					  OTF2_StringRef description,
					  OTF2_RegionRole regionRole,
					  OTF2_Paradigm paradigm,
					  OTF2_RegionFlag regionFlags,
					  OTF2_StringRef sourceFile,
					  uint32_t beginLineNumber,
					  uint32_t endLineNumber) {
  OTF2_PRINT_DEF("Global_def_region(userData=%p, self=%d, name=%d, canonicalName=%d, description=%d, regionRole=%d, paradigm=%d, regionFlags=%d, sourceFile=%d, beginLineNumber=%d, endLineNumber=%d)\n",
	 userData,
	 self,
	 name,
	 canonicalName,
	 description,
	 regionRole,
	 paradigm,
	 regionFlags,
	 sourceFile,
	 beginLineNumber,
	 endLineNumber);
  struct otf2_print_user_data* user_data = userData;

  struct otf2_region* r = get_region_t(user_data, self);
  pallas_assert(r == NULL);
  int index = user_data->nb_regions++;
  user_data->regions = realloc(user_data->regions, sizeof(struct otf2_region) * user_data->nb_regions);
  r = &user_data->regions[index];
  r->ref = self;
  r->name = name;
  r->name_str = NULL;

  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_global_def_attribute(void *userData,
					     OTF2_AttributeRef self,
					     OTF2_StringRef name,
					     OTF2_StringRef description,
					     OTF2_Type type) {
  OTF2_PRINT_DEF("Global_def_attribute(userData=%p, self=%d, name=%d, description=%d, type=%d)\n",
	 userData,
	 self,
	 name,
	 description,
	 type);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_global_def_location_group(void *userData,
						  OTF2_LocationGroupRef self,
						  OTF2_StringRef name,
						  OTF2_LocationGroupType locationGroupType,
						  OTF2_SystemTreeNodeRef systemTreeParent,
						  OTF2_LocationGroupRef creatingLocationGroup) {
  OTF2_PRINT_DEF("Global_def_location_group(userData=%p, self=%d, name=%d, locationGroupType=%d, systemTreeParent=%d, creatingLocationGroup=%d)\n",
	 userData,
	 self,
	 name,
	 locationGroupType,
	 systemTreeParent,
	 creatingLocationGroup);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_global_def_location(void *userData,
					    OTF2_LocationRef self,
					    OTF2_StringRef name,
					    OTF2_LocationType locationType,
					    uint64_t numberOfEvents,
					    OTF2_LocationGroupRef locationGroup) {
  OTF2_PRINT_DEF("Global_def_location(userData=%p, self=%ld, name=%d, locationType=%d, numberOfEvents=%ld, locationGroup=%d)\n",
	 userData,
	 self,
	 name,
	 locationType,
	 numberOfEvents,
	 locationGroup);

  struct otf2_print_user_data *user_data = userData;
  int index = user_data->nb_locations++;
  user_data->locations = realloc(user_data->locations, sizeof(uint64_t)*user_data->nb_locations);
  pallas_assert(user_data->locations);
  user_data->locations[index] = self;
  return OTF2_CALLBACK_SUCCESS;
  
}

OTF2_CallbackCode print_global_def_group(void* userData,
					 OTF2_GroupRef self,
					 OTF2_StringRef name,
					 OTF2_GroupType groupType,
					 OTF2_Paradigm paradigm,
					 OTF2_GroupFlag groupFlags,
					 uint32_t numberOfMembers,
					 const uint32_t* members) {
  OTF2_PRINT_DEF("Global_def_group(userData=%p, self=%d, name=%d, groupType=%d, paradigm=%d, groupFlags=%x, numberOfMembers=%d, [", userData, self, name, groupType, paradigm, groupFlags, numberOfMembers);
  for(int i=0; i<numberOfMembers; i++ ) {
    OTF2_PRINT_DEF("%s%u", i>0?", ":"", members[i]);
  }
  OTF2_PRINT_DEF("]%c", '\n');

  struct otf2_print_user_data* user_data = userData;

  struct otf2_group* g = get_group_t(user_data, self);
  pallas_assert(g == NULL);
  int index = user_data->nb_groups++;
  user_data->groups = realloc(user_data->groups, sizeof(struct otf2_group) * user_data->nb_groups);
  g = &user_data->groups[index];
  g->ref = self;
  g->name = name;
  g->numberOfMembers = numberOfMembers;
  g->members = malloc(sizeof(uint64_t) * numberOfMembers);
  for(int i=0; i<numberOfMembers; i++) {
    g->members[i] = members[i];
  }
  g->name_str = NULL;

  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_global_def_comm(void* userData,
					OTF2_CommRef self,
					OTF2_StringRef name,
					OTF2_CommRef group,
					OTF2_CommRef parent,
					OTF2_CommFlag flags) {
  OTF2_PRINT_DEF("Global_def_comm(userData=%p, self=%d, name=%d, group=%x, parent=%d, flags=%x)\n", userData, self, name, group, parent, flags);
  struct otf2_print_user_data* user_data = userData;

  struct otf2_comm* c = get_comm_t(user_data, self);
  pallas_assert(c == NULL);
  int index = user_data->nb_comms++;
  user_data->comms = realloc(user_data->comms, sizeof(struct otf2_comm) * user_data->nb_comms);
  c = &user_data->comms[index];
  c->ref = self;
  c->name = name;
  c->group = group;
  c->parent = parent;
  c->name_str = NULL;
  return OTF2_CALLBACK_SUCCESS;
}


OTF2_GlobalDefReaderCallbacks* otf2_print_define_global_def_callbacks(OTF2_Reader *reader) {
  OTF2_GlobalDefReaderCallbacks* def_callbacks = OTF2_GlobalDefReaderCallbacks_New();

  OTF2_GlobalDefReaderCallbacks_SetStringCallback( def_callbacks, print_global_def_string );
  OTF2_GlobalDefReaderCallbacks_SetAttributeCallback( def_callbacks, print_global_def_attribute );
//  OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodeCallback( def_callbacks, print_global_def_system_tree_node );
  OTF2_GlobalDefReaderCallbacks_SetLocationGroupCallback( def_callbacks, print_global_def_location_group );
  OTF2_GlobalDefReaderCallbacks_SetLocationCallback( def_callbacks, print_global_def_location );
  OTF2_GlobalDefReaderCallbacks_SetRegionCallback( def_callbacks, print_global_def_region );
  OTF2_GlobalDefReaderCallbacks_SetGroupCallback( def_callbacks, print_global_def_group );
  OTF2_GlobalDefReaderCallbacks_SetCommCallback( def_callbacks, print_global_def_comm );

  return def_callbacks;
}



#define OTF2_PRINT_GENERIC(event, location, timestamp, attributes)	\
  printf( "%-*s %15ld %20lu  %s\n", otf2_EVENT_COLUMN_WIDTH, event, location, timestamp, attributes)


OTF2_CallbackCode print_enter(OTF2_LocationRef locationID,
			      OTF2_TimeStamp time,
			      void *userData,
			      OTF2_AttributeList *attributeList,
			      OTF2_RegionRef region) {
  OTF2_PRINT_GENERIC("ENTER",  locationID, time, get_region_str(userData, region));

  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_leave(OTF2_LocationRef locationID,
			      OTF2_TimeStamp time,
			      void *userData,
			      OTF2_AttributeList *attributeList,
			      OTF2_RegionRef region) {
  OTF2_PRINT_GENERIC("LEAVE",  locationID, time, get_region_str(userData, region));
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_mpi_send(OTF2_LocationRef locationID,
				  OTF2_TimeStamp time,
				  void* userData,
				  OTF2_AttributeList* attributeList,
				  uint32_t receiver,
				  OTF2_CommRef communicator,
				  uint32_t msgTag,
				  uint64_t msgLength) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "receiver: %d, comm: %x, tag: %d, len: %lu",
	   receiver, communicator, msgTag, msgLength);
  OTF2_PRINT_GENERIC("MPI_SEND",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_mpi_isend(OTF2_LocationRef locationID,
				  OTF2_TimeStamp time,
				  void* userData,
				  OTF2_AttributeList* attributeList,
				  uint32_t receiver,
				  OTF2_CommRef communicator,
				  uint32_t msgTag,
				  uint64_t msgLength,
				  uint64_t requestID) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "receiver: %d, comm: %x, tag: %d, len: %lu, req: %p",
	   receiver, communicator, msgTag, msgLength, (void*)requestID);
  OTF2_PRINT_GENERIC("MPI_ISEND",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_mpi_isend_complete(OTF2_LocationRef locationID,
					   OTF2_TimeStamp time,
					   void* userData,
					   OTF2_AttributeList* attributeList,
					   uint64_t requestID) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "req: %p", (void*)requestID);
  OTF2_PRINT_GENERIC("MPI_ISEND_COMPLETE",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_mpi_irecv_request(OTF2_LocationRef locationID,
					  OTF2_TimeStamp time,
					  void* userData,
					  OTF2_AttributeList* attributeList,
					  uint64_t requestID) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "req: %p", (void*)requestID);
  OTF2_PRINT_GENERIC("MPI_IRECV_REQUEST",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_mpi_recv(OTF2_LocationRef locationID,
				 OTF2_TimeStamp time,
				 void* userData,
				 OTF2_AttributeList* attributeList,
				 uint32_t sender,
				 OTF2_CommRef communicator,
				 uint32_t msgTag,
				 uint64_t msgLength) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "sender: %d, comm: %x, tag: %d, len: %lu",
	   sender, communicator, msgTag, msgLength);
  OTF2_PRINT_GENERIC("MPI_RECV",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_mpi_irecv(OTF2_LocationRef locationID,
				  OTF2_TimeStamp time,
				  void* userData,
				  OTF2_AttributeList* attributeList,
				  uint32_t sender,
				  OTF2_CommRef communicator,
				  uint32_t msgTag,
				  uint64_t msgLength,
				  uint64_t requestID) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "sender: %d, comm: %x, tag: %d, len: %lu, req: %p",
	   sender, communicator, msgTag, msgLength, (void*)requestID);
  OTF2_PRINT_GENERIC("MPI_IRECV",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_mpi_request_test(OTF2_LocationRef locationID,
					 OTF2_TimeStamp time,
					 void* userData,
					 OTF2_AttributeList* attributeList,
					 uint64_t requestID) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "req: %p", (void*)requestID);
  OTF2_PRINT_GENERIC("MPI_REQUEST_TEST",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_mpi_request_cancelled(OTF2_LocationRef locationID,
					      OTF2_TimeStamp time,
					      void* userData,
					      OTF2_AttributeList* attributeList,
					      uint64_t requestID) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "req: %p", (void*)requestID);
  OTF2_PRINT_GENERIC("MPI_REQUEST_CANCELLED",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_mpi_collective_begin(OTF2_LocationRef locationID,
					     OTF2_TimeStamp time,
					     void* userData,
					     OTF2_AttributeList* attributeList) {
  OTF2_PRINT_GENERIC("MPI_COLLECTIVE_BEGIN",  locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_mpi_collective_end(OTF2_LocationRef locationID,
					   OTF2_TimeStamp time,
					   void* userData,
					   OTF2_AttributeList* attributeList,
					   OTF2_CollectiveOp collectiveOp,
					   OTF2_CommRef communicator,
					   uint32_t root,
					   uint64_t sizeSent,
					   uint64_t sizeReceived) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "op: %d, comm: %x, root: %d, sent: %lu, recved: %lu",
	   collectiveOp, communicator, root, sizeSent, sizeReceived);
  OTF2_PRINT_GENERIC("MPI_COLLECTIVE_END",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_omp_fork(OTF2_LocationRef locationID,
				 OTF2_TimeStamp time,
				 void* userData,
				 OTF2_AttributeList* attributeList,
				 uint32_t numberOfRequestedThreads) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "nbThreads: %d", numberOfRequestedThreads);
  OTF2_PRINT_GENERIC("OMP_FORK",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_omp_join(OTF2_LocationRef locationID,
				 OTF2_TimeStamp time,
				 void* userData,
				 OTF2_AttributeList* attributeList) {
  OTF2_PRINT_GENERIC("OMP_JOIN",  locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_omp_acquire_lock(OTF2_LocationRef locationID,
					 OTF2_TimeStamp time,
					 void* userData,
					 OTF2_AttributeList* attributeList,
					 uint32_t lockID,
					 uint32_t acquisitionOrder) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "lockID: %d", lockID);
  OTF2_PRINT_GENERIC("OMP_ACQUIRE_LOCK",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_omp_release_lock(OTF2_LocationRef locationID,
					 OTF2_TimeStamp time,
					 void* userData,
					 OTF2_AttributeList* attributeList,
					 uint32_t lockID,
					 uint32_t acquisitionOrder) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "lockID: %d", lockID);
  OTF2_PRINT_GENERIC("OMP_RELEASE_LOCK",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_omp_task_create(OTF2_LocationRef locationID,
					OTF2_TimeStamp time,
					void* userData,
					OTF2_AttributeList* attributeList,
					uint64_t taskID) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "taskID: %lu", taskID);
  OTF2_PRINT_GENERIC("OMP_TASK_CREATE",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_omp_task_switch(OTF2_LocationRef locationID,
					OTF2_TimeStamp time,
					void* userData,
					OTF2_AttributeList* attributeList,
					uint64_t taskID) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "taskID: %lu", taskID);
  OTF2_PRINT_GENERIC("OMP_TASK_SWITCH",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_omp_task_complete(OTF2_LocationRef locationID,
					  OTF2_TimeStamp time,
					  void* userData,
					  OTF2_AttributeList* attributeList,
					  uint64_t taskID) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "taskID: %lu", taskID);
  OTF2_PRINT_GENERIC("OMP_TASK_COMPLETE",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_thread_fork(OTF2_LocationRef locationID,
				    OTF2_TimeStamp time,
				    void* userData,
				    OTF2_AttributeList* attributeList,
				    OTF2_Paradigm model,
				    uint32_t numberOfRequestedThreads) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "nbThreads: %d", numberOfRequestedThreads);
  OTF2_PRINT_GENERIC("THREAD_FORK",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}
  

OTF2_CallbackCode print_thread_team_begin(OTF2_LocationRef locationID,
					  OTF2_TimeStamp time,
					  void* userData,
					  OTF2_AttributeList* attributeList,
					  OTF2_CommRef threadTeam) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "team: %d", threadTeam);
  OTF2_PRINT_GENERIC("THREAD_TEAM_BEGIN",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}
  

OTF2_CallbackCode print_thread_acquire_lock(OTF2_LocationRef locationID,
					    OTF2_TimeStamp time,
					    void* userData,
					    OTF2_AttributeList* attributeList,
					    OTF2_Paradigm model,
					    uint32_t lockID,
					    uint32_t acquisitionOrder) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "lockID: %d", lockID);
  OTF2_PRINT_GENERIC("THREAD_ACQUIRE_LOCK",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}
  

OTF2_CallbackCode print_thread_task_create(OTF2_LocationRef locationID,
					   OTF2_TimeStamp time,
					   void* userData,
					   OTF2_AttributeList* attributeList,
					   OTF2_CommRef threadTeam,
					   uint32_t creatingThread,
					   uint32_t generationNumber) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "team: %d", threadTeam);
  OTF2_PRINT_GENERIC("THREAD_TASK_CREATE",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}
  

OTF2_CallbackCode print_thread_task_complete(OTF2_LocationRef locationID,
					     OTF2_TimeStamp time,
					     void* userData,
					     OTF2_AttributeList* attributeList,
					     OTF2_CommRef threadTeam,
					     uint32_t creatingThread,
					     uint32_t generationNumber) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "team: %d", threadTeam);
  OTF2_PRINT_GENERIC("THREAD_TASK_COMPLETE",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}
  

OTF2_CallbackCode print_thread_begin(OTF2_LocationRef locationID,
				     OTF2_TimeStamp time,
				     void* userData,
				     OTF2_AttributeList* attributeList,
				     OTF2_CommRef threadContingent,
				     uint64_t sequenceCount) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "team: %d", threadContingent);
  OTF2_PRINT_GENERIC("THREAD_BEGIN",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
  }
  

OTF2_CallbackCode print_thread_end(OTF2_LocationRef locationID,
				   OTF2_TimeStamp time,
				   void* userData,
				   OTF2_AttributeList* attributeList,
				   OTF2_CommRef threadContingent,
				   uint64_t sequenceCount) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "team: %d", threadContingent);
  OTF2_PRINT_GENERIC("THREAD_END",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}
  
OTF2_CallbackCode print_thread_wait(OTF2_LocationRef locationID,
				    OTF2_TimeStamp time,
				    void* userData,
				    OTF2_AttributeList* attributeList,
				    OTF2_CommRef threadContingent,
				    uint64_t sequenceCount) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "team: %d", threadContingent);
  OTF2_PRINT_GENERIC("THREAD_WAIT",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_thread_create(OTF2_LocationRef locationID,
				      OTF2_TimeStamp time,
				      void* userData,
				      OTF2_AttributeList* attributeList,
				      OTF2_CommRef threadContingent,
				      uint64_t sequenceCount) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "team: %d", threadContingent);
  OTF2_PRINT_GENERIC("THREAD_CREATE",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_thread_task_switch(OTF2_LocationRef locationID,
					   OTF2_TimeStamp time,
					   void* userData,
					   OTF2_AttributeList* attributeList,
					   OTF2_CommRef threadTeam,
					   uint32_t creatingThread,
					   uint32_t generationNumber) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "team: %d", threadTeam);
  OTF2_PRINT_GENERIC("THREAD_TASK_SWITCH",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_thread_release_lock(OTF2_LocationRef locationID,
					    OTF2_TimeStamp time,
					    void* userData,
					    OTF2_AttributeList* attributeList,
					    OTF2_Paradigm model,
					    uint32_t lockID,
					    uint32_t acquisitionOrder) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "lockID: %d", lockID);
  OTF2_PRINT_GENERIC("THREAD_RELEASE_LOCK",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_thread_team_end(OTF2_LocationRef locationID,
					OTF2_TimeStamp time,
					void* userData,
					OTF2_AttributeList* attributeList,
					OTF2_CommRef threadTeam) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "threadTeam: %d", threadTeam);
  OTF2_PRINT_GENERIC("THREAD_TEAM_END_LOCK",  locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_thread_join(OTF2_LocationRef locationID,
				    OTF2_TimeStamp time,
				    void* userData,
				    OTF2_AttributeList* attributeList,
				    OTF2_Paradigm model) {
  OTF2_PRINT_GENERIC("THREAD_JOIN",  locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_io_create_handle(OTF2_LocationRef locationID,
					 OTF2_TimeStamp time,
					 void* userData,
					 OTF2_AttributeList* attributeList,
					 OTF2_IoHandleRef handle,
					 OTF2_IoAccessMode mode,
					 OTF2_IoCreationFlag creationFlags,
					 OTF2_IoStatusFlag statusFlags) {
  OTF2_PRINT_GENERIC("IO_CREATE_HANDLE",  locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}
  

OTF2_CallbackCode print_io_duplicate_handle(OTF2_LocationRef locationID,
					    OTF2_TimeStamp time,
					    void* userData,
					    OTF2_AttributeList* attributeList,
					    OTF2_IoHandleRef oldHandle,
					    OTF2_IoHandleRef newHandle,
					    OTF2_IoStatusFlag statusFlags) {
  OTF2_PRINT_GENERIC("IO_DUPLICATE_HANDLE", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}
  

OTF2_CallbackCode print_io_change_status_flags(OTF2_LocationRef locationID,
					       OTF2_TimeStamp time,
					       void* userData,
					       OTF2_AttributeList* attributeList,
					       OTF2_IoHandleRef handle,
					       OTF2_IoStatusFlag statusFlags) {
  OTF2_PRINT_GENERIC("IO_CHANGE_STATUS_FLAGS", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}
  

OTF2_CallbackCode print_io_operation_begin(OTF2_LocationRef locationID,
					   OTF2_TimeStamp time,
					   void* userData,
					   OTF2_AttributeList* attributeList,
					   OTF2_IoHandleRef handle,
					   OTF2_IoOperationMode mode,
					   OTF2_IoOperationFlag operationFlags,
					   uint64_t bytesRequest,
					   uint64_t matchingId) {
  OTF2_PRINT_GENERIC("IO_OPERATION_BEGIN", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}
  

OTF2_CallbackCode print_io_operation_issued(OTF2_LocationRef locationID,
					    OTF2_TimeStamp time,
					    void* userData,
					    OTF2_AttributeList* attributeList,
					    OTF2_IoHandleRef handle,
					    uint64_t matchingId) {
  OTF2_PRINT_GENERIC("IO_OPERATION_ISSUED", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_io_operation_test(OTF2_LocationRef locationID,
					  OTF2_TimeStamp time,
					  void* userData,
					  OTF2_AttributeList* attributeList,
					  OTF2_IoHandleRef handle,
					  uint64_t matchingId) {
  OTF2_PRINT_GENERIC("IO_OPERATION_TEST", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_io_delete_file(OTF2_LocationRef locationID,
				       OTF2_TimeStamp time,
				       void* userData,
				       OTF2_AttributeList* attributeList,
				       OTF2_IoParadigmRef ioParadigm,
				       OTF2_IoFileRef file) {
  OTF2_PRINT_GENERIC("IO_DELETE_FILE", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_io_seek(OTF2_LocationRef locationID,
				OTF2_TimeStamp time,
				void* userData,
				OTF2_AttributeList* attributeList,
				OTF2_IoHandleRef handle,
				int64_t offsetRequest,
				OTF2_IoSeekOption whence,
				uint64_t offsetResult) {
  OTF2_PRINT_GENERIC("IO_SEEK", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_io_destroy_handle(OTF2_LocationRef locationID,
					  OTF2_TimeStamp time,
					  void* userData,
					  OTF2_AttributeList* attributeList,
					  OTF2_IoHandleRef handle) {
  OTF2_PRINT_GENERIC("IO_DESTROY_HANDLE", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_io_operation_complete(OTF2_LocationRef locationID,
					      OTF2_TimeStamp time,
					      void* userData,
					      OTF2_AttributeList* attributeList,
					      OTF2_IoHandleRef handle,
					      uint64_t bytesResult,
					      uint64_t matchingId) {
  OTF2_PRINT_GENERIC("IO_OPERATION_COMPLETE", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_io_operation_cancelled(OTF2_LocationRef locationID,
					       OTF2_TimeStamp time,
					       void* userData,
					       OTF2_AttributeList* attributeList,
					       OTF2_IoHandleRef handle,
					       uint64_t matchingId) {
  OTF2_PRINT_GENERIC("IO_OPERATION_CANCELLED", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_io_acquire_lock(OTF2_LocationRef locationID,
					OTF2_TimeStamp time,
					void* userData,
					OTF2_AttributeList* attributeList,
					OTF2_IoHandleRef handle,
					OTF2_LockType lockType) {
  OTF2_PRINT_GENERIC("IO_ACQUIRE_LOCK", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_io_release_lock(OTF2_LocationRef locationID,
					OTF2_TimeStamp time,
					void* userData,
					OTF2_AttributeList* attributeList,
					OTF2_IoHandleRef handle,
					OTF2_LockType lockType) {
  OTF2_PRINT_GENERIC("IO_RELEASE_LOCK", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_io_try_lock(OTF2_LocationRef locationID,
				    OTF2_TimeStamp time,
				    void* userData,
				    OTF2_AttributeList* attributeList,
				    OTF2_IoHandleRef handle,
				    OTF2_LockType lockType) {
  OTF2_PRINT_GENERIC("IO_TRY_LOCK", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_program_begin(OTF2_LocationRef locationID,
				      OTF2_TimeStamp time,
				      void* userData,
				      OTF2_AttributeList* attributeList,
				      OTF2_StringRef programName,
				      uint32_t numberOfArguments,
				      const OTF2_StringRef* programArguments) {
  OTF2_PRINT_GENERIC("PROGRAM_BEGIN", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_program_end(OTF2_LocationRef locationID,
				    OTF2_TimeStamp time,
				    void* userData,
				    OTF2_AttributeList* attributeList,
				    int64_t exitStatus) {
  OTF2_PRINT_GENERIC("PROGRAM_END", locationID, time, "");
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_non_blocking_collective_request(
							OTF2_LocationRef locationID,
							OTF2_TimeStamp time,
							void* userData,
							OTF2_AttributeList* attributeList,
							uint64_t requestID) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "requestID: %p", (void*)requestID);
  OTF2_PRINT_GENERIC("NON_BLOCKING_COLLECTIVE_REQUEST", locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_non_blocking_collective_complete(
							 OTF2_LocationRef locationID,
							 OTF2_TimeStamp time,
							 void* userData,
							 OTF2_AttributeList* attributeList,
							 OTF2_CollectiveOp collectiveOp,
							 OTF2_CommRef communicator,
							 uint32_t root,
							 uint64_t sizeSent,
							 uint64_t sizeReceived,
							 uint64_t requestID) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "op: %d, comm: %x, root: %d, sent: %lu, recved:%lu, requestID: %p",
	   collectiveOp, communicator, root, sizeSent, sizeReceived, (void*)requestID);
  OTF2_PRINT_GENERIC("NON_BLOCKING_COLLECTIVE_COMPLETE", locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_comm_create(OTF2_LocationRef locationID,
				    OTF2_TimeStamp time,
				    void* userData,
				    OTF2_AttributeList* attributeList,
				    OTF2_CommRef communicator) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "comm: %x", communicator);
  OTF2_PRINT_GENERIC("COMM_CREATE", locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_comm_destroy(OTF2_LocationRef locationID,
				     OTF2_TimeStamp time,
				     void* userData,
				     OTF2_AttributeList* attributeList,
				     OTF2_CommRef communicator) {
  char buffer[STRING_MAX_LEN];
  snprintf(buffer, STRING_MAX_LEN, "comm: %x", communicator);
  OTF2_PRINT_GENERIC("COMM_DESTROY", locationID, time, buffer);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_GlobalEvtReaderCallbacks* otf2_print_create_global_evt_callbacks(OTF2_Reader *reader) {
  OTF2_GlobalEvtReaderCallbacks* evt_callbacks = OTF2_GlobalEvtReaderCallbacks_New();

  OTF2_GlobalEvtReaderCallbacks_SetEnterCallback( evt_callbacks, print_enter );
  OTF2_GlobalEvtReaderCallbacks_SetLeaveCallback( evt_callbacks, print_leave );
  OTF2_GlobalEvtReaderCallbacks_SetMpiSendCallback( evt_callbacks, print_mpi_send );
  OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCallback( evt_callbacks, print_mpi_isend );
  OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCompleteCallback( evt_callbacks, print_mpi_isend_complete );
  OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvRequestCallback( evt_callbacks, print_mpi_irecv_request );
  OTF2_GlobalEvtReaderCallbacks_SetMpiRecvCallback( evt_callbacks, print_mpi_recv );
  OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvCallback( evt_callbacks, print_mpi_irecv );
  OTF2_GlobalEvtReaderCallbacks_SetMpiRequestTestCallback( evt_callbacks, print_mpi_request_test );
  OTF2_GlobalEvtReaderCallbacks_SetMpiRequestCancelledCallback( evt_callbacks, print_mpi_request_cancelled );
  OTF2_GlobalEvtReaderCallbacks_SetMpiCollectiveBeginCallback( evt_callbacks, print_mpi_collective_begin );
  OTF2_GlobalEvtReaderCallbacks_SetMpiCollectiveEndCallback( evt_callbacks, print_mpi_collective_end );
  OTF2_GlobalEvtReaderCallbacks_SetOmpForkCallback( evt_callbacks, print_omp_fork );
  OTF2_GlobalEvtReaderCallbacks_SetOmpJoinCallback( evt_callbacks, print_omp_join );
  OTF2_GlobalEvtReaderCallbacks_SetOmpAcquireLockCallback( evt_callbacks, print_omp_acquire_lock );
  OTF2_GlobalEvtReaderCallbacks_SetOmpReleaseLockCallback( evt_callbacks, print_omp_release_lock );
  OTF2_GlobalEvtReaderCallbacks_SetOmpTaskCreateCallback( evt_callbacks, print_omp_task_create );
  OTF2_GlobalEvtReaderCallbacks_SetOmpTaskSwitchCallback( evt_callbacks, print_omp_task_switch );
  OTF2_GlobalEvtReaderCallbacks_SetOmpTaskCompleteCallback( evt_callbacks, print_omp_task_complete );
  OTF2_GlobalEvtReaderCallbacks_SetThreadForkCallback( evt_callbacks, print_thread_fork );
  OTF2_GlobalEvtReaderCallbacks_SetThreadJoinCallback( evt_callbacks, print_thread_join );
  OTF2_GlobalEvtReaderCallbacks_SetThreadTeamBeginCallback( evt_callbacks, print_thread_team_begin );
  OTF2_GlobalEvtReaderCallbacks_SetThreadTeamEndCallback( evt_callbacks, print_thread_team_end );
  OTF2_GlobalEvtReaderCallbacks_SetThreadAcquireLockCallback( evt_callbacks, print_thread_acquire_lock );
  OTF2_GlobalEvtReaderCallbacks_SetThreadReleaseLockCallback( evt_callbacks, print_thread_release_lock );
  OTF2_GlobalEvtReaderCallbacks_SetThreadTaskCreateCallback( evt_callbacks, print_thread_task_create );
  OTF2_GlobalEvtReaderCallbacks_SetThreadTaskSwitchCallback( evt_callbacks, print_thread_task_switch );
  OTF2_GlobalEvtReaderCallbacks_SetThreadTaskCompleteCallback( evt_callbacks, print_thread_task_complete );
  OTF2_GlobalEvtReaderCallbacks_SetThreadCreateCallback( evt_callbacks, print_thread_create );
  OTF2_GlobalEvtReaderCallbacks_SetThreadBeginCallback( evt_callbacks, print_thread_begin );
  OTF2_GlobalEvtReaderCallbacks_SetThreadWaitCallback( evt_callbacks, print_thread_wait );
  OTF2_GlobalEvtReaderCallbacks_SetThreadEndCallback( evt_callbacks, print_thread_end );
  OTF2_GlobalEvtReaderCallbacks_SetIoCreateHandleCallback( evt_callbacks, print_io_create_handle );
  OTF2_GlobalEvtReaderCallbacks_SetIoDestroyHandleCallback( evt_callbacks, print_io_destroy_handle );
  OTF2_GlobalEvtReaderCallbacks_SetIoDuplicateHandleCallback( evt_callbacks, print_io_duplicate_handle );
  OTF2_GlobalEvtReaderCallbacks_SetIoSeekCallback( evt_callbacks, print_io_seek );
  OTF2_GlobalEvtReaderCallbacks_SetIoChangeStatusFlagsCallback( evt_callbacks, print_io_change_status_flags );
  OTF2_GlobalEvtReaderCallbacks_SetIoDeleteFileCallback( evt_callbacks, print_io_delete_file );
  OTF2_GlobalEvtReaderCallbacks_SetIoOperationBeginCallback( evt_callbacks, print_io_operation_begin );
  OTF2_GlobalEvtReaderCallbacks_SetIoOperationTestCallback( evt_callbacks, print_io_operation_test );
  OTF2_GlobalEvtReaderCallbacks_SetIoOperationIssuedCallback( evt_callbacks, print_io_operation_issued );
  OTF2_GlobalEvtReaderCallbacks_SetIoOperationCompleteCallback( evt_callbacks, print_io_operation_complete );
  OTF2_GlobalEvtReaderCallbacks_SetIoOperationCancelledCallback( evt_callbacks, print_io_operation_cancelled );
  OTF2_GlobalEvtReaderCallbacks_SetIoAcquireLockCallback( evt_callbacks, print_io_acquire_lock );
  OTF2_GlobalEvtReaderCallbacks_SetIoReleaseLockCallback( evt_callbacks, print_io_release_lock );
  OTF2_GlobalEvtReaderCallbacks_SetIoTryLockCallback( evt_callbacks, print_io_try_lock );
  OTF2_GlobalEvtReaderCallbacks_SetProgramBeginCallback( evt_callbacks, print_program_begin );
  OTF2_GlobalEvtReaderCallbacks_SetProgramEndCallback( evt_callbacks, print_program_end );
  OTF2_GlobalEvtReaderCallbacks_SetNonBlockingCollectiveRequestCallback( evt_callbacks, print_non_blocking_collective_request );
  OTF2_GlobalEvtReaderCallbacks_SetNonBlockingCollectiveCompleteCallback( evt_callbacks, print_non_blocking_collective_complete );
  OTF2_GlobalEvtReaderCallbacks_SetCommCreateCallback( evt_callbacks, print_comm_create );
  OTF2_GlobalEvtReaderCallbacks_SetCommDestroyCallback( evt_callbacks, print_comm_destroy );
  
  return evt_callbacks;
}

int main(int argc, char** argv) {
  if(argc<2) {
    fprintf(stderr, "usage: %s trace_file\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char* trace_file = argv[1];
  OTF2_Reader* reader = OTF2_Reader_Open(trace_file);

  /* Define definition callbacks. */
  OTF2_GlobalDefReaderCallbacks* def_callbacks = otf2_print_define_global_def_callbacks(reader);

  /* Get number of locations from the anchor file. */
  uint64_t num_locations = 0;
  OTF2_ErrorCode status = OTF2_SUCCESS;
  status = OTF2_Reader_GetNumberOfLocations( reader, &num_locations );
  check_status(status, "OTF2_Reader_GetNumberOfLocations failed. Number of locations: %" PRIu64,
	       num_locations );

  struct otf2_print_user_data user_data;
  memset(&user_data, 0, sizeof(user_data));
  /* Read global definitions. */
  uint64_t              definitions_read  = 0;
  OTF2_GlobalDefReader* global_def_reader = OTF2_Reader_GetGlobalDefReader( reader );
  check_pointer(global_def_reader, "Create global definition reader handle: failed\n");

  status = OTF2_Reader_RegisterGlobalDefCallbacks( reader, global_def_reader,
						   def_callbacks,
						   &user_data );
  check_status( status, "Register global definition callbacks." );

  status = OTF2_Reader_ReadGlobalDefinitions( reader, global_def_reader,
					      OTF2_UNDEFINED_UINT64,
					      &definitions_read );
  check_status( status, "Read global definitions. Number of definitions: %" PRIu64,
		definitions_read );

  OTF2_Reader_CloseGlobalDefReader( reader,
				    global_def_reader );


  OTF2_DefReaderCallbacks* local_def_callbacks = OTF2_DefReaderCallbacks_New();
  check_pointer( def_callbacks, "Create global definition callback handle." );

  for ( size_t i = 0; i < user_data.nb_locations; i++ ) {
    uint64_t* location_item = &user_data.locations[i];
    status = OTF2_Reader_SelectLocation( reader, *location_item );
    check_status( status, "Select location to read." );
  }

  status = OTF2_Reader_OpenDefFiles( reader );
  check_status( status, "Open local definition files for reading." );

  status = OTF2_Reader_OpenEvtFiles( reader );
  check_status( status, "Open event files for reading." );

    for ( size_t i = 0; i < user_data.nb_locations; i++ )
    {
        uint64_t* location_item      = &user_data.locations[i];
        uint64_t  locationIdentifier = *location_item;

	OTF2_EvtReader* evt_reader = OTF2_Reader_GetEvtReader( reader,
                                                                   locationIdentifier );
	check_pointer( evt_reader, "Create local event reader for location %" PRIu64 ".",
		       locationIdentifier );

	OTF2_DefReader* def_reader = OTF2_Reader_GetDefReader( reader,
							       locationIdentifier );
	/* a local def file is not mandatory */
	if ( def_reader )
	  {
	    status = OTF2_Reader_RegisterDefCallbacks( reader,
						       def_reader,
						       local_def_callbacks,
						       &locationIdentifier );
	    check_status( status, "Register local definition callbacks." );

	    uint64_t definitions_read = 0;
	    status = OTF2_SUCCESS;
	    do
	      {
		uint64_t def_reads = 0;
		status = OTF2_Reader_ReadAllLocalDefinitions( reader,
							      def_reader,
							      &def_reads );
		definitions_read += def_reads;

		/* continue reading, if we have a duplicate mapping table */
		if ( OTF2_ERROR_DUPLICATE_MAPPING_TABLE != status )
		  {
		    break;
		  }
	      }
	    while ( true );
	    check_status( status,
			  "Read %" PRIu64 " definitions for location %" PRIu64,
			  definitions_read,
			  locationIdentifier );

	    /* Close def reader, it is no longer useful and occupies memory */
	    status = OTF2_Reader_CloseDefReader( reader, def_reader );
	    check_status( status, "Close local definition reader." );
	  }
    }

    OTF2_DefReaderCallbacks_Delete( local_def_callbacks );
    status = OTF2_Reader_CloseDefFiles( reader );
    check_status( status, "Close local definition files for reading." );

    OTF2_Reader_Close( reader );


    /* ___ Read Event Records ____________________________________________________*/


    /* Add a nice table header. */
    printf( "=== Events =====================================================================\n" );

    /* Define event callbacks. */
    OTF2_GlobalEvtReaderCallbacks* evt_callbacks = otf2_print_create_global_evt_callbacks(reader);

    /* Get global event reader. */
    OTF2_GlobalEvtReader* global_evt_reader = OTF2_Reader_GetGlobalEvtReader( reader );
    check_pointer( global_evt_reader, "Create global event reader." );


    /* Register the above defined callbacks to the global event reader. */
    status = OTF2_Reader_RegisterGlobalEvtCallbacks( reader,
						     global_evt_reader,
						     evt_callbacks,
						     &user_data );
    check_status( status, "Register global event callbacks." );
    OTF2_GlobalEvtReaderCallbacks_Delete( evt_callbacks );
    /* Read until events are all read. */
    uint64_t otf2_STEP = 1;
    uint64_t events_read = otf2_STEP;

    printf( "%-*s %15s %20s  Attributes\n",
	    otf2_EVENT_COLUMN_WIDTH, "Event", "Location", "Timestamp" );
    printf( "--------------------------------------------------------------------------------\n" );
    while ( events_read > 0 )  {
        status = OTF2_Reader_ReadGlobalEvents( reader,
                                               global_evt_reader,
                                               1,
                                               &events_read );
        check_status( status, "Read %" PRIu64 " events.", events_read );
    }
    status = OTF2_Reader_CloseGlobalEvtReader( reader,
                                               global_evt_reader );
    check_status( status, "Close global definition reader." );
    status = OTF2_Reader_CloseEvtFiles( reader );
    check_status( status, "Close event files for reading." );
    

    return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
