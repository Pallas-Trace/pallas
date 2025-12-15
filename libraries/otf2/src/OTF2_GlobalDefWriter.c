#include <stdio.h>
#include <stdlib.h>

#include "pallas/pallas.h"
#include "pallas/utils/pallas_log.h"
#include "otf2/OTF2_GlobalDefWriter.h"
#include "otf2/otf2.h"

OTF2_ErrorCode OTF2_GlobalDefWriter_GetNumberOfDefinitions(OTF2_GlobalDefWriter* writerHandle,
                                                           uint64_t* numberOfDefinitions) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_GetNumberOfLocations(OTF2_GlobalDefWriter* writerHandle,
                                                         uint64_t* numberOfLocations) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteClockProperties(OTF2_GlobalDefWriter* writerHandle,
                                                         uint64_t timerResolution,
                                                         uint64_t globalOffset,
                                                         uint64_t traceLength,
                                                         uint64_t realtimeTimestamp) {
  TO_BE_IMPLEMENTED;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteParadigm(OTF2_GlobalDefWriter* writerHandle,
                                                  OTF2_Paradigm paradigm,
                                                  OTF2_StringRef name,
                                                  OTF2_ParadigmClass paradigmClass) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteParadigmProperty(OTF2_GlobalDefWriter* writerHandle,
                                                          OTF2_Paradigm paradigm,
                                                          OTF2_ParadigmProperty property,
                                                          OTF2_Type type,
                                                          OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteIoParadigm(OTF2_GlobalDefWriter* writerHandle,
                                                    OTF2_IoParadigmRef self,
                                                    OTF2_StringRef identification,
                                                    OTF2_StringRef name,
                                                    OTF2_IoParadigmClass ioParadigmClass,
                                                    OTF2_IoParadigmFlag ioParadigmFlags,
                                                    uint8_t numberOfProperties,
                                                    const OTF2_IoParadigmProperty* properties,
                                                    const OTF2_Type* types,
                                                    const OTF2_AttributeValue* values) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteString(OTF2_GlobalDefWriter* writerHandle,
                                                OTF2_StringRef self,
                                                const char* string) {
  //  NOT_IMPLEMENTED;
  pallas_global_archive_register_string(writerHandle->archive, self, string);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteAttribute(OTF2_GlobalDefWriter* writerHandle,
                                                   OTF2_AttributeRef self,
                                                   OTF2_StringRef name,
                                                   OTF2_StringRef description,
                                                   OTF2_Type type) {
  pallas_global_archive_register_attribute(writerHandle->archive, self, name, description, OTF2_PALLAS_TYPE(type));
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteSystemTreeNode(OTF2_GlobalDefWriter* writerHandle,
                                                        OTF2_SystemTreeNodeRef self,
                                                        OTF2_StringRef name,
                                                        OTF2_StringRef className,
                                                        OTF2_SystemTreeNodeRef parent) {
  //  NOT_IMPLEMENTED;
  return OTF2_SUCCESS;
}

struct location_group_map {
  LocationGroupId pallas_location_group;
  OTF2_LocationGroupRef location_group;
};

struct location_group_map* location_group_map = NULL;
int nb_location_group = 0;

LocationGroupId next_location_group_id(int ref) {
  /* TODO: buggy ?  */
  return ref;
  static LocationGroupId next_id = PALLAS_LOCATION_GROUP_ID_INVALID;  // todo: pb with MPI ?
  if (next_id == PALLAS_LOCATION_GROUP_ID_INVALID)
    next_id = ref;
  return next_id++;
}

ThreadId next_thread_id(int ref) {
  /* TODO: buggy ?  */
  return ref;
  static ThreadId next_id = PALLAS_THREAD_ID_INVALID;  // todo: pb with MPI ?
  if (next_id == PALLAS_THREAD_ID_INVALID)
    next_id = ref;
  return next_id++;
}

LocationGroupId _otf_get_location_group_id(OTF2_LocationGroupRef ref) {
  for (int i = 0; i < nb_location_group; i++) {
    if (location_group_map[i].location_group == ref)
      return location_group_map[i].pallas_location_group;
  }
  return PALLAS_LOCATION_GROUP_ID_INVALID;
}

LocationGroupId _otf_register_location_group(OTF2_LocationGroupRef ref) {
  pallas_assert(_otf_get_location_group_id(ref) == PALLAS_LOCATION_GROUP_ID_INVALID);
  int index = nb_location_group++;
  location_group_map = realloc(location_group_map, sizeof(struct location_group_map) * nb_location_group);
  location_group_map[index].pallas_location_group = next_location_group_id(ref);
  location_group_map[index].location_group = ref;
  return location_group_map[index].pallas_location_group;
}

struct location_map {
  ThreadId thread_id;
  OTF2_LocationRef location;
};

struct location_map* location_map = NULL;
int nb_location = 0;

ThreadId _otf_get_location_id(OTF2_LocationRef ref) {
  for (int i = 0; i < nb_location; i++) {
    if (location_map[i].location == ref)
      return location_map[i].thread_id;
  }
  return PALLAS_THREAD_ID_INVALID;
}

ThreadId _otf_register_location(OTF2_LocationRef ref) {
  pallas_assert(_otf_get_location_id(ref) == PALLAS_THREAD_ID_INVALID);
  int index = nb_location++;
  location_map = realloc(location_map, sizeof(struct location_map) * nb_location);
  location_map[index].thread_id = next_thread_id(ref);
  location_map[index].location = ref;
  return location_map[index].thread_id;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteLocationGroup(OTF2_GlobalDefWriter* writerHandle,
                                                       OTF2_LocationGroupRef self,
                                                       OTF2_StringRef name,
                                                       OTF2_LocationGroupType locationGroupType,
                                                       OTF2_SystemTreeNodeRef systemTreeParent,
                                                       OTF2_LocationGroupRef creatingLocationGroup) {
  LocationGroupId lg_id = _otf_register_location_group(self);
  LocationGroupId parent_id = _otf_get_location_group_id(creatingLocationGroup);
  pallas_global_archive_define_location_group(writerHandle->archive, lg_id, name, parent_id);

  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteLocation(OTF2_GlobalDefWriter* writerHandle,
                                                  OTF2_LocationRef self,
                                                  OTF2_StringRef name,
                                                  OTF2_LocationType locationType,
                                                  uint64_t numberOfEvents,
                                                  OTF2_LocationGroupRef locationGroup) {
  ThreadId thread_id = _otf_register_location(self);
  LocationGroupId parent_id = _otf_get_location_group_id(locationGroup);
  pallas_global_archive_define_location(writerHandle->archive, thread_id, name, parent_id);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteRegion(OTF2_GlobalDefWriter* writerHandle,
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
  /* TODO (in Pallas)
   * - add a global writeRegion function
   * - uppon thread creation, copy the write region to the new thread regions
   * - when creating a global region, add it to the existing threads region
   */
  pallas_global_archive_register_region(writerHandle->archive, self, name);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteCallsite(OTF2_GlobalDefWriter* writerHandle,
                                                  OTF2_CallsiteRef self,
                                                  OTF2_StringRef sourceFile,
                                                  uint32_t lineNumber,
                                                  OTF2_RegionRef enteredRegion,
                                                  OTF2_RegionRef leftRegion) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteCallpath(OTF2_GlobalDefWriter* writerHandle,
                                                  OTF2_CallpathRef self,
                                                  OTF2_CallpathRef parent,
                                                  OTF2_RegionRef region) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteGroup(OTF2_GlobalDefWriter* writerHandle,
                                               OTF2_GroupRef self,
                                               OTF2_StringRef name,
                                               OTF2_GroupType groupType,
                                               OTF2_Paradigm paradigm,
                                               OTF2_GroupFlag groupFlags,
                                               uint32_t numberOfMembers,
                                               const uint64_t* members) {
  pallas_global_archive_register_group(writerHandle->archive, self, name, groupType, paradigm, numberOfMembers, members);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteMetricMember(OTF2_GlobalDefWriter* writerHandle,
                                                      OTF2_MetricMemberRef self,
                                                      OTF2_StringRef name,
                                                      OTF2_StringRef description,
                                                      OTF2_MetricType metricType,
                                                      OTF2_MetricMode metricMode,
                                                      OTF2_Type valueType,
                                                      OTF2_Base base,
                                                      int64_t exponent,
                                                      OTF2_StringRef unit) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteMetricClass(OTF2_GlobalDefWriter* writerHandle,
                                                     OTF2_MetricRef self,
                                                     uint8_t numberOfMetrics,
                                                     const OTF2_MetricMemberRef* metricMembers,
                                                     OTF2_MetricOccurrence metricOccurrence,
                                                     OTF2_RecorderKind recorderKind) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteMetricInstance(OTF2_GlobalDefWriter* writerHandle,
                                                        OTF2_MetricRef self,
                                                        OTF2_MetricRef metricClass,
                                                        OTF2_LocationRef recorder,
                                                        OTF2_MetricScope metricScope,
                                                        uint64_t scope) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteComm(OTF2_GlobalDefWriter* writerHandle,
                                              OTF2_CommRef self,
                                              OTF2_StringRef name,
                                              OTF2_GroupRef group,
                                              OTF2_CommRef parent,
                                              OTF2_CommFlag flags) {
  pallas_global_archive_register_comm(writerHandle->archive, self, name, group, parent);

  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteParameter(OTF2_GlobalDefWriter* writerHandle,
                                                   OTF2_ParameterRef self,
                                                   OTF2_StringRef name,
                                                   OTF2_ParameterType parameterType) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteRmaWin(OTF2_GlobalDefWriter* writerHandle,
                                                OTF2_RmaWinRef self,
                                                OTF2_StringRef name,
                                                OTF2_CommRef comm,
                                                OTF2_RmaWinFlag flags) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteMetricClassRecorder(OTF2_GlobalDefWriter* writerHandle,
                                                             OTF2_MetricRef metric,
                                                             OTF2_LocationRef recorder) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteSystemTreeNodeProperty(OTF2_GlobalDefWriter* writerHandle,
                                                                OTF2_SystemTreeNodeRef systemTreeNode,
                                                                OTF2_StringRef name,
                                                                OTF2_Type type,
                                                                OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteSystemTreeNodeDomain(OTF2_GlobalDefWriter* writerHandle,
                                                              OTF2_SystemTreeNodeRef systemTreeNode,
                                                              OTF2_SystemTreeDomain systemTreeDomain) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteLocationGroupProperty(OTF2_GlobalDefWriter* writerHandle,
                                                               OTF2_LocationGroupRef locationGroup,
                                                               OTF2_StringRef name,
                                                               OTF2_Type type,
                                                               OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteLocationProperty(OTF2_GlobalDefWriter* writerHandle,
                                                          OTF2_LocationRef location,
                                                          OTF2_StringRef name,
                                                          OTF2_Type type,
                                                          OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteCartDimension(OTF2_GlobalDefWriter* writerHandle,
                                                       OTF2_CartDimensionRef self,
                                                       OTF2_StringRef name,
                                                       uint32_t size,
                                                       OTF2_CartPeriodicity cartPeriodicity) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteCartTopology(OTF2_GlobalDefWriter* writerHandle,
                                                      OTF2_CartTopologyRef self,
                                                      OTF2_StringRef name,
                                                      OTF2_CommRef communicator,
                                                      uint8_t numberOfDimensions,
                                                      const OTF2_CartDimensionRef* cartDimensions) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteCartCoordinate(OTF2_GlobalDefWriter* writerHandle,
                                                        OTF2_CartTopologyRef cartTopology,
                                                        uint32_t rank,
                                                        uint8_t numberOfDimensions,
                                                        const uint32_t* coordinates) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteSourceCodeLocation(OTF2_GlobalDefWriter* writerHandle,
                                                            OTF2_SourceCodeLocationRef self,
                                                            OTF2_StringRef file,
                                                            uint32_t lineNumber) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteCallingContext(OTF2_GlobalDefWriter* writerHandle,
                                                        OTF2_CallingContextRef self,
                                                        OTF2_RegionRef region,
                                                        OTF2_SourceCodeLocationRef sourceCodeLocation,
                                                        OTF2_CallingContextRef parent) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteCallingContextProperty(OTF2_GlobalDefWriter* writerHandle,
                                                                OTF2_CallingContextRef callingContext,
                                                                OTF2_StringRef name,
                                                                OTF2_Type type,
                                                                OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteInterruptGenerator(OTF2_GlobalDefWriter* writerHandle,
                                                            OTF2_InterruptGeneratorRef self,
                                                            OTF2_StringRef name,
                                                            OTF2_InterruptGeneratorMode interruptGeneratorMode,
                                                            OTF2_Base base,
                                                            int64_t exponent,
                                                            uint64_t period) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteIoFileProperty(OTF2_GlobalDefWriter* writerHandle,
                                                        OTF2_IoFileRef ioFile,
                                                        OTF2_StringRef name,
                                                        OTF2_Type type,
                                                        OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteIoRegularFile(OTF2_GlobalDefWriter* writerHandle,
                                                       OTF2_IoFileRef self,
                                                       OTF2_StringRef name,
                                                       OTF2_SystemTreeNodeRef scope) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteIoDirectory(OTF2_GlobalDefWriter* writerHandle,
                                                     OTF2_IoFileRef self,
                                                     OTF2_StringRef name,
                                                     OTF2_SystemTreeNodeRef scope) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteIoHandle(OTF2_GlobalDefWriter* writerHandle,
                                                  OTF2_IoHandleRef self,
                                                  OTF2_StringRef name,
                                                  OTF2_IoFileRef file,
                                                  OTF2_IoParadigmRef ioParadigm,
                                                  OTF2_IoHandleFlag ioHandleFlags,
                                                  OTF2_CommRef comm,
                                                  OTF2_IoHandleRef parent) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteIoPreCreatedHandleState(OTF2_GlobalDefWriter* writerHandle,
                                                                 OTF2_IoHandleRef ioHandle,
                                                                 OTF2_IoAccessMode mode,
                                                                 OTF2_IoStatusFlag statusFlags) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteCallpathParameter(OTF2_GlobalDefWriter* writerHandle,
                                                           OTF2_CallpathRef callpath,
                                                           OTF2_ParameterRef parameter,
                                                           OTF2_Type type,
                                                           OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_GlobalDefWriter_WriteInterComm(OTF2_GlobalDefWriter* writerHandle,
                                                   OTF2_CommRef self,
                                                   OTF2_StringRef name,
                                                   OTF2_GroupRef groupA,
                                                   OTF2_GroupRef groupB,
                                                   OTF2_CommRef commonCommunicator,
                                                   OTF2_CommFlag flags) {
  NOT_IMPLEMENTED;
}
