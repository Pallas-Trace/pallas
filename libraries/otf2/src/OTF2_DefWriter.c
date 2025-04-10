#include <stdio.h>
#include <stdlib.h>

#include "pallas/pallas.h"
#include "pallas/pallas_write.h"
#include "otf2/OTF2_DefWriter.h"
#include "otf2/otf2.h"

OTF2_ErrorCode OTF2_DefWriter_GetLocationID(const OTF2_DefWriter* writer, OTF2_LocationRef* location) {
  *location = writer->locationRef;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_DefWriter_WriteMappingTable(OTF2_DefWriter* writer,
                                                OTF2_MappingType mappingType,
                                                const OTF2_IdMap* idMap) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteClockOffset(OTF2_DefWriter* writer,
                                               OTF2_TimeStamp time,
                                               int64_t offset,
                                               double standardDeviation) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteString(OTF2_DefWriter* writer, OTF2_StringRef self, const char* string) {
  pallas_archive_register_string(writer->archive, self, string);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_DefWriter_WriteAttribute(OTF2_DefWriter* writer,
                                             OTF2_AttributeRef self,
                                             OTF2_StringRef name,
                                             OTF2_StringRef description,
                                             OTF2_Type type) {
  pallas_archive_register_attribute(writer->archive, self, name, description, OTF2_PALLAS_TYPE(type));
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_DefWriter_WriteSystemTreeNode(OTF2_DefWriter* writer,
                                                  OTF2_SystemTreeNodeRef self,
                                                  OTF2_StringRef name,
                                                  OTF2_StringRef className,
                                                  OTF2_SystemTreeNodeRef parent) {
  NOT_IMPLEMENTED;
}

ThreadId _otf_register_location(OTF2_LocationRef ref);
ThreadId _otf_get_thread_id(OTF2_LocationRef ref);
LocationGroupId _otf_register_location_group(OTF2_LocationGroupRef ref);
LocationGroupId _otf_get_location_group_id(OTF2_LocationGroupRef ref);

OTF2_ErrorCode OTF2_DefWriter_WriteLocationGroup(OTF2_DefWriter* writer,
                                                 OTF2_LocationGroupRef self,
                                                 OTF2_StringRef name,
                                                 OTF2_LocationGroupType locationGroupType,
                                                 OTF2_SystemTreeNodeRef systemTreeParent,
                                                 OTF2_LocationGroupRef creatingLocationGroup) {
  LocationGroupId lg_id = _otf_register_location_group(self);
  LocationGroupId parent_id = _otf_get_location_group_id(creatingLocationGroup);
  pallas_archive_define_location_group(writer->archive, lg_id, name, parent_id);

  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_DefWriter_WriteLocation(OTF2_DefWriter* writer,
                                            OTF2_LocationRef self,
                                            OTF2_StringRef name,
                                            OTF2_LocationType locationType,
                                            uint64_t numberOfEvents,
                                            OTF2_LocationGroupRef locationGroup) {
  ThreadId thread_id = _otf_register_location(self);
  static int first_call = 1;
  if (first_call) {
    if (writer->archive->id == 0)
      writer->archive->id = locationGroup;
  }
  pallas_archive_define_location(writer->archive, thread_id, name, locationGroup);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_DefWriter_WriteRegion(OTF2_DefWriter* writer,
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
  pallas_archive_register_region(writer->archive, self, name);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_DefWriter_WriteCallsite(OTF2_DefWriter* writer,
                                            OTF2_CallsiteRef self,
                                            OTF2_StringRef sourceFile,
                                            uint32_t lineNumber,
                                            OTF2_RegionRef enteredRegion,
                                            OTF2_RegionRef leftRegion) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteCallpath(OTF2_DefWriter* writer,
                                            OTF2_CallpathRef self,
                                            OTF2_CallpathRef parent,
                                            OTF2_RegionRef region) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteGroup(OTF2_DefWriter* writer,
                                         OTF2_GroupRef self,
                                         OTF2_StringRef name,
                                         OTF2_GroupType groupType,
                                         OTF2_Paradigm paradigm,
                                         OTF2_GroupFlag groupFlags,
                                         uint32_t numberOfMembers,
                                         const uint64_t* members) {
  pallas_archive_register_group(writer->archive, self, name, numberOfMembers, members);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_DefWriter_WriteMetricMember(OTF2_DefWriter* writer,
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

OTF2_ErrorCode OTF2_DefWriter_WriteMetricClass(OTF2_DefWriter* writer,
                                               OTF2_MetricRef self,
                                               uint8_t numberOfMetrics,
                                               const OTF2_MetricMemberRef* metricMembers,
                                               OTF2_MetricOccurrence metricOccurrence,
                                               OTF2_RecorderKind recorderKind) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteMetricInstance(OTF2_DefWriter* writer,
                                                  OTF2_MetricRef self,
                                                  OTF2_MetricRef metricClass,
                                                  OTF2_LocationRef recorder,
                                                  OTF2_MetricScope metricScope,
                                                  uint64_t scope) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteComm(OTF2_DefWriter* writer,
                                        OTF2_CommRef self,
                                        OTF2_StringRef name,
                                        OTF2_GroupRef group,
                                        OTF2_CommRef parent,
                                        OTF2_CommFlag flags) {
  pallas_archive_register_comm(writer->archive, self, name, group, parent);

  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_DefWriter_WriteParameter(OTF2_DefWriter* writer,
                                             OTF2_ParameterRef self,
                                             OTF2_StringRef name,
                                             OTF2_ParameterType parameterType) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteRmaWin(OTF2_DefWriter* writer,
                                          OTF2_RmaWinRef self,
                                          OTF2_StringRef name,
                                          OTF2_CommRef comm,
                                          OTF2_RmaWinFlag flags) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteMetricClassRecorder(OTF2_DefWriter* writer,
                                                       OTF2_MetricRef metric,
                                                       OTF2_LocationRef recorder) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteSystemTreeNodeProperty(OTF2_DefWriter* writer,
                                                          OTF2_SystemTreeNodeRef systemTreeNode,
                                                          OTF2_StringRef name,
                                                          OTF2_Type type,
                                                          OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteSystemTreeNodeDomain(OTF2_DefWriter* writer,
                                                        OTF2_SystemTreeNodeRef systemTreeNode,
                                                        OTF2_SystemTreeDomain systemTreeDomain) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteLocationGroupProperty(OTF2_DefWriter* writer,
                                                         OTF2_LocationGroupRef locationGroup,
                                                         OTF2_StringRef name,
                                                         OTF2_Type type,
                                                         OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteLocationProperty(OTF2_DefWriter* writer,
                                                    OTF2_LocationRef location,
                                                    OTF2_StringRef name,
                                                    OTF2_Type type,
                                                    OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteCartDimension(OTF2_DefWriter* writer,
                                                 OTF2_CartDimensionRef self,
                                                 OTF2_StringRef name,
                                                 uint32_t size,
                                                 OTF2_CartPeriodicity cartPeriodicity) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteCartTopology(OTF2_DefWriter* writer,
                                                OTF2_CartTopologyRef self,
                                                OTF2_StringRef name,
                                                OTF2_CommRef communicator,
                                                uint8_t numberOfDimensions,
                                                const OTF2_CartDimensionRef* cartDimensions) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteCartCoordinate(OTF2_DefWriter* writer,
                                                  OTF2_CartTopologyRef cartTopology,
                                                  uint32_t rank,
                                                  uint8_t numberOfDimensions,
                                                  const uint32_t* coordinates) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteSourceCodeLocation(OTF2_DefWriter* writer,
                                                      OTF2_SourceCodeLocationRef self,
                                                      OTF2_StringRef file,
                                                      uint32_t lineNumber) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteCallingContext(OTF2_DefWriter* writer,
                                                  OTF2_CallingContextRef self,
                                                  OTF2_RegionRef region,
                                                  OTF2_SourceCodeLocationRef sourceCodeLocation,
                                                  OTF2_CallingContextRef parent) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteCallingContextProperty(OTF2_DefWriter* writer,
                                                          OTF2_CallingContextRef callingContext,
                                                          OTF2_StringRef name,
                                                          OTF2_Type type,
                                                          OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteInterruptGenerator(OTF2_DefWriter* writer,
                                                      OTF2_InterruptGeneratorRef self,
                                                      OTF2_StringRef name,
                                                      OTF2_InterruptGeneratorMode interruptGeneratorMode,
                                                      OTF2_Base base,
                                                      int64_t exponent,
                                                      uint64_t period) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteIoFileProperty(OTF2_DefWriter* writer,
                                                  OTF2_IoFileRef ioFile,
                                                  OTF2_StringRef name,
                                                  OTF2_Type type,
                                                  OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteIoRegularFile(OTF2_DefWriter* writer,
                                                 OTF2_IoFileRef self,
                                                 OTF2_StringRef name,
                                                 OTF2_SystemTreeNodeRef scope) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteIoDirectory(OTF2_DefWriter* writer,
                                               OTF2_IoFileRef self,
                                               OTF2_StringRef name,
                                               OTF2_SystemTreeNodeRef scope) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteIoHandle(OTF2_DefWriter* writer,
                                            OTF2_IoHandleRef self,
                                            OTF2_StringRef name,
                                            OTF2_IoFileRef file,
                                            OTF2_IoParadigmRef ioParadigm,
                                            OTF2_IoHandleFlag ioHandleFlags,
                                            OTF2_CommRef comm,
                                            OTF2_IoHandleRef parent) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteIoPreCreatedHandleState(OTF2_DefWriter* writer,
                                                           OTF2_IoHandleRef ioHandle,
                                                           OTF2_IoAccessMode mode,
                                                           OTF2_IoStatusFlag statusFlags) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteCallpathParameter(OTF2_DefWriter* writer,
                                                     OTF2_CallpathRef callpath,
                                                     OTF2_ParameterRef parameter,
                                                     OTF2_Type type,
                                                     OTF2_AttributeValue value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_DefWriter_WriteInterComm(OTF2_DefWriter* writer,
                                             OTF2_CommRef self,
                                             OTF2_StringRef name,
                                             OTF2_GroupRef groupA,
                                             OTF2_GroupRef groupB,
                                             OTF2_CommRef commonCommunicator,
                                             OTF2_CommFlag flags) {
  NOT_IMPLEMENTED;
}
