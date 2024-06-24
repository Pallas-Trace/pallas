//
// Created by khatharsis on 06/06/24.
//
/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2013,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2014,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2014-2015,
 * Technische Universitaet Dresden, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 *
 */

static size_t
get_attribute_list( OTF2_EventSizeEstimator* estimator,
                   const string&            input,
                   size_t&                  pos );

static void
handle_get( OTF2_EventSizeEstimator* estimator,
           const string&            input,
           size_t&                  pos )
{
  string event_name = get_token( input, pos );
  string key        = event_name;
  if ( pos != string::npos )
  {
    key += input.substr( pos );
  }
  size_t size = 0;

  if ( event_name == "" )
  {
    die( "missing event specification for get command" );
  }
  else if ( event_name == "DefChunkSize" )
  {
    size = OTF2_EventSizeEstimator_GetDefChunkSize( estimator );
  }
  else if ( event_name == "Timestamp" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfTimestamp( estimator );
  }
  else if ( event_name == "AttributeList" )
  {
    size = get_attribute_list( estimator, input, pos );
  }
  else if ( event_name == "BufferFlush" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfBufferFlushEvent( estimator );
  }
  else if ( event_name == "MeasurementOnOff" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfMeasurementOnOffEvent( estimator );
  }
  else if ( event_name == "Enter" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfEnterEvent( estimator );
  }
  else if ( event_name == "Leave" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfLeaveEvent( estimator );
  }
  else if ( event_name == "MpiSend" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfMpiSendEvent( estimator );
  }
  else if ( event_name == "MpiIsend" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfMpiIsendEvent( estimator );
  }
  else if ( event_name == "MpiIsendComplete" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfMpiIsendCompleteEvent( estimator );
  }
  else if ( event_name == "MpiIrecvRequest" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfMpiIrecvRequestEvent( estimator );
  }
  else if ( event_name == "MpiRecv" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfMpiRecvEvent( estimator );
  }
  else if ( event_name == "MpiIrecv" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfMpiIrecvEvent( estimator );
  }
  else if ( event_name == "MpiRequestTest" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfMpiRequestTestEvent( estimator );
  }
  else if ( event_name == "MpiRequestCancelled" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfMpiRequestCancelledEvent( estimator );
  }
  else if ( event_name == "MpiCollectiveBegin" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfMpiCollectiveBeginEvent( estimator );
  }
  else if ( event_name == "MpiCollectiveEnd" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfMpiCollectiveEndEvent( estimator );
  }
  else if ( event_name == "OmpFork" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfOmpForkEvent( estimator );
  }
  else if ( event_name == "OmpJoin" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfOmpJoinEvent( estimator );
  }
  else if ( event_name == "OmpAcquireLock" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfOmpAcquireLockEvent( estimator );
  }
  else if ( event_name == "OmpReleaseLock" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfOmpReleaseLockEvent( estimator );
  }
  else if ( event_name == "OmpTaskCreate" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfOmpTaskCreateEvent( estimator );
  }
  else if ( event_name == "OmpTaskSwitch" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfOmpTaskSwitchEvent( estimator );
  }
  else if ( event_name == "OmpTaskComplete" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfOmpTaskCompleteEvent( estimator );
  }
  else if ( event_name == "Metric" )
  {
    string number_of_metrics_arg = get_token( input, pos );
    if ( number_of_metrics_arg == "" )
    {
      die( "missing numberOfMetrics parameter for Metric event" );
    }
    uint8_t number_of_metrics = strtol( number_of_metrics_arg.c_str(), NULL, 0 );

    size = OTF2_EventSizeEstimator_GetSizeOfMetricEvent( estimator,
                                                        number_of_metrics );
  }
  else if ( event_name == "ParameterString" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfParameterStringEvent( estimator );
  }
  else if ( event_name == "ParameterInt" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfParameterIntEvent( estimator );
  }
  else if ( event_name == "ParameterUnsignedInt" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfParameterUnsignedIntEvent( estimator );
  }
  else if ( event_name == "RmaWinCreate" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaWinCreateEvent( estimator );
  }
  else if ( event_name == "RmaWinDestroy" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaWinDestroyEvent( estimator );
  }
  else if ( event_name == "RmaCollectiveBegin" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaCollectiveBeginEvent( estimator );
  }
  else if ( event_name == "RmaCollectiveEnd" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaCollectiveEndEvent( estimator );
  }
  else if ( event_name == "RmaGroupSync" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaGroupSyncEvent( estimator );
  }
  else if ( event_name == "RmaRequestLock" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaRequestLockEvent( estimator );
  }
  else if ( event_name == "RmaAcquireLock" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaAcquireLockEvent( estimator );
  }
  else if ( event_name == "RmaTryLock" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaTryLockEvent( estimator );
  }
  else if ( event_name == "RmaReleaseLock" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaReleaseLockEvent( estimator );
  }
  else if ( event_name == "RmaSync" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaSyncEvent( estimator );
  }
  else if ( event_name == "RmaWaitChange" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaWaitChangeEvent( estimator );
  }
  else if ( event_name == "RmaPut" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaPutEvent( estimator );
  }
  else if ( event_name == "RmaGet" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaGetEvent( estimator );
  }
  else if ( event_name == "RmaAtomic" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaAtomicEvent( estimator );
  }
  else if ( event_name == "RmaOpCompleteBlocking" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaOpCompleteBlockingEvent( estimator );
  }
  else if ( event_name == "RmaOpCompleteNonBlocking" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaOpCompleteNonBlockingEvent( estimator );
  }
  else if ( event_name == "RmaOpTest" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaOpTestEvent( estimator );
  }
  else if ( event_name == "RmaOpCompleteRemote" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfRmaOpCompleteRemoteEvent( estimator );
  }
  else if ( event_name == "ThreadFork" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadForkEvent( estimator );
  }
  else if ( event_name == "ThreadJoin" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadJoinEvent( estimator );
  }
  else if ( event_name == "ThreadTeamBegin" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadTeamBeginEvent( estimator );
  }
  else if ( event_name == "ThreadTeamEnd" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadTeamEndEvent( estimator );
  }
  else if ( event_name == "ThreadAcquireLock" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadAcquireLockEvent( estimator );
  }
  else if ( event_name == "ThreadReleaseLock" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadReleaseLockEvent( estimator );
  }
  else if ( event_name == "ThreadTaskCreate" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadTaskCreateEvent( estimator );
  }
  else if ( event_name == "ThreadTaskSwitch" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadTaskSwitchEvent( estimator );
  }
  else if ( event_name == "ThreadTaskComplete" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadTaskCompleteEvent( estimator );
  }
  else if ( event_name == "ThreadCreate" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadCreateEvent( estimator );
  }
  else if ( event_name == "ThreadBegin" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadBeginEvent( estimator );
  }
  else if ( event_name == "ThreadWait" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadWaitEvent( estimator );
  }
  else if ( event_name == "ThreadEnd" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfThreadEndEvent( estimator );
  }
  else if ( event_name == "CallingContextEnter" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfCallingContextEnterEvent( estimator );
  }
  else if ( event_name == "CallingContextLeave" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfCallingContextLeaveEvent( estimator );
  }
  else if ( event_name == "CallingContextSample" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfCallingContextSampleEvent( estimator );
  }
  else if ( event_name == "IoCreateHandle" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoCreateHandleEvent( estimator );
  }
  else if ( event_name == "IoDestroyHandle" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoDestroyHandleEvent( estimator );
  }
  else if ( event_name == "IoDuplicateHandle" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoDuplicateHandleEvent( estimator );
  }
  else if ( event_name == "IoSeek" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoSeekEvent( estimator );
  }
  else if ( event_name == "IoChangeStatusFlags" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoChangeStatusFlagsEvent( estimator );
  }
  else if ( event_name == "IoDeleteFile" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoDeleteFileEvent( estimator );
  }
  else if ( event_name == "IoOperationBegin" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoOperationBeginEvent( estimator );
  }
  else if ( event_name == "IoOperationTest" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoOperationTestEvent( estimator );
  }
  else if ( event_name == "IoOperationIssued" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoOperationIssuedEvent( estimator );
  }
  else if ( event_name == "IoOperationComplete" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoOperationCompleteEvent( estimator );
  }
  else if ( event_name == "IoOperationCancelled" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoOperationCancelledEvent( estimator );
  }
  else if ( event_name == "IoAcquireLock" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoAcquireLockEvent( estimator );
  }
  else if ( event_name == "IoReleaseLock" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoReleaseLockEvent( estimator );
  }
  else if ( event_name == "IoTryLock" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfIoTryLockEvent( estimator );
  }
  else if ( event_name == "ProgramBegin" )
  {
    string number_of_arguments_arg = get_token( input, pos );
    if ( number_of_arguments_arg == "" )
    {
      die( "missing numberOfArguments parameter for ProgramBegin event" );
    }
    uint32_t number_of_arguments = strtol( number_of_arguments_arg.c_str(), NULL, 0 );

    size = OTF2_EventSizeEstimator_GetSizeOfProgramBeginEvent( estimator,
                                                              number_of_arguments );
  }
  else if ( event_name == "ProgramEnd" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfProgramEndEvent( estimator );
  }
  else if ( event_name == "NonBlockingCollectiveRequest" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfNonBlockingCollectiveRequestEvent( estimator );
  }
  else if ( event_name == "NonBlockingCollectiveComplete" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfNonBlockingCollectiveCompleteEvent( estimator );
  }
  else if ( event_name == "CommCreate" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfCommCreateEvent( estimator );
  }
  else if ( event_name == "CommDestroy" )
  {
    size = OTF2_EventSizeEstimator_GetSizeOfCommDestroyEvent( estimator );
  }
  else
  {
    die( "unknown event '" << key << "'" );
  }

  cout << key << " " << size << endl;
}

size_t
get_attribute_list( OTF2_EventSizeEstimator* estimator,
                   const string&            input,
                   size_t&                  pos )
{
  OTF2_AttributeList* attribute_list = OTF2_AttributeList_New();
  if ( !attribute_list )
  {
    die( "Out of memory for AttributeList" );
  }

  OTF2_AttributeRef attribute = 0;
  for ( string attribute_type_arg = get_token( input, pos );
       attribute_type_arg != "";
       attribute_type_arg = get_token( input, pos ) )
  {
    OTF2_Type           type;
    OTF2_AttributeValue attribute_value;

    if ( attribute_type_arg == "NONE" )
    {
      continue;
    }
    else if ( attribute_type_arg == "UINT8" )
    {
      type                  = OTF2_TYPE_UINT8;
      attribute_value.uint8 = 0;
    }
    else if ( attribute_type_arg == "UINT16" )
    {
      type                   = OTF2_TYPE_UINT16;
      attribute_value.uint16 = 0;
    }
    else if ( attribute_type_arg == "UINT32" )
    {
      type                   = OTF2_TYPE_UINT32;
      attribute_value.uint32 = 0;
    }
    else if ( attribute_type_arg == "UINT64" )
    {
      type                   = OTF2_TYPE_UINT64;
      attribute_value.uint64 = 0;
    }
    else if ( attribute_type_arg == "INT8" )
    {
      type                 = OTF2_TYPE_INT8;
      attribute_value.int8 = 0;
    }
    else if ( attribute_type_arg == "INT16" )
    {
      type                  = OTF2_TYPE_INT16;
      attribute_value.int16 = 0;
    }
    else if ( attribute_type_arg == "INT32" )
    {
      type                  = OTF2_TYPE_INT32;
      attribute_value.int32 = 0;
    }
    else if ( attribute_type_arg == "INT64" )
    {
      type                  = OTF2_TYPE_INT64;
      attribute_value.int64 = 0;
    }
    else if ( attribute_type_arg == "FLOAT" )
    {
      type                    = OTF2_TYPE_FLOAT;
      attribute_value.float32 = 0;
    }
    else if ( attribute_type_arg == "DOUBLE" )
    {
      type                    = OTF2_TYPE_DOUBLE;
      attribute_value.float64 = 0;
    }
    else if ( attribute_type_arg == "STRING" )
    {
      type                      = OTF2_TYPE_STRING;
      attribute_value.stringRef = 0;
    }
    else if ( attribute_type_arg == "ATTRIBUTE" )
    {
      type                         = OTF2_TYPE_ATTRIBUTE;
      attribute_value.attributeRef = 0;
    }
    else if ( attribute_type_arg == "LOCATION" )
    {
      type                        = OTF2_TYPE_LOCATION;
      attribute_value.locationRef = 0;
    }
    else if ( attribute_type_arg == "REGION" )
    {
      type                      = OTF2_TYPE_REGION;
      attribute_value.regionRef = 0;
    }
    else if ( attribute_type_arg == "GROUP" )
    {
      type                     = OTF2_TYPE_GROUP;
      attribute_value.groupRef = 0;
    }
    else if ( attribute_type_arg == "METRIC" )
    {
      type                      = OTF2_TYPE_METRIC;
      attribute_value.metricRef = 0;
    }
    else if ( attribute_type_arg == "COMM" )
    {
      type                    = OTF2_TYPE_COMM;
      attribute_value.commRef = 0;
    }
    else if ( attribute_type_arg == "PARAMETER" )
    {
      type                         = OTF2_TYPE_PARAMETER;
      attribute_value.parameterRef = 0;
    }
    else if ( attribute_type_arg == "RMA_WIN" )
    {
      type                      = OTF2_TYPE_RMA_WIN;
      attribute_value.rmaWinRef = 0;
    }
    else if ( attribute_type_arg == "SOURCE_CODE_LOCATION" )
    {
      type                                  = OTF2_TYPE_SOURCE_CODE_LOCATION;
      attribute_value.sourceCodeLocationRef = 0;
    }
    else if ( attribute_type_arg == "CALLING_CONTEXT" )
    {
      type                              = OTF2_TYPE_CALLING_CONTEXT;
      attribute_value.callingContextRef = 0;
    }
    else if ( attribute_type_arg == "INTERRUPT_GENERATOR" )
    {
      type                                  = OTF2_TYPE_INTERRUPT_GENERATOR;
      attribute_value.interruptGeneratorRef = 0;
    }
    else if ( attribute_type_arg == "IO_FILE" )
    {
      type                      = OTF2_TYPE_IO_FILE;
      attribute_value.ioFileRef = 0;
    }
    else if ( attribute_type_arg == "IO_HANDLE" )
    {
      type                        = OTF2_TYPE_IO_HANDLE;
      attribute_value.ioHandleRef = 0;
    }
    else if ( attribute_type_arg == "LOCATION_GROUP" )
    {
      type                             = OTF2_TYPE_LOCATION_GROUP;
      attribute_value.locationGroupRef = 0;
    }
    else
    {
      die( "unknown attribute type: '" << attribute_type_arg << "'" );
    }

    OTF2_ErrorCode status = OTF2_AttributeList_AddAttribute(
      attribute_list,
      attribute++,
      type,
      attribute_value );
    if ( OTF2_SUCCESS != status )
    {
      die( "can't add attribute of type '" << attribute_type_arg << "'" );
    }
  }

  size_t size = OTF2_EventSizeEstimator_GetSizeOfAttributeList( estimator,
                                                               attribute_list );

  OTF2_AttributeList_Delete( attribute_list );

  return size;
}

static void
handle_set( OTF2_EventSizeEstimator* estimator,
           const string&            input,
           size_t&                  pos )
{
  string def_name = get_token( input, pos );

  if ( def_name == "" )
  {
    die( "missing definition specification in set command" );
  }

  string value_str = get_token( input, pos );
  if ( value_str == "" )
  {
    die( "missing number of definitions in set command" );
  }

  uint64_t value = strtol( value_str.c_str(), NULL, 0 );
  if ( def_name == "String" )
  {
    OTF2_EventSizeEstimator_SetNumberOfStringDefinitions( estimator,
                                                         value );
    return;
  }
  if ( def_name == "Attribute" )
  {
    OTF2_EventSizeEstimator_SetNumberOfAttributeDefinitions( estimator,
                                                            value );
    return;
  }
  if ( def_name == "Location" )
  {
    OTF2_EventSizeEstimator_SetNumberOfLocationDefinitions( estimator,
                                                           value );
    return;
  }
  if ( def_name == "Region" )
  {
    OTF2_EventSizeEstimator_SetNumberOfRegionDefinitions( estimator,
                                                         value );
    return;
  }
  if ( def_name == "Group" )
  {
    OTF2_EventSizeEstimator_SetNumberOfGroupDefinitions( estimator,
                                                        value );
    return;
  }
  if ( def_name == "Metric" )
  {
    OTF2_EventSizeEstimator_SetNumberOfMetricDefinitions( estimator,
                                                         value );
    return;
  }
  if ( def_name == "Comm" )
  {
    OTF2_EventSizeEstimator_SetNumberOfCommDefinitions( estimator,
                                                       value );
    return;
  }
  if ( def_name == "Parameter" )
  {
    OTF2_EventSizeEstimator_SetNumberOfParameterDefinitions( estimator,
                                                            value );
    return;
  }
  if ( def_name == "RmaWin" )
  {
    OTF2_EventSizeEstimator_SetNumberOfRmaWinDefinitions( estimator,
                                                         value );
    return;
  }
  if ( def_name == "SourceCodeLocation" )
  {
    OTF2_EventSizeEstimator_SetNumberOfSourceCodeLocationDefinitions( estimator,
                                                                     value );
    return;
  }
  if ( def_name == "CallingContext" )
  {
    OTF2_EventSizeEstimator_SetNumberOfCallingContextDefinitions( estimator,
                                                                 value );
    return;
  }
  if ( def_name == "InterruptGenerator" )
  {
    OTF2_EventSizeEstimator_SetNumberOfInterruptGeneratorDefinitions( estimator,
                                                                     value );
    return;
  }
  if ( def_name == "IoFile" )
  {
    OTF2_EventSizeEstimator_SetNumberOfIoFileDefinitions( estimator,
                                                         value );
    return;
  }
  if ( def_name == "IoHandle" )
  {
    OTF2_EventSizeEstimator_SetNumberOfIoHandleDefinitions( estimator,
                                                           value );
    return;
  }
  if ( def_name == "LocationGroup" )
  {
    OTF2_EventSizeEstimator_SetNumberOfLocationGroupDefinitions( estimator,
                                                                value );
    return;
  }

  die( "unknown definition '" << def_name << "'" );
}

static void
print_event_list( void )
{
  cout
    << "Timestamp" << endl
    << "AttributeList" << endl
    << "BufferFlush"
    << endl
    << "MeasurementOnOff"
    << endl
    << "Enter"
    << endl
    << "Leave"
    << endl
    << "MpiSend"
    << endl
    << "MpiIsend"
    << endl
    << "MpiIsendComplete"
    << endl
    << "MpiIrecvRequest"
    << endl
    << "MpiRecv"
    << endl
    << "MpiIrecv"
    << endl
    << "MpiRequestTest"
    << endl
    << "MpiRequestCancelled"
    << endl
    << "MpiCollectiveBegin"
    << endl
    << "MpiCollectiveEnd"
    << endl
    << "OmpFork"
    << endl
    << "OmpJoin"
    << endl
    << "OmpAcquireLock"
    << endl
    << "OmpReleaseLock"
    << endl
    << "OmpTaskCreate"
    << endl
    << "OmpTaskSwitch"
    << endl
    << "OmpTaskComplete"
    << endl
    << "Metric"
    << " <numberOfMetrics>"
    << endl
    << "ParameterString"
    << endl
    << "ParameterInt"
    << endl
    << "ParameterUnsignedInt"
    << endl
    << "RmaWinCreate"
    << endl
    << "RmaWinDestroy"
    << endl
    << "RmaCollectiveBegin"
    << endl
    << "RmaCollectiveEnd"
    << endl
    << "RmaGroupSync"
    << endl
    << "RmaRequestLock"
    << endl
    << "RmaAcquireLock"
    << endl
    << "RmaTryLock"
    << endl
    << "RmaReleaseLock"
    << endl
    << "RmaSync"
    << endl
    << "RmaWaitChange"
    << endl
    << "RmaPut"
    << endl
    << "RmaGet"
    << endl
    << "RmaAtomic"
    << endl
    << "RmaOpCompleteBlocking"
    << endl
    << "RmaOpCompleteNonBlocking"
    << endl
    << "RmaOpTest"
    << endl
    << "RmaOpCompleteRemote"
    << endl
    << "ThreadFork"
    << endl
    << "ThreadJoin"
    << endl
    << "ThreadTeamBegin"
    << endl
    << "ThreadTeamEnd"
    << endl
    << "ThreadAcquireLock"
    << endl
    << "ThreadReleaseLock"
    << endl
    << "ThreadTaskCreate"
    << endl
    << "ThreadTaskSwitch"
    << endl
    << "ThreadTaskComplete"
    << endl
    << "ThreadCreate"
    << endl
    << "ThreadBegin"
    << endl
    << "ThreadWait"
    << endl
    << "ThreadEnd"
    << endl
    << "CallingContextEnter"
    << endl
    << "CallingContextLeave"
    << endl
    << "CallingContextSample"
    << endl
    << "IoCreateHandle"
    << endl
    << "IoDestroyHandle"
    << endl
    << "IoDuplicateHandle"
    << endl
    << "IoSeek"
    << endl
    << "IoChangeStatusFlags"
    << endl
    << "IoDeleteFile"
    << endl
    << "IoOperationBegin"
    << endl
    << "IoOperationTest"
    << endl
    << "IoOperationIssued"
    << endl
    << "IoOperationComplete"
    << endl
    << "IoOperationCancelled"
    << endl
    << "IoAcquireLock"
    << endl
    << "IoReleaseLock"
    << endl
    << "IoTryLock"
    << endl
    << "ProgramBegin"
    << " <numberOfArguments>"
    << endl
    << "ProgramEnd"
    << endl
    << "NonBlockingCollectiveRequest"
    << endl
    << "NonBlockingCollectiveComplete"
    << endl
    << "CommCreate"
    << endl
    << "CommDestroy"
    << endl
    ;
}

static void
print_definition_list( void )
{
  cout
    << "String" << endl
    << "Attribute" << endl
    << "Location" << endl
    << "Region" << endl
    << "Group" << endl
    << "Metric" << endl
    << "Comm" << endl
    << "Parameter" << endl
    << "RmaWin" << endl
    << "SourceCodeLocation" << endl
    << "CallingContext" << endl
    << "InterruptGenerator" << endl
    << "IoFile" << endl
    << "IoHandle" << endl
    << "LocationGroup" << endl
    ;
}

static void
print_type_list( void )
{
  cout
    << "UINT8" << endl
    << "UINT16" << endl
    << "UINT32" << endl
    << "UINT64" << endl
    << "INT8" << endl
    << "INT16" << endl
    << "INT32" << endl
    << "INT64" << endl
    << "FLOAT" << endl
    << "DOUBLE" << endl
    << "STRING" << endl
    << "ATTRIBUTE" << endl
    << "LOCATION" << endl
    << "REGION" << endl
    << "GROUP" << endl
    << "METRIC" << endl
    << "COMM" << endl
    << "PARAMETER" << endl
    << "RMA_WIN" << endl
    << "SOURCE_CODE_LOCATION" << endl
    << "CALLING_CONTEXT" << endl
    << "INTERRUPT_GENERATOR" << endl
    << "IO_FILE" << endl
    << "IO_HANDLE" << endl
    << "LOCATION_GROUP" << endl
    ;
}
