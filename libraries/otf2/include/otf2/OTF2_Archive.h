/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2013,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2013,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2014, 2021,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2013,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2013,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2013,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2013,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 *
 */

#ifndef OTF2_ARCHIVE_H
#define OTF2_ARCHIVE_H

/**
 *  @file
 *
 *  @brief      Writing interface for OTF2 archives.
 *
 *  @addtogroup usage OTF2 usage examples
 *  @{
 *
 *  @defgroup usage_writing Usage in writing mode - a simple example
 *  @{
 *
 *  This is a short example of how to use the OTF2 writing interface. This
 *  example is available as source code in the file @refexample{otf2_writer_example,c}.
 *  See also @refexample{otf2_openmp_writer_example,c} and
 *  @refexample{otf2_pthread_writer_example,c} when writing with multiple threads.
 *
 *  @dontinclude otf2_writer_example.c
 *
 *  First include the OTF2 header.
 *  @skipline otf2/otf2.h
 *
 *  For this example an additional include statement is necessary.
 *  @skipline stdlib.h
 *
 *  Furthermore this example uses a function delivering dummy timestamps.
 *  Real world applications will use a timer like clock_gettime.
 *  @line OTF2_TimeStamp
 *  @until }
 *
 *  Define a pre and post flush callback.
 *  If no memory is left in OTF2's internal memory buffer or the writer handle
 *  is closed a memory buffer flushing routine is triggered.
 *  The pre flush callback is triggered right before a buffer flush. It needs to
 *  return either OTF2_FLUSH to flush the recorded data to a file or
 *  OTF2_NO_FLUSH to suppress flushing data to a file.
 *  The post flush callback is triggered right after a memory buffer flush. It
 *  has to return a current timestamp which is recorded to mark the time spent
 *  in a buffer flush.
 *  Note that while the pre flush callback is always called, the post flush
 *  callback is only called during measurement when flushing an event file.
 *  In particular this means that closing the writer handle will only trigger
 *  the pre flush callback, and therefore the pre and post flush callbacks must
 *  be independent of each other--it is not permissible to start an operation
 *  in the pre flush callback and rely on the post flush callback to complete
 *  it.
 *  The callbacks are passed via a struct to OTF2.
 *  @line OTF2_FlushType
 *  @until };
 *
 *  Now everything is prepared to begin with the main program.
 *  @line int
 *  @until {
 *
 *  Create new archive handle.
 *  @line Open
 *  @until );
 *
 *  Set the previously defined flush callbacks.
 *  @skipline SetFlushCallbacks
 *
 *  We will operate in a serial context.
 *  @skipline SerialCollective
 *
 *  Now we can create the event files. Though physical files aren't created yet.
 *  @skipline OpenEvtFiles
 *
 *  Get a local event writer for location 0.
 *  @skipline GetEvtWriter
 *
 *  Write an enter and a leave record for region 0 to the local event writer.
 *  @line Enter
 *  @until );
 *  @until );
 *
 *  Now close the event writer, before closing the event files collectively.
 *  @skipline CloseEvtWriter
 *
 *  After we wrote all of the events we close the event files again.
 *  @skipline CloseEvtFiles
 *
 *  The local definition files are optional, but create them nevertheless to
 *  please the reader.
 *  @line OpenDefFiles
 *  @until CloseDefFiles
 *
 *  Now write the global definitions by getting a writer object for it.
 *  @skipline GetGlobalDefWriter
 *
 *  We need to define the clock used for this trace and the overall timestamp
 *  range.
 *  @line WriteClockProperties
 *  @until );
 *
 *  Now we can start writing the referenced definitions, starting with the strings.
 *  @line String
 *  @until 7
 *
 *  Write definition for the code region which was just entered and left to the
 *  global definition writer.
 *  @line Region
 *  @until );
 *
 *  Write the system tree including a definition for the location group to the
 *  global definition writer.
 *  @line SystemTreeNode
 *  @until );
 *  @until );
 *
 *  Write a definition for the location to the global definition writer.
 *  @line Location
 *  @until );
 *
 *  At the end, close the archive and exit.
 *  @line Close
 *  @until }
 *
 *  To compile your program use a command like the following. Note that we need
 *  to activate the C99 standard explicitly for GCC.
 *  @dontinclude Makefile
 *  @skip otf2_writer_example.c
 *  @line gcc
 *  @until -o
 *
 *  Now you can link your program with:
 *  @line gcc
 *  @until -o
 *
 *  @}
 *
 *  @}
 */

#include <stdint.h>
#include "otf2_compiler.h"

#include <otf2/OTF2_ErrorCodes.h>

#include <otf2/OTF2_Callbacks.h>
#include <otf2/OTF2_DefReader.h>
#include <otf2/OTF2_DefWriter.h>
#include <otf2/OTF2_EvtReader.h>
#include <otf2/OTF2_EvtWriter.h>
#include <otf2/OTF2_GlobalDefReader.h>
#include <otf2/OTF2_GlobalDefWriter.h>
#include <otf2/OTF2_GlobalEvtReader.h>
#include <otf2/OTF2_GlobalSnapReader.h>
#include <otf2/OTF2_MarkerReader.h>
#include <otf2/OTF2_MarkerWriter.h>
#include <otf2/OTF2_SnapReader.h>
#include <otf2/OTF2_SnapWriter.h>
#include <otf2/OTF2_Thumbnail.h>
#include "otf2/otf2.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** @brief Default size for OTF2's internal event chunk memory handling.
 *
 *  If you are not sure which chunk size is the best to use, use this default
 *  value.
 */
#define OTF2_CHUNK_SIZE_EVENTS_DEFAULT (1024 * 1024)

/** @brief Default size for OTF2's internal event chunk memory handling.
 *
 *  If you are not sure which chunk size is the best to use, use this default
 *  value.
 */
#define OTF2_CHUNK_SIZE_DEFINITIONS_DEFAULT (4 * 1024 * 1024)

/** @brief Keeps all meta-data for an OTF2 archive.
 *
 *  An OTF2 archive handle keeps all runtime information about an OTF2 archive.
 *  It is the central handle to at and set information about the archive and to
 *  request event and definition writer handles. */
typedef struct OTF2_Archive_struct OTF2_Archive;

/** @brief Create a new archive.
 *
 *  Creates a new archive handle that keeps all meta data about the archive on
 *  runtime.
 *
 *  @param archivePath      Path to the archive i.e. the directory where the
 *                          anchor file is located.
 *  @param archiveName      Name of the archive. It is used to generate sub
 *                          paths e.g. 'archiveName.otf2'.
 *  @param fileMode         Determines if in reading or writing mode.
 *                          Available values are @eref{OTF2_FILEMODE_WRITE} or
 *                          @eref{OTF2_FILEMODE_READ}.
 *  @param chunkSizeEvents  Requested size of OTF2's internal event chunks in
 *                          writing mode. Available values are from 256kB to
 *                          16MB. The event chunk size affects performance as
 *                          well as total memory usage.
 *                          A value satisfying both is about 1MB.
 *                          If you are not sure which chunk size is the best to
 *                          use, use @eref{OTF2_CHUNK_SIZE_EVENTS_DEFAULT}.
 *                          In reading mode this value is ignored because the
 *                          correct chunk size is extracted from the anchor
 *                          file.
 *  @param chunkSizeDefs    Requested size of OTF2's internal definition chunks
 *                          in writing mode. Available values are from 256kB to
 *                          16MB. The definition chunk size affects performance
 *                          as well as total memory usage. In addition, the
 *                          definition chunk size must be big enough to carry
 *                          the largest possible definition record. Therefore,
 *                          the definition chunk size must be at least 10 times
 *                          the number of locations.
 *                          A value satisfying these requirements is about 4MB.
 *                          If you are not sure which chunk size is the best to
 *                          use, use @eref{OTF2_CHUNK_SIZE_DEFINITIONS_DEFAULT}.
 *                          In reading mode this value is ignored because the
 *                          correct chunk size is extracted from the anchor
 *                          file.
 *  @param fileSubstrate    Determines which file substrate should be used in
 *                          writing mode.
 *                          Available values are @eref{OTF2_SUBSTRATE_POSIX} to use the
 *                          standard Posix interface, @eref{OTF2_SUBSTRATE_SION} to use
 *                          an installed SION library to store multiple logical
 *                          files into fewer or one physical file, and
 *                          @eref{OTF2_SUBSTRATE_NONE} to suppress file writing at all.
 *                          In reading mode this value is ignored because the
 *                          correct file substrate is extracted from the anchor
 *                          file.
 *  @param compression      Determines if compression is used to reduce the size
 *                          of data in files.
 *                          Available values are @eref{OTF2_COMPRESSION_ZLIB}
 *                          to use an installed zlib and
 *                          @eref{OTF2_COMPRESSION_NONE} to disable compression.
 *                          In reading mode this value is ignored because the
 *                          correct file compression is extracted from the
 *                          anchor file.
 *
 *  @return                 Returns an archive handle if successful, NULL
 *                          otherwise.
 */
OTF2_Archive* OTF2_Archive_Open(const char* archivePath,
                                const char* archiveName,
                                const OTF2_FileMode fileMode,
                                const uint64_t chunkSizeEvents,
                                const uint64_t chunkSizeDefs,
                                const OTF2_FileSubstrate fileSubstrate,
                                const OTF2_Compression compression);

/** @brief Close an opened archive.
 *
 *  Closes an opened archive and releases the associated resources. Closes also
 *  all opened writer and reader handles.
 *  Does nothing if NULL is passed.
 *
 *  @param archive          Archive handle.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_Close(OTF2_Archive* archive);

/** @brief Switch file mode of the archive.
 *
 *  Currently only a switch from @eref{OTF2_FILEMODE_READ} to
 *  @eref{OTF2_FILEMODE_WRITE} is permitted.
 *  Currently it is also only permitted when operating on an OTF2 archive
 *  with the @eref{OTF2_SUBSTRATE_POSIX} file substrate.
 *
 *  @param archive          Archive handle.
 *  @param newFileMode      New @eref{OTF2_FileMode} to switch to.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 *  @since Version 1.2
 */
OTF2_ErrorCode OTF2_Archive_SwitchFileMode(OTF2_Archive* archive, OTF2_FileMode newFileMode);

/** @brief Set the definition chunksize.
 *
 *  This maybe called for archives in the @eref{OTF2_FILEMODE_WRITE} mode and
 *  with an initial value of @eref{OTF2_UNDEFINED_UINT64} for the definition
 *  chunksize when opening the archive, but before requesting any local or
 *  global definition, or marker writer.
 *
 *  This function is a collective operation. The actual value is taken from the
 *  primary archive. The @p chunkSize argument on non-primary archives must be
 *  @eref{OTF2_UNDEFINED_UINT64}.
 *
 *  @param archive   Archive handle.
 *  @param chunkSize Chunk size for definition files.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_SetDefChunkSize(OTF2_Archive* archive, uint64_t chunkSize);

/** @brief Set machine name.
 *
 *  Sets the name for the machine the trace was recorded.
 *  This value is optional. When there is a primary archive, only the value
 *  provided for the primary archive will be used.
 *
 *  @param archive          Archive handle.
 *  @param machineName      Machine name.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_SetMachineName(OTF2_Archive* archive, const char* machineName);

/** @brief Set a description.
 *
 *  Sets a description for a trace archive.
 *  This value is optional. When there is a primary archive, only the value
 *  provided for the primary archive will be used.
 *
 *  @param archive          Archive handle.
 *  @param description      Description.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_SetDescription(OTF2_Archive* archive, const char* description);

/** @brief Set creator.
 *
 *  Sets information about the creator of the trace archive.
 *  This value is optional. When there is a primary archive, only the value
 *  provided for the primary archive will be used.
 *
 *  @param archive          Archive handle.
 *  @param creator          Creator information.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_SetCreator(OTF2_Archive* archive, const char* creator);

/** @brief Set the flush callbacks for the archive.
 *
 *  @param archive          Archive handle.
 *  @param flushCallbacks   Struct holding the flush callback functions.
 *  @param flushData        Data passed to the flush callbacks in the
 *                          @p userData argument.
 *
 *  @return OTF2_ErrorCode, or error code.
 */
OTF2_ErrorCode OTF2_Archive_SetFlushCallbacks(OTF2_Archive* archive,
                                              const OTF2_FlushCallbacks* flushCallbacks,
                                              void* flushData);

/** @brief Set the memory callbacks for the archive.
 *
 *  @param archive          Archive handle.
 *  @param memoryCallbacks  Struct holding the memory callback functions.
 *  @param memoryData       Data passed to the memory callbacks in the
 *                          @p userData argument.
 *
 *  @return OTF2_ErrorCode, or error code.
 */
OTF2_ErrorCode OTF2_Archive_SetMemoryCallbacks(OTF2_Archive* archive,
                                               const OTF2_MemoryCallbacks* memoryCallbacks,
                                               void* memoryData);

/** @brief Set the collective callbacks for the archive.
 *
 *  This function is a collective operation.
 *
 *  @param archive              Archive handle.
 *  @param collectiveCallbacks  Struct holding the collective callback functions.
 *  @param collectiveData       Data passed to the collective callbacks in the
 *                              @p userData argument.
 *  @param globalCommContext    Global communication context.
 *  @param localCommContext     Local communication context.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_SetCollectiveCallbacks(OTF2_Archive* archive,
                                                   const OTF2_CollectiveCallbacks* collectiveCallbacks,
                                                   void* collectiveData,
                                                   OTF2_CollectiveContext* globalCommContext,
                                                   OTF2_CollectiveContext* localCommContext);

/** @brief Convenient function to set the collective callbacks to an serial
 *         implementation.
 *
 *  @param archive Archive handle.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_SetSerialCollectiveCallbacks(OTF2_Archive* archive);

/** @brief Determines if this archive is the primary archive which handles the
 *  global archive state (anchor file, global definitions, marker, ...).
 *
 *  @param archive      Archive handle.
 *  @param[out] result  Storage for the result.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_IsPrimary(OTF2_Archive* archive, bool* result);

/** @brief Same as OTF2_Archive_IsPrimary.
 *
 *  @param archive      Archive handle.
 *  @param[out] result  Storage for the result.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 * @deprecated In version 3.0
 */
OTF2_ErrorCode OTF2_Archive_IsMaster(OTF2_Archive* archive, bool* result);

/** @brief Set the locking callbacks for the archive.
 *
 *  Can be called any time, but only once. Before this call no thread-safety
 *  is guaranteed.
 *
 *  @param archive           Archive handle.
 *  @param lockingCallbacks  Struct holding the locking callback functions.
 *  @param lockingData       Data passed to the locking callbacks in the
 *                           @p userData argument.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             in case of NULL pointers for @a archive or @a lockingCallbacks\,
 *             or mandatory callbacks in @a lockingCallbacks are missing}
 *    @retcode{OTF2_ERROR_INVALID_CALL,
 *             in case there were locking callbacks already set}
 *  @retend
 */
OTF2_ErrorCode OTF2_Archive_SetLockingCallbacks(OTF2_Archive* archive,
                                                const OTF2_LockingCallbacks* lockingCallbacks,
                                                void* lockingData);

/** @brief Set the @a hint in the @a archive to the given @a value.
 *
 *  Hints can only be set once and only before OTF2 itself uses the hint
 *  the first time.
 *
 *  @param archive Archive handle.
 *  @param hint    Name of the hint.
 *  @param value   Reference to the hint value.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             in case of NULL pointers for @a archive or @a value\,
 *             or an unknown @a hint value}
 *    @retcode{OTF2_ERROR_HINT_INVALID,
 *             in case the hint is not valid for this handle}
 *    @retcode{OTF2_ERROR_HINT_LOCKED,
 *             in case the hint was already set or was queried at least
 *             once by the handle}
 *    @retcode{OTF2_ERROR_HINT_INVALID_VALUE,
 *             in case the provided value is invalid for this hint}
 *  @retend
 */
OTF2_ErrorCode OTF2_Archive_SetHint(OTF2_Archive* archive, OTF2_Hint hint, void* value);

/** @brief Add or remove a trace file property to this archive.
 *
 *  Removing a trace file property is done by passing "" in the @p value parameter.
 *  The @p overwrite parameter is ignored than.
 *
 *  @note This call is only allowed when the archive was opened with mode
 *        @eref{OTF2_FILEMODE_WRITE}.
 *
 *  @param archive          Archive handle.
 *  @param name             Name of the trace file property (case insensitive, [A-Z0-9_]).
 *  @param value            Value of property.
 *  @param overwrite        If true a previous trace file property with the same
 *                          name @p name will be overwritten.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_PROPERTY_NAME_INVALID,
 *             if property name does not conform to the naming scheme}
 *    @retcode{OTF2_ERROR_PROPERTY_NOT_FOUND,
 *             if property was not found\, but requested to remove}
 *    @retcode{OTF2_ERROR_PROPERTY_EXISTS,
 *             if property exists but overwrite was not set}
 *  @retend
 */
OTF2_ErrorCode OTF2_Archive_SetProperty(OTF2_Archive* archive, const char* name, const char* value, bool overwrite);

/** @brief Add or remove a boolean trace file property to this archive.
 *
 *  @note This call is only allowed when the archive was opened with mode
 *        @eref{OTF2_FILEMODE_WRITE}.
 *
 *  @param archive          Archive handle.
 *  @param name             Name of the trace file property (case insensitive, [A-Z0-9_]).
 *  @param value            Boolean value of property (e.g. true or false).
 *  @param overwrite        If true a previous trace file property with the same
 *                          name @p name will be overwritten.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_PROPERTY_NAME_INVALID,
 *             if property name does not conform to the naming scheme}
 *    @retcode{OTF2_ERROR_PROPERTY_NOT_FOUND,
 *             if property was not found\, but requested to remove}
 *    @retcode{OTF2_ERROR_PROPERTY_EXISTS,
 *             if property exists but overwrite was not set}
 *  @retend
 */
OTF2_ErrorCode OTF2_Archive_SetBoolProperty(OTF2_Archive* archive, const char* name, bool value, bool overwrite);

/** @brief Get the names of all trace file properties.
 *
 *  @param archive                 Archive handle.
 *  @param[out] numberOfProperties Returned number of trace file properties.
 *  @param[out] names              Returned list of property names. Allocated
 *                                 with @a malloc. To release memory, just pass
 *                                 @p *names to @a free.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_GetPropertyNames(OTF2_Archive* archive, uint32_t* numberOfProperties, char*** names);

/** @brief Get the value of the named trace file property.
 *
 *  @param archive          Archive handle.
 *  @param name             Name of the property.
 *  @param[out] value       Returned value of the property. Allocated with
 *                          @a malloc.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_PROPERTY_NOT_FOUND,
 *             if the named property was not found}
 *  @retend
 */
OTF2_ErrorCode OTF2_Archive_GetProperty(OTF2_Archive* archive, const char* name, char** value);

/** @brief Get the identifier of the trace file.
 *
 *  @note This call is only allowed when the archive was opened with mode
 *        OTF2_FILEMODE_READ.
 *
 *  @param archive          Archive handle.
 *  @param[out] id          Trace identifier.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_GetTraceId(OTF2_Archive* archive, uint64_t* id);

/** @brief Get the number of locations.
 *
 *  @param archive                Archive handle.
 *  @param[out] numberOfLocations Return pointer to the number of locations.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_GetNumberOfLocations(OTF2_Archive* archive, uint64_t* numberOfLocations);

/** @brief Get the number of global definitions.
 *
 *  @param archive                  Archive handle.
 *  @param[out] numberOfDefinitions Return pointer to the number of global definitions.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_GetNumberOfGlobalDefinitions(OTF2_Archive* archive, uint64_t* numberOfDefinitions);

/** @brief Get machine name.
 *
 *  @param archive          Archive handle.
 *  @param[out] machineName Returned machine name. Allocated with @a malloc.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_GetMachineName(OTF2_Archive* archive, char** machineName);

/** @brief Get description.
 *
 *  @param archive          Archive handle.
 *  @param[out] description Returned description. Allocated with @a malloc.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_GetDescription(OTF2_Archive* archive, char** description);

/** @brief Get creator information.
 *
 *  @param archive          Archive handle.
 *  @param[out] creator     Returned creator. Allocated with @a malloc.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_GetCreator(OTF2_Archive* archive, char** creator);

/** @brief Get format version.
 *
 *  @param archive          Archive handle
 *  @param[out] major       Major version number
 *  @param[out] minor       Minor version number
 *  @param[out] bugfix      Bugfix revision
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_GetVersion(OTF2_Archive* archive, uint8_t* major, uint8_t* minor, uint8_t* bugfix);

/** @brief Get the chunksize.
 *
 *  @param archive              Archive handle.
 *  @param[out] chunkSizeEvents Chunk size for event files.
 *  @param[out] chunkSizeDefs   Chunk size for definition files.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_GetChunkSize(OTF2_Archive* archive, uint64_t* chunkSizeEvents, uint64_t* chunkSizeDefs);

/** @brief Get the file substrate (posix, sion, none)
 *
 *  @param archive          Archive handle.
 *  @param[out] substrate   Returned file substrate.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_GetFileSubstrate(OTF2_Archive* archive, OTF2_FileSubstrate* substrate);

/** @brief Get compression mode (none or zlib)
 *
 *  @param archive          Archive handle.
 *  @param[out] compression Returned compression mode.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_GetCompression(OTF2_Archive* archive, OTF2_Compression* compression);

/** @brief Get a local event writer.
 *
 *  @param archive          Archive handle.
 *  @param location         Location ID of the requested writer handle.
 *
 *  @return                 Returns a local event writer handle if successful,
 *                          NULL if an error occurs.
 */
OTF2_EvtWriter* OTF2_Archive_GetEvtWriter(OTF2_Archive* archive, OTF2_LocationRef location);

/** @brief Get a local definition writer.
 *
 *  @param archive          Archive handle.
 *  @param location         Location ID of the requested writer handle.
 *
 *  @return                 Returns a local definition writer handle if
 *                          successful, NULL if an error occurs.
 */
OTF2_DefWriter* OTF2_Archive_GetDefWriter(OTF2_Archive* archive, OTF2_LocationRef location);

/** @brief Get a global definition writer.
 *
 *  Only the primary archive can call this function.
 *
 *  @param archive          Archive handle.
 *
 *  @return                 Returns a global definition writer handle if
 *                          successful, NULL if an error occurs.
 */
OTF2_GlobalDefWriter* OTF2_Archive_GetGlobalDefWriter(OTF2_Archive* archive);

/** @brief Get a local snap writer.
 *
 *  @param archive          Archive handle.
 *  @param location         Location ID of the requested writer handle.
 *
 *  @since Version 1.2
 *
 *  @return                 Returns a local event writer handle if successful,
 *                          NULL if an error occurs.
 */
OTF2_SnapWriter* OTF2_Archive_GetSnapWriter(OTF2_Archive* archive, OTF2_LocationRef location);

/** @brief Get a thumb writer.
 *
 *  @param archive          Archive handle.
 *
 *  @param name             Name of thumb.
 *  @param description      Description of thumb.
 *  @param type             Type of thumb.
 *  @param numberOfSamples  Number of samples.
 *  @param numberOfMetrics  Number of metrics.
 *  @param refsToDefs       @a numberOfMetrics references to definition matching
 *                          the thumbnail type.
 *
 *  @since Version 1.2
 *
 *  @return                 Returns a thumb writer handle if
 *                          successful, NULL if an error occurs.
 */
OTF2_ThumbWriter* OTF2_Archive_GetThumbWriter(OTF2_Archive* archive,
                                              const char* name,
                                              const char* description,
                                              OTF2_ThumbnailType type,
                                              uint32_t numberOfSamples,
                                              uint32_t numberOfMetrics,
                                              const uint64_t* refsToDefs);

/** @brief Get a marker writer.
 *
 *  @param archive          Archive handle.
 *
 *  @since Version 1.2
 *
 *  @return                 Returns a marker writer handle if
 *                          successful, NULL if an error occurs.
 */
OTF2_MarkerWriter* OTF2_Archive_GetMarkerWriter(OTF2_Archive* archive);

/** @brief Get a local event reader.
 *
 *  @param archive          Archive handle.
 *  @param location         Location ID of the requested reader handle.
 *
 *  @return                 Returns a local event reader handle if successful,
 *                          NULL if an error occurs.
 */
OTF2_EvtReader* OTF2_Archive_GetEvtReader(OTF2_Archive* archive, OTF2_LocationRef location);

/** @brief Get a global event reader.
 *
 *  @param archive          Archive handle.
 *
 *  @return                 Returns a global event reader handle if successful,
 *                          NULL if an error occurs.
 */
OTF2_GlobalEvtReader* OTF2_Archive_GetGlobalEvtReader(OTF2_Archive* archive);

/** @brief Get a local definition reader.
 *
 *  @param archive          Archive handle.
 *  @param location         Location ID of the requested reader handle.
 *
 *  @return                 Returns a local definition reader handle if
 *                          successful, NULL if an error occurs.
 */
OTF2_DefReader* OTF2_Archive_GetDefReader(OTF2_Archive* archive, OTF2_LocationRef location);

/** @brief Get a global definition reader.
 *
 *  @param archive          Archive handle.
 *
 *  @return                 Returns a global definition reader handle if
 *                          successful, NULL if an error occurs.
 */
OTF2_GlobalDefReader* OTF2_Archive_GetGlobalDefReader(OTF2_Archive* archive);

/** @brief Get a local snap reader.
 *
 *  @param archive          Archive handle.
 *  @param location         Location ID of the requested snap handle.
 *
 *  @since Version 1.2
 *
 *  @return                 Returns a local snap handle if successful,
 *                          NULL if an error occurs.
 */
OTF2_SnapReader* OTF2_Archive_GetSnapReader(OTF2_Archive* archive, OTF2_LocationRef location);

/** @brief Get a global snap reader.
 *
 *  @param archive          Archive handle.
 *
 *  @since Version 1.2
 *
 *  @return                 Returns a global snap reader handle if successful,
 *                          NULL if an error occurs.
 */
OTF2_GlobalSnapReader* OTF2_Archive_GetGlobalSnapReader(OTF2_Archive* archive);

/** @brief Get a thumb reader.
 *
 *  @param archive          Archive handle.
 *  @param number           Thumbnail number.
 *
 *  @since Version 1.2
 *
 *  @return                 Returns a global definition writer handle if
 *                          successful, NULL if an error occurs.
 */
OTF2_ThumbReader* OTF2_Archive_GetThumbReader(OTF2_Archive* archive, uint32_t number);

/** @brief Get a marker reader.
 *
 *  @param archive          Archive handle.
 *
 *  @since Version 1.2
 *
 *  @return                 Returns a marker reader handle if
 *                          successful, NULL if an error occurs.
 */
OTF2_MarkerReader* OTF2_Archive_GetMarkerReader(OTF2_Archive* archive);

/** @brief Close an opened local event writer.
 *
 *  @param archive          Archive handle.
 *  @param writer           Writer handle to be closed.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_CloseEvtWriter(OTF2_Archive* archive, OTF2_EvtWriter* writer);

/** @brief Close an opened local definition writer.
 *
 *  @param archive          Archive handle.
 *  @param writer           Writer handle to be closed.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_CloseDefWriter(OTF2_Archive* archive, OTF2_DefWriter* writer);

/** @brief Close an opened marker writer.
 *
 *  @param archive          Archive handle.
 *  @param writer           Writer handle to be closed.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_CloseMarkerWriter(OTF2_Archive* archive, OTF2_MarkerWriter* writer);

/** @brief Close an opened local snap writer.
 *
 *  @param archive          Archive handle.
 *  @param writer           Writer handle to be closed.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_CloseSnapWriter(OTF2_Archive* archive, OTF2_SnapWriter* writer);

/** @brief Close an opened global definition writer.
 *
 *  Only the primary archive can call this function.
 *
 *  @param archive   Archive handle.
 *  @param writer    Writer handle to be closed.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if the archive or writer argument is invalid}
 *    @retcode{OTF2_ERROR_INVALID_CALL,
 *             if this is not the primary archive}
 *  @retend
 */
OTF2_ErrorCode OTF2_Archive_CloseGlobalDefWriter(OTF2_Archive* archive, OTF2_GlobalDefWriter* writer);

/** @brief Close an opened local event reader.
 *
 *  @param archive          Archive handle.
 *  @param reader           Reader handle to be closed.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_CloseEvtReader(OTF2_Archive* archive, OTF2_EvtReader* reader);

/** @brief Close an opened thumbnail reader.
 *
 *  @param archive          Archive handle.
 *  @param reader           Reader handle to be closed.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_CloseThumbReader(OTF2_Archive* archive, OTF2_ThumbReader* reader);

/** @brief Closes the global event reader.
 *
 *  This closes also all local event readers.
 *
 *  @param archive          Archive handle.
 *  @param globalEvtReader  The global event reader.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_CloseGlobalEvtReader(OTF2_Archive* archive, OTF2_GlobalEvtReader* globalEvtReader);

/** @brief Close an opened local definition reader.
 *
 *  @param archive          Archive handle.
 *  @param reader           Reader handle to be closed.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_CloseDefReader(OTF2_Archive* archive, OTF2_DefReader* reader);

/** @brief Closes the global definition reader.
 *
 *  @param archive          Archive handle.
 *  @param globalDefReader  The global definition reader.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_CloseGlobalDefReader(OTF2_Archive* archive, OTF2_GlobalDefReader* globalDefReader);

/** @brief Close an opened local snap reader.
 *
 *  @param archive          Archive handle.
 *  @param reader           Reader handle to be closed.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 *  @since Version 1.2
 */
OTF2_ErrorCode OTF2_Archive_CloseSnapReader(OTF2_Archive* archive, OTF2_SnapReader* reader);

/** @brief Close the opened global snapshot reader.
 *
 *  @param archive          Archive handle.
 *  @param globalSnapReader Reader handle to be closed.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 */
OTF2_ErrorCode OTF2_Archive_CloseGlobalSnapReader(OTF2_Archive* archive, OTF2_GlobalSnapReader* globalSnapReader);

/** @brief Closes the marker reader.
 *
 *  @param archive          Archive handle.
 *  @param markerReader     The marker reader.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Archive_CloseMarkerReader(OTF2_Archive* archive, OTF2_MarkerReader* markerReader);

/** @brief Get the number of thumbnails.
 *
 *  @param archive          Archive handle.
 *  @param number           Thumb number.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 */
OTF2_ErrorCode OTF2_Archive_GetNumberOfThumbnails(OTF2_Archive* archive, uint32_t* number);

/** @brief Set the number of snapshots.
 *
 *  @param archive          Archive handle.
 *  @param number           Snapshot number.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 */
OTF2_ErrorCode OTF2_Archive_SetNumberOfSnapshots(OTF2_Archive* archive, uint32_t number);

/** @brief Get the number of snapshots.
 *
 *  @param archive          Archive handle.
 *  @param number           Snapshot number.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 */
OTF2_ErrorCode OTF2_Archive_GetNumberOfSnapshots(OTF2_Archive* archive, uint32_t* number);

/** @brief Open the events file container.
 *
 *  This function is a collective operation.
 *
 *  @param archive          Archive handle.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 */
OTF2_ErrorCode OTF2_Archive_OpenEvtFiles(OTF2_Archive* archive);

/** @brief Closes the events file container.
 *
 *  This function is a collective operation.
 *
 *  @param archive          Archive handle.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 */
OTF2_ErrorCode OTF2_Archive_CloseEvtFiles(OTF2_Archive* archive);

/** @brief Open the local definitions file container.
 *
 *  This function is a collective operation.
 *
 *  @param archive          Archive handle.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 */
OTF2_ErrorCode OTF2_Archive_OpenDefFiles(OTF2_Archive* archive);

/** @brief Closes the local definitions file container.
 *
 *  This function is a collective operation.
 *
 *  @param archive          Archive handle.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 */
OTF2_ErrorCode OTF2_Archive_CloseDefFiles(OTF2_Archive* archive);

/** @brief Open the snapshots file container.
 *
 *  This function is a collective operation.
 *
 *  @param archive          Archive handle.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 */
OTF2_ErrorCode OTF2_Archive_OpenSnapFiles(OTF2_Archive* archive);

/** @brief Closes the snapshots file container.
 *
 *  This function is a collective operation.
 *
 *  @param archive          Archive handle.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 */
OTF2_ErrorCode OTF2_Archive_CloseSnapFiles(OTF2_Archive* archive);

/** @brief Select a location to be read.
 *
 *  @param archive          Archive handle.
 *  @param location         Location ID.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 *
 */
OTF2_ErrorCode OTF2_Archive_SelectLocation(OTF2_Archive* archive, OTF2_LocationRef location);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !OTF2_ARCHIVE_H */
