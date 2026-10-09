/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2012,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2012,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2012, 2014, 2021,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2012,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2012,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2012,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2012,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 *
 */

#ifndef OTF2_GLOBAL_DEF_READER_CALLBACKS_H
#define OTF2_GLOBAL_DEF_READER_CALLBACKS_H

/**
 *  @file
 *  @source     templates/OTF2_GlobalDefReaderCallbacks.tmpl.h
 *
 *  @brief      This defines the callbacks for the global definition reader.
 */

#include <stdint.h>

#include "otf2_compiler.h"

#include <otf2/OTF2_ErrorCodes.h>

#include <otf2/OTF2_AttributeValue.h>
#include <otf2/OTF2_Definitions.h>
#include <otf2/OTF2_GeneralDefinitions.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** @brief Opaque struct which holds all global definition record callbacks.
 */
typedef struct OTF2_GlobalDefReaderCallbacks_struct OTF2_GlobalDefReaderCallbacks;

/** @brief Allocates a new struct for the global definition callbacks.
 *
 *  @return A newly allocated struct of type @eref{OTF2_GlobalDefReaderCallbacks}.
 */
OTF2_GlobalDefReaderCallbacks* OTF2_GlobalDefReaderCallbacks_New(void);

/** @brief Deallocates a struct for the global definition callbacks.
 *
 *  @param globalDefReaderCallbacks Handle to a struct previously allocated
 *                                  with @eref{OTF2_GlobalDefReaderCallbacks_New}.
 */
void OTF2_GlobalDefReaderCallbacks_Delete(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks);

/** @brief Clears a struct for the global definition callbacks.
 *
 *  @param globalDefReaderCallbacks Handle to a struct previously allocated
 *                                  with @eref{OTF2_GlobalDefReaderCallbacks_New}.
 */
void OTF2_GlobalDefReaderCallbacks_Clear(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks);

/** @brief Function pointer definition for the callback which is triggered by
 *         an unknown definition record.
 *
 *  @param userData User data as set by
 *                  @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                  @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_Unknown)(void* userData);

/** @brief Registers the callback for an unknown definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param unknownCallback          Function which should be called for all
 *                                  Unknown definitions.
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetUnknownCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                                OTF2_GlobalDefReaderCallback_Unknown unknownCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{ClockProperties} definition record.
 *
 *  Defines the timer resolution and time range of this trace. There will
 *  be no event with a timestamp less than @p globalOffset, and no
 *  event with timestamp greater than (@p globalOffset + @p
 *  traceLength).
 *
 *  @param userData          User data as set by
 *                           @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                           @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param timerResolution   Ticks per seconds.
 *  @param globalOffset      A timestamp smaller than all event timestamps.
 *  @param traceLength       A timespan which includes the timespan between the
 *                           smallest and greatest timestamp of all event
 *                           timestamps.
 *  @param realtimeTimestamp A realtime timestamp of the `globalOffset` timestamp
 *                           in nanoseconds since 1970-01-01T00:00 UTC. Use
 *                           @eref{OTF2_UNDEFINED_TIMESTAMP} if no such
 *                           timestamp exists. Since version 3.0.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_ClockProperties)(void* userData,
                                                                          uint64_t timerResolution,
                                                                          uint64_t globalOffset,
                                                                          uint64_t traceLength,
                                                                          uint64_t realtimeTimestamp);

/** @brief Registers the callback for the @eref{ClockProperties} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param clockPropertiesCallback  Function which should be called for all
 *                                  @eref{ClockProperties} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetClockPropertiesCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_ClockProperties clockPropertiesCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{Paradigm} definition record.
 *
 *  Attests that the following parallel paradigm was available at the time
 *  when the trace was recorded, and vice versa.  Note that this does
 *  not attest that the paradigm was used.  For convenience, this also
 *  includes a proper name for the paradigm and a classification.
 *  This definition is only allowed to appear at most once in the
 *  definitions per @eref{Paradigm}.
 *
 *  @param userData      User data as set by
 *                       @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                       @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param paradigm      The paradigm to attest.
 *  @param name          The name of the paradigm. References a @eref{String}
 *                       definition.
 *  @param paradigmClass The class of this paradigm.
 *
 *  @since Version 1.5
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_Paradigm)(void* userData,
                                                                   OTF2_Paradigm paradigm,
                                                                   OTF2_StringRef name,
                                                                   OTF2_ParadigmClass paradigmClass);

/** @brief Registers the callback for the @eref{Paradigm} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param paradigmCallback         Function which should be called for all
 *                                  @eref{Paradigm} definitions.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetParadigmCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_Paradigm paradigmCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{ParadigmProperty} definition record.
 *
 *  Extensible annotation for the @eref{Paradigm} definition.
 *
 *  The tuple (@a paradigm, @a property) must be unique.
 *
 *  @param userData User data as set by
 *                  @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                  @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param paradigm The paradigm to annotate.
 *  @param property The property.
 *  @param type     The type of this property. Must match with the defined type
 *                  of the @a property.
 *  @param value    The value of this property.
 *
 *  @since Version 1.5
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_ParadigmProperty)(void* userData,
                                                                           OTF2_Paradigm paradigm,
                                                                           OTF2_ParadigmProperty property,
                                                                           OTF2_Type type,
                                                                           OTF2_AttributeValue value);

/** @brief Registers the callback for the @eref{ParadigmProperty} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param paradigmPropertyCallback Function which should be called for all
 *                                  @eref{ParadigmProperty} definitions.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetParadigmPropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_ParadigmProperty paradigmPropertyCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{IoParadigm} definition record.
 *
 *  Attests that the following I/O paradigm was available at the time when
 *  the trace was recorded, and vice versa.  Note that this does not
 *  attest that the paradigm was used.  For convenience, this also
 *  includes a proper name for the paradigm and a classification.
 *
 *  @param userData           User data as set by
 *                            @eref{OTF2_Reader_RegisterGlobalDefCallbacks}
 *                            or @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self               The unique identifier for this @eref{IoParadigm}
 *                            definition.
 *  @param identification     The I/O paradigm identification.  This should be
 *                            used programmatically to identify a specific
 *                            I/O paradigm.  For a human-readable name use
 *                            the @p name attribute.  If this identification
 *                            matches one of the known I/O paradigms listed
 *                            in the OTF2 documentation @ref
 *                            otf2_known_io_paradigms, then the attributes of
 *                            this definition must match those specified
 *                            there. References a @eref{String} definition.
 *  @param name               The name of the I/O paradigm.  This should be
 *                            presented to humans as the name of this I/O
 *                            paradigm. References a @eref{String}
 *                            definition.
 *  @param ioParadigmClass    The class of this I/O paradigm.
 *  @param ioParadigmFlags    Boolean properties of this I/O paradigm.
 *  @param numberOfProperties Number of properties.
 *  @param properties         The property.
 *  @param types              The type of this property. Must match with the
 *                            defined type of the @a property.
 *  @param values             The value of this property.
 *
 *  @since Version 2.1
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_IoParadigm)(void* userData,
                                                                     OTF2_IoParadigmRef self,
                                                                     OTF2_StringRef identification,
                                                                     OTF2_StringRef name,
                                                                     OTF2_IoParadigmClass ioParadigmClass,
                                                                     OTF2_IoParadigmFlag ioParadigmFlags,
                                                                     uint8_t numberOfProperties,
                                                                     const OTF2_IoParadigmProperty* properties,
                                                                     const OTF2_Type* types,
                                                                     const OTF2_AttributeValue* values);

/** @brief Registers the callback for the @eref{IoParadigm} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param ioParadigmCallback       Function which should be called for all
 *                                  @eref{IoParadigm} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoParadigmCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoParadigm ioParadigmCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{String} definition record.
 *
 *  The string definition.
 *
 *  @param userData User data as set by
 *                  @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                  @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self     The unique identifier for this @eref{String} definition.
 *  @param string   The string, null terminated.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_String)(void* userData,
                                                                 OTF2_StringRef self,
                                                                 const char* string);

/** @brief Registers the callback for the @eref{String} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param stringCallback           Function which should be called for all
 *                                  @eref{String} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetStringCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                               OTF2_GlobalDefReaderCallback_String stringCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{Attribute} definition record.
 *
 *  The attribute definition.
 *
 *  @param userData    User data as set by
 *                     @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                     @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self        The unique identifier for this @eref{Attribute}
 *                     definition.
 *  @param name        Name of the attribute. References a @eref{String}
 *                     definition.
 *  @param description Description of the attribute. References a @eref{String}
 *                     definition. Since version 1.4.
 *  @param type        Type of the attribute value.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_Attribute)(void* userData,
                                                                    OTF2_AttributeRef self,
                                                                    OTF2_StringRef name,
                                                                    OTF2_StringRef description,
                                                                    OTF2_Type type);

/** @brief Registers the callback for the @eref{Attribute} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param attributeCallback        Function which should be called for all
 *                                  @eref{Attribute} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetAttributeCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_Attribute attributeCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{SystemTreeNode} definition record.
 *
 *  The system tree node definition.
 *
 *  @param userData  User data as set by
 *                   @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                   @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self      The unique identifier for this @eref{SystemTreeNode}
 *                   definition.
 *  @param name      Free form instance name of this node. References a
 *                   @eref{String} definition.
 *  @param className Free form class name of this node References a @eref{String}
 *                   definition.
 *  @param parent    Parent ID of this node. May be
 *                   @eref{OTF2_UNDEFINED_SYSTEM_TREE_NODE} to indicate that
 *                   there is no parent. References a @eref{SystemTreeNode}
 *                   definition.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_SystemTreeNode)(void* userData,
                                                                         OTF2_SystemTreeNodeRef self,
                                                                         OTF2_StringRef name,
                                                                         OTF2_StringRef className,
                                                                         OTF2_SystemTreeNodeRef parent);

/** @brief Registers the callback for the @eref{SystemTreeNode} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param systemTreeNodeCallback   Function which should be called for all
 *                                  @eref{SystemTreeNode} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodeCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_SystemTreeNode systemTreeNodeCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{LocationGroup} definition record.
 *
 *  The location group definition.
 *
 *  @param userData              User data as set by
 *                               @eref{OTF2_Reader_RegisterGlobalDefCallbacks}
 *                               or @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self                  The unique identifier for this
 *                               @eref{LocationGroup} definition.
 *  @param name                  Name of the group. References a @eref{String}
 *                               definition.
 *  @param locationGroupType     Type of this group.
 *  @param systemTreeParent      Parent of this location group in the system
 *                               tree. References a @eref{SystemTreeNode}
 *                               definition.
 *  @param creatingLocationGroup The creating location group of this group. For
 *                               type @eref{OTF2_LOCATION_GROUP_TYPE_PROCESS}
 *                               this may be another group of type
 *                               @eref{OTF2_LOCATION_GROUP_TYPE_PROCESS} or
 *                               @eref{OTF2_UNDEFINED_LOCATION_GROUP}. For
 *                               type
 *                               @eref{OTF2_LOCATION_GROUP_TYPE_ACCELERATOR},
 *                               this must be a group of type
 *                               @eref{OTF2_LOCATION_GROUP_TYPE_PROCESS}.
 *                               References a @eref{LocationGroup}
 *                               definition. Since version 3.0.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_LocationGroup)(void* userData,
                                                                        OTF2_LocationGroupRef self,
                                                                        OTF2_StringRef name,
                                                                        OTF2_LocationGroupType locationGroupType,
                                                                        OTF2_SystemTreeNodeRef systemTreeParent,
                                                                        OTF2_LocationGroupRef creatingLocationGroup);

/** @brief Registers the callback for the @eref{LocationGroup} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param locationGroupCallback    Function which should be called for all
 *                                  @eref{LocationGroup} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetLocationGroupCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_LocationGroup locationGroupCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{Location} definition record.
 *
 *  The location definition.
 *
 *  @param userData       User data as set by
 *                        @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                        @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self           The unique identifier for this @eref{Location}
 *                        definition.
 *  @param name           Name of the location References a @eref{String}
 *                        definition.
 *  @param locationType   Location type.
 *  @param numberOfEvents Number of events this location has recorded.
 *  @param locationGroup  Location group which includes this location. References
 *                        a @eref{LocationGroup} definition.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_Location)(void* userData,
                                                                   OTF2_LocationRef self,
                                                                   OTF2_StringRef name,
                                                                   OTF2_LocationType locationType,
                                                                   uint64_t numberOfEvents,
                                                                   OTF2_LocationGroupRef locationGroup);

/** @brief Registers the callback for the @eref{Location} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param locationCallback         Function which should be called for all
 *                                  @eref{Location} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetLocationCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_Location locationCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{Region} definition record.
 *
 *  The region definition.
 *
 *  @param userData        User data as set by
 *                         @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                         @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self            The unique identifier for this @eref{Region}
 *                         definition.
 *  @param name            Name of the region (demangled name if available).
 *                         References a @eref{String} definition.
 *  @param canonicalName   Alternative name of the region (e.g. mangled name).
 *                         References a @eref{String} definition. Since
 *                         version 1.1.
 *  @param description     A more detailed description of this region. References
 *                         a @eref{String} definition.
 *  @param regionRole      Region role. Since version 1.1.
 *  @param paradigm        Paradigm. Since version 1.1.
 *  @param regionFlags     Region flags. Since version 1.1.
 *  @param sourceFile      The source file where this region was declared.
 *                         References a @eref{String} definition.
 *  @param beginLineNumber Starting line number of this region in the source
 *                         file.
 *  @param endLineNumber   Ending line number of this region in the source file.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_Region)(void* userData,
                                                                 OTF2_RegionRef self,
                                                                 OTF2_StringRef name,
                                                                 OTF2_StringRef canonicalName,
                                                                 OTF2_StringRef description,
                                                                 OTF2_RegionRole regionRole,
                                                                 OTF2_Paradigm paradigm,
                                                                 OTF2_RegionFlag regionFlags,
                                                                 OTF2_StringRef sourceFile,
                                                                 uint32_t beginLineNumber,
                                                                 uint32_t endLineNumber);

/** @brief Registers the callback for the @eref{Region} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param regionCallback           Function which should be called for all
 *                                  @eref{Region} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetRegionCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                               OTF2_GlobalDefReaderCallback_Region regionCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{Callsite} definition record.
 *
 *  The callsite definition.
 *
 *  @param userData      User data as set by
 *                       @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                       @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self          The unique identifier for this @eref{Callsite}
 *                       definition.
 *  @param sourceFile    The source file where this call was made. References a
 *                       @eref{String} definition.
 *  @param lineNumber    Line number in the source file where this call was made.
 *  @param enteredRegion The region which was called. References a @eref{Region}
 *                       definition.
 *  @param leftRegion    The region which made the call. References a
 *                       @eref{Region} definition.
 *
 *  @since Version 1.0
 *
 *  @deprecated In version 2.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_Callsite)(void* userData,
                                                                   OTF2_CallsiteRef self,
                                                                   OTF2_StringRef sourceFile,
                                                                   uint32_t lineNumber,
                                                                   OTF2_RegionRef enteredRegion,
                                                                   OTF2_RegionRef leftRegion);

/** @brief Registers the callback for the @eref{Callsite} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param callsiteCallback         Function which should be called for all
 *                                  @eref{Callsite} definitions.
 *
 *  @since Version 1.0
 *
 *  @deprecated In version 2.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCallsiteCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_Callsite callsiteCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{Callpath} definition record.
 *
 *  The callpath definition.
 *
 *  @param userData User data as set by
 *                  @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                  @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self     The unique identifier for this @eref{Callpath} definition.
 *  @param parent   The parent of this callpath. References a @eref{Callpath}
 *                  definition.
 *  @param region   The region of this callpath. References a @eref{Region}
 *                  definition.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_Callpath)(void* userData,
                                                                   OTF2_CallpathRef self,
                                                                   OTF2_CallpathRef parent,
                                                                   OTF2_RegionRef region);

/** @brief Registers the callback for the @eref{Callpath} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param callpathCallback         Function which should be called for all
 *                                  @eref{Callpath} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCallpathCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_Callpath callpathCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{Group} definition record.
 *
 *  The group definition.
 *
 *  @param userData        User data as set by
 *                         @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                         @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self            The unique identifier for this @eref{Group}
 *                         definition.
 *  @param name            Name of this group References a @eref{String}
 *                         definition.
 *  @param groupType       The type of this group. Since version 1.2.
 *  @param paradigm        The paradigm of this communication group. Since
 *                         version 1.2.
 *  @param groupFlags      Flags for this group. Since version 1.2.
 *  @param numberOfMembers The number of members in this group.
 *  @param members         The identifiers of the group members.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_Group)(void* userData,
                                                                OTF2_GroupRef self,
                                                                OTF2_StringRef name,
                                                                OTF2_GroupType groupType,
                                                                OTF2_Paradigm paradigm,
                                                                OTF2_GroupFlag groupFlags,
                                                                uint32_t numberOfMembers,
                                                                const uint64_t* members);

/** @brief Registers the callback for the @eref{Group} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param groupCallback            Function which should be called for all
 *                                  @eref{Group} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetGroupCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                              OTF2_GlobalDefReaderCallback_Group groupCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{MetricMember} definition record.
 *
 *  A metric is defined by a @e MetricMember definition. A metric member
 *  is always a member of a metric class. Therefore, a single metric
 *  is a special case of a metric class with only one member. It is
 *  not allowed to reference a metric member ID in a @eref{Metric}
 *  event, but only metric class IDs.
 *
 *  @param userData    User data as set by
 *                     @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                     @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self        The unique identifier for this @eref{MetricMember}
 *                     definition.
 *  @param name        Name of the metric. References a @eref{String} definition.
 *  @param description Description of the metric. References a @eref{String}
 *                     definition.
 *  @param metricType  Metric type: PAPI, etc.
 *  @param metricMode  Metric mode: accumulative, fix, relative, etc.
 *  @param valueType   Type of the value. Only @eref{OTF2_TYPE_INT64},
 *                     @eref{OTF2_TYPE_UINT64}, and @eref{OTF2_TYPE_DOUBLE}
 *                     are valid types. If this metric member is recorded in
 *                     a @eref{Metric} event, than this type and the type in
 *                     the event must match.
 *  @param base        The recorded values should be handled in this given base,
 *                     either binary or decimal. This information can be used
 *                     if the value needs to be scaled.
 *  @param exponent    The values inside the Metric events should be scaled by
 *                     the factor base^exponent, to at the value in its base
 *                     unit. For example, if the metric values come in as
 *                     KiBi, than the base should be @eref{OTF2_BASE_BINARY}
 *                     and the exponent 10. Than the writer does not need to
 *                     scale the values up to bytes, but can directly write
 *                     the KiBi values into the Metric event. At reading
 *                     time, the reader can apply the scaling factor to at
 *                     the value in its base unit, ie. in bytes.
 *  @param unit        Unit of the metric. This needs to be the scale free base
 *                     unit, ie. "bytes", "operations", or "seconds". In
 *                     particular this unit should not have any scale prefix.
 *                     References a @eref{String} definition.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_MetricMember)(void* userData,
                                                                       OTF2_MetricMemberRef self,
                                                                       OTF2_StringRef name,
                                                                       OTF2_StringRef description,
                                                                       OTF2_MetricType metricType,
                                                                       OTF2_MetricMode metricMode,
                                                                       OTF2_Type valueType,
                                                                       OTF2_Base base,
                                                                       int64_t exponent,
                                                                       OTF2_StringRef unit);

/** @brief Registers the callback for the @eref{MetricMember} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param metricMemberCallback     Function which should be called for all
 *                                  @eref{MetricMember} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetMetricMemberCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_MetricMember metricMemberCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{MetricClass} definition record.
 *
 *  For a metric class it is implicitly given that the event stream that
 *  records the metric is also the scope. A metric class can contain
 *  multiple different metrics.
 *
 *  @param userData         User data as set by
 *                          @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                          @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self             The unique identifier for this @eref{MetricClass}
 *                          definition.
 *  @param numberOfMetrics  Number of metrics within the set.
 *  @param metricMembers    List of metric members. References a
 *                          @eref{MetricMember} definition.
 *  @param metricOccurrence Defines occurrence of a metric set.
 *  @param recorderKind     What kind of locations will record this metric class,
 *                          or will this metric class only be recorded by
 *                          metric instances. Since version 1.2.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_MetricClass)(void* userData,
                                                                      OTF2_MetricRef self,
                                                                      uint8_t numberOfMetrics,
                                                                      const OTF2_MetricMemberRef* metricMembers,
                                                                      OTF2_MetricOccurrence metricOccurrence,
                                                                      OTF2_RecorderKind recorderKind);

/** @brief Registers the callback for the @eref{MetricClass} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param metricClassCallback      Function which should be called for all
 *                                  @eref{MetricClass} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetMetricClassCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_MetricClass metricClassCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{MetricInstance} definition record.
 *
 *  A @e MetricInstance is used to define metrics that are recorded at one
 *  location for multiple locations or for another location. The
 *  occurrence of a metric instance is implicitly of type
 *  @eref{OTF2_METRIC_ASYNCHRONOUS}.
 *
 *  @param userData    User data as set by
 *                     @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                     @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self        The unique identifier for this @eref{MetricClass}
 *                     definition.
 *  @param metricClass The instanced @eref{MetricClass}. This metric class must
 *                     be of kind @eref{OTF2_RECORDER_KIND_ABSTRACT}.
 *                     References a @eref{MetricClass}, or a
 *                     @eref{MetricInstance} definition.
 *  @param recorder    Recorder of the metric: location ID. References a
 *                     @eref{Location} definition.
 *  @param metricScope Defines type of scope: location, location group, system
 *                     tree node, or a generic group of locations.
 *  @param scope       Scope of metric: ID of a location, location group, system
 *                     tree node, or a generic group of locations.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_MetricInstance)(void* userData,
                                                                         OTF2_MetricRef self,
                                                                         OTF2_MetricRef metricClass,
                                                                         OTF2_LocationRef recorder,
                                                                         OTF2_MetricScope metricScope,
                                                                         uint64_t scope);

/** @brief Registers the callback for the @eref{MetricInstance} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param metricInstanceCallback   Function which should be called for all
 *                                  @eref{MetricInstance} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetMetricInstanceCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_MetricInstance metricInstanceCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{Comm} definition record.
 *
 *  The communicator definition.
 *
 *  @param userData User data as set by
 *                  @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                  @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self     The unique identifier for this @eref{Comm} definition.
 *  @param name     The name given by calling MPI_Comm_set_name on this
 *                  communicator. Or the empty name to indicate that no name
 *                  was given. References a @eref{String} definition.
 *  @param group    The describing MPI group of this MPI communicator  The group
 *                  needs to be of type @eref{OTF2_GROUP_TYPE_COMM_GROUP} or
 *                  @eref{OTF2_GROUP_TYPE_COMM_SELF}. References a
 *                  @eref{Group} definition.
 *  @param parent   The parent MPI communicator from which this communicator was
 *                  created, if any. Use @eref{OTF2_UNDEFINED_COMM} to
 *                  indicate no parent. References a @eref{Comm} definition.
 *  @param flags    Special characteristics of this communicator. Since version
 *                  3.0.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_Comm)(void* userData,
                                                               OTF2_CommRef self,
                                                               OTF2_StringRef name,
                                                               OTF2_GroupRef group,
                                                               OTF2_CommRef parent,
                                                               OTF2_CommFlag flags);

/** @brief Registers the callback for the @eref{Comm} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param commCallback             Function which should be called for all
 *                                  @eref{Comm} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCommCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                             OTF2_GlobalDefReaderCallback_Comm commCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{Parameter} definition record.
 *
 *  The parameter definition.
 *
 *  @param userData      User data as set by
 *                       @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                       @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self          The unique identifier for this @eref{Parameter}
 *                       definition.
 *  @param name          Name of the parameter (variable name etc.) References a
 *                       @eref{String} definition.
 *  @param parameterType Type of the parameter, @eref{OTF2_ParameterType} for
 *                       possible types.
 *
 *  @since Version 1.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_Parameter)(void* userData,
                                                                    OTF2_ParameterRef self,
                                                                    OTF2_StringRef name,
                                                                    OTF2_ParameterType parameterType);

/** @brief Registers the callback for the @eref{Parameter} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param parameterCallback        Function which should be called for all
 *                                  @eref{Parameter} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetParameterCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_Parameter parameterCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{RmaWin} definition record.
 *
 *  A window defines the communication context for any remote-memory
 *  access operation.
 *
 *  @param userData User data as set by
 *                  @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                  @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self     The unique identifier for this @eref{RmaWin} definition.
 *  @param name     Name, e.g. 'GASPI Queue 1', 'NVidia Card 2', etc.. References
 *                  a @eref{String} definition.
 *  @param comm     Communicator object used to create the window. References a
 *                  @eref{Comm} definition.
 *  @param flags    Special characteristics of this RMA window. Since version
 *                  3.0.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_RmaWin)(void* userData,
                                                                 OTF2_RmaWinRef self,
                                                                 OTF2_StringRef name,
                                                                 OTF2_CommRef comm,
                                                                 OTF2_RmaWinFlag flags);

/** @brief Registers the callback for the @eref{RmaWin} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param rmaWinCallback           Function which should be called for all
 *                                  @eref{RmaWin} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetRmaWinCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                               OTF2_GlobalDefReaderCallback_RmaWin rmaWinCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{MetricClassRecorder} definition record.
 *
 *  The metric class recorder definition.
 *
 *  @param userData User data as set by
 *                  @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                  @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param metric   Parent @eref{MetricClass}, or @eref{MetricInstance}
 *                  definition to which this one is a supplementary
 *                  definition. References a @eref{MetricClass}, or a
 *                  @eref{MetricInstance} definition.
 *  @param recorder The location which recorded the referenced metric class.
 *                  References a @eref{Location} definition.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_MetricClassRecorder)(void* userData,
                                                                              OTF2_MetricRef metric,
                                                                              OTF2_LocationRef recorder);

/** @brief Registers the callback for the @eref{MetricClassRecorder} definition.
 *
 *  @param globalDefReaderCallbacks    Struct for all callbacks.
 *  @param metricClassRecorderCallback Function which should be called for all
 *                                     @eref{MetricClassRecorder}
 *                                     definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetMetricClassRecorderCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_MetricClassRecorder metricClassRecorderCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{SystemTreeNodeProperty} definition record.
 *
 *  An arbitrary key/value property for a @eref{SystemTreeNode}
 *  definition.
 *
 *  @param userData       User data as set by
 *                        @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                        @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param systemTreeNode Parent @eref{SystemTreeNode} definition to which this
 *                        one is a supplementary definition. References a
 *                        @eref{SystemTreeNode} definition.
 *  @param name           Name of the property. References a @eref{String}
 *                        definition.
 *  @param type           The type of this property. Since version 2.0.
 *  @param value          The value of this property. Since version 2.0.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_SystemTreeNodeProperty)(void* userData,
                                                                                 OTF2_SystemTreeNodeRef systemTreeNode,
                                                                                 OTF2_StringRef name,
                                                                                 OTF2_Type type,
                                                                                 OTF2_AttributeValue value);

/** @brief Registers the callback for the @eref{SystemTreeNodeProperty} definition.
 *
 *  @param globalDefReaderCallbacks       Struct for all callbacks.
 *  @param systemTreeNodePropertyCallback Function which should be called for all
 *                                        @eref{SystemTreeNodeProperty}
 *                                        definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodePropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_SystemTreeNodeProperty systemTreeNodePropertyCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{SystemTreeNodeDomain} definition record.
 *
 *  The system tree node domain definition.
 *
 *  @param userData         User data as set by
 *                          @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                          @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param systemTreeNode   Parent @eref{SystemTreeNode} definition to which this
 *                          one is a supplementary definition. References a
 *                          @eref{SystemTreeNode} definition.
 *  @param systemTreeDomain The domain in which the referenced
 *                          @eref{SystemTreeNode} operates in.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_SystemTreeNodeDomain)(void* userData,
                                                                               OTF2_SystemTreeNodeRef systemTreeNode,
                                                                               OTF2_SystemTreeDomain systemTreeDomain);

/** @brief Registers the callback for the @eref{SystemTreeNodeDomain} definition.
 *
 *  @param globalDefReaderCallbacks     Struct for all callbacks.
 *  @param systemTreeNodeDomainCallback Function which should be called for all
 *                                      @eref{SystemTreeNodeDomain}
 *                                      definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodeDomainCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_SystemTreeNodeDomain systemTreeNodeDomainCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{LocationGroupProperty} definition record.
 *
 *  An arbitrary key/value property for a @eref{LocationGroup} definition.
 *
 *  @param userData      User data as set by
 *                       @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                       @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param locationGroup Parent @eref{LocationGroup} definition to which this one
 *                       is a supplementary definition. References a
 *                       @eref{LocationGroup} definition.
 *  @param name          Name of the property. References a @eref{String}
 *                       definition.
 *  @param type          The type of this property. Since version 2.0.
 *  @param value         The value of this property. Since version 2.0.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_LocationGroupProperty)(void* userData,
                                                                                OTF2_LocationGroupRef locationGroup,
                                                                                OTF2_StringRef name,
                                                                                OTF2_Type type,
                                                                                OTF2_AttributeValue value);

/** @brief Registers the callback for the @eref{LocationGroupProperty} definition.
 *
 *  @param globalDefReaderCallbacks      Struct for all callbacks.
 *  @param locationGroupPropertyCallback Function which should be called for all
 *                                       @eref{LocationGroupProperty}
 *                                       definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetLocationGroupPropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_LocationGroupProperty locationGroupPropertyCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{LocationProperty} definition record.
 *
 *  An arbitrary key/value property for a @eref{Location} definition.
 *
 *  @param userData    User data as set by
 *                     @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                     @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param location    Parent @eref{Location} definition to which this one is a
 *                     supplementary definition. References a @eref{Location}
 *                     definition.
 *  @param name        Name of the property. References a @eref{String}
 *                     definition.
 *  @param type        The type of this property. Since version 2.0.
 *  @param value       The value of this property. Since version 2.0.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_LocationProperty)(void* userData,
                                                                           OTF2_LocationRef location,
                                                                           OTF2_StringRef name,
                                                                           OTF2_Type type,
                                                                           OTF2_AttributeValue value);

/** @brief Registers the callback for the @eref{LocationProperty} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param locationPropertyCallback Function which should be called for all
 *                                  @eref{LocationProperty} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetLocationPropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_LocationProperty locationPropertyCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{CartDimension} definition record.
 *
 *  Each dimension in a Cartesian topology is composed of a global ID, a
 *  name, its size, and whether it is periodic or not.
 *
 *  @param userData        User data as set by
 *                         @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                         @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self            The unique identifier for this @eref{CartDimension}
 *                         definition.
 *  @param name            The name of the Cartesian topology dimension.
 *                         References a @eref{String} definition.
 *  @param size            The size of the Cartesian topology dimension.
 *  @param cartPeriodicity Periodicity of the Cartesian topology dimension.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_CartDimension)(void* userData,
                                                                        OTF2_CartDimensionRef self,
                                                                        OTF2_StringRef name,
                                                                        uint32_t size,
                                                                        OTF2_CartPeriodicity cartPeriodicity);

/** @brief Registers the callback for the @eref{CartDimension} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param cartDimensionCallback    Function which should be called for all
 *                                  @eref{CartDimension} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCartDimensionCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CartDimension cartDimensionCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{CartTopology} definition record.
 *
 *  Each topology is described by a global ID, a reference to its name, a
 *  reference to a communicator, the number of dimensions, and
 *  references to those dimensions. The topology type is defined by
 *  the paradigm of the group referenced by the associated
 *  communicator.
 *
 *  @param userData           User data as set by
 *                            @eref{OTF2_Reader_RegisterGlobalDefCallbacks}
 *                            or @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self               The unique identifier for this @eref{CartTopology}
 *                            definition.
 *  @param name               The name of the topology. References a
 *                            @eref{String} definition.
 *  @param communicator       Communicator object used to create the topology.
 *                            References a @eref{Comm} definition.
 *  @param numberOfDimensions Number of dimensions.
 *  @param cartDimensions     The dimensions of this topology. References a
 *                            @eref{CartDimension} definition.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_CartTopology)(void* userData,
                                                                       OTF2_CartTopologyRef self,
                                                                       OTF2_StringRef name,
                                                                       OTF2_CommRef communicator,
                                                                       uint8_t numberOfDimensions,
                                                                       const OTF2_CartDimensionRef* cartDimensions);

/** @brief Registers the callback for the @eref{CartTopology} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param cartTopologyCallback     Function which should be called for all
 *                                  @eref{CartTopology} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCartTopologyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CartTopology cartTopologyCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{CartCoordinate} definition record.
 *
 *  Defines the coordinate of the location referenced by the given rank
 *  (w.r.t. the communicator associated to the topology) in the
 *  referenced topology.
 *
 *  @param userData           User data as set by
 *                            @eref{OTF2_Reader_RegisterGlobalDefCallbacks}
 *                            or @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param cartTopology       Parent @eref{CartTopology} definition to which this
 *                            one is a supplementary definition. References a
 *                            @eref{CartTopology} definition.
 *  @param rank               The rank w.r.t. the communicator associated to the
 *                            topology referencing this coordinate.
 *  @param numberOfDimensions Number of dimensions.
 *  @param coordinates        Coordinates, indexed by dimension.
 *
 *  @since Version 1.3
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_CartCoordinate)(void* userData,
                                                                         OTF2_CartTopologyRef cartTopology,
                                                                         uint32_t rank,
                                                                         uint8_t numberOfDimensions,
                                                                         const uint32_t* coordinates);

/** @brief Registers the callback for the @eref{CartCoordinate} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param cartCoordinateCallback   Function which should be called for all
 *                                  @eref{CartCoordinate} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCartCoordinateCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CartCoordinate cartCoordinateCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{SourceCodeLocation} definition record.
 *
 *  The definition of a source code location as tuple of the corresponding
 *  file name and line number.
 *
 *  When used to attach source code annotations to events, use the
 *  @eref{OTF2_AttributeList} with an @eref{Attribute} definition
 *  named @c "SOURCE_CODE_LOCATION" and of type
 *  @eref{OTF2_TYPE_SOURCE_CODE_LOCATION}.
 *
 *  @param userData   User data as set by
 *                    @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                    @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self       The unique identifier for this @eref{SourceCodeLocation}
 *                    definition.
 *  @param file       The name of the file for the source code location.
 *                    References a @eref{String} definition.
 *  @param lineNumber The line number for the source code location.
 *
 *  @since Version 1.5
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_SourceCodeLocation)(void* userData,
                                                                             OTF2_SourceCodeLocationRef self,
                                                                             OTF2_StringRef file,
                                                                             uint32_t lineNumber);

/** @brief Registers the callback for the @eref{SourceCodeLocation} definition.
 *
 *  @param globalDefReaderCallbacks   Struct for all callbacks.
 *  @param sourceCodeLocationCallback Function which should be called for all
 *                                    @eref{SourceCodeLocation} definitions.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetSourceCodeLocationCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_SourceCodeLocation sourceCodeLocationCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{CallingContext} definition record.
 *
 *  Defines a node in the calling context tree. These nodes are referenced
 *  in the @eref{CallingContextSample}, @eref{CallingContextEnter},
 *  and @eref{CallingContextLeave} events.
 *
 *  The referenced @eref{CallingContext} node in these events form a path
 *  which represents the calling context at this time.  This path will
 *  be partitioned into at most three sub-paths by the @a
 *  unwindDistance attribute.  For the @eref{CallingContextLeave}
 *  event, the @a unwindDistance is defined to be 1.
 *
 *  Starting from the referenced @eref{CallingContext} node, the first @f$
 *  N \ge 0 @f$ nodes were newly entered regions since the previous
 *  calling context event.  The next node is a region which was not
 *  left but made progress since the previous calling context event.
 *  All other nodes did not make progress at all, and thus the regions
 *  were neither left nor entered again.  The @a unwindDistance is
 *  then @f$ N + 1 @f$. In case the @p unwindDistance is @f$ 0 @f$,
 *  there are neither newly entered regions nor regions which made
 *  progress.
 *
 *  It is guaranteed, that the node referenced by the @a unwindDistance
 *  exists in the previous and current calling context.  All
 *  descendants of this node's child in the previous calling context
 *  were left since the previous calling context event.
 *
 *  It is valid that this node is the
 *  @eref{OTF2_UNDEFINED_CALLING_CONTEXT} node and that this node is
 *  already reached after @a unwindDistance @f$ - 1 @f$ steps.  In the
 *  latter case, there exists no region which made progress, all
 *  regions in the previous calling context were left and all regions
 *  in the current calling context were newly entered.
 *
 *  Note that for @eref{CallingContextLeave} events, the parent of the
 *  referenced @eref{CallingContext} must be used as the previous
 *  calling context for the next event.
 *
 *  Regions which were entered with a @eref{CallingContextEnter} event
 *  form an upper bound for the unwind distance, i.e., the @a
 *  unwindDistance points either to the parent of the last such
 *  entered region, or a node which is a descendant to this parent.
 *
 *  To summarize, an @a unwindDistance of @f$ 0 @f$ means that no regions
 *  were left, newly entered, or made any progress.  An @a
 *  unwindDistance of @f$ 1 @f$ means that some regions were left
 *  regarding the previous calling context, no regions were newly
 *  entered, and there was progress in the region of the first node.
 *  An @a unwindDistance greater than @f$ 1 @f$ means that some
 *  regions were left regarding the previous calling context, there
 *  was progress in one region, and the first @a unwindDistance @f$ -
 *  1 @f$ regions were newly entered.
 *
 *  @param userData           User data as set by
 *                            @eref{OTF2_Reader_RegisterGlobalDefCallbacks}
 *                            or @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self               The unique identifier for this
 *                            @eref{CallingContext} definition.
 *  @param region             The region. References a @eref{Region} definition.
 *  @param sourceCodeLocation The absolute source code location of this calling
 *                            context. References a @eref{SourceCodeLocation}
 *                            definition.
 *  @param parent             Parent ID of this context. References a
 *                            @eref{CallingContext} definition.
 *
 *  @since Version 1.5
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_CallingContext)(void* userData,
                                                                         OTF2_CallingContextRef self,
                                                                         OTF2_RegionRef region,
                                                                         OTF2_SourceCodeLocationRef sourceCodeLocation,
                                                                         OTF2_CallingContextRef parent);

/** @brief Registers the callback for the @eref{CallingContext} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param callingContextCallback   Function which should be called for all
 *                                  @eref{CallingContext} definitions.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCallingContextCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CallingContext callingContextCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{CallingContextProperty} definition record.
 *
 *  An arbitrary key/value property for a @eref{CallingContext}
 *  definition.
 *
 *  @param userData       User data as set by
 *                        @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                        @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param callingContext Parent @eref{CallingContext} definition to which this
 *                        one is a supplementary definition. References a
 *                        @eref{CallingContext} definition.
 *  @param name           Property name. References a @eref{String} definition.
 *  @param type           The type of this property. Must match with the defined
 *                        type of the @a property.
 *  @param value          The value of this property.
 *
 *  @since Version 2.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_CallingContextProperty)(void* userData,
                                                                                 OTF2_CallingContextRef callingContext,
                                                                                 OTF2_StringRef name,
                                                                                 OTF2_Type type,
                                                                                 OTF2_AttributeValue value);

/** @brief Registers the callback for the @eref{CallingContextProperty} definition.
 *
 *  @param globalDefReaderCallbacks       Struct for all callbacks.
 *  @param callingContextPropertyCallback Function which should be called for all
 *                                        @eref{CallingContextProperty}
 *                                        definitions.
 *
 *  @since Version 2.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCallingContextPropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CallingContextProperty callingContextPropertyCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{InterruptGenerator} definition record.
 *
 *  Defines an interrupt generator which periodically triggers
 *  @eref{CallingContextSample} events.  If the mode of the interrupt
 *  generator is set to @eref{OTF2_INTERRUPT_GENERATOR_MODE_TIME}, the
 *  generator produces interrupts which are uniformly distributed over
 *  time, and the unit of the period is implicitly in seconds.  If the
 *  mode is @eref{OTF2_INTERRUPT_GENERATOR_MODE_COUNT}, the interrupt
 *  is triggered if a specific counter threshold is reached in the
 *  system.  Therefore these samples are unlikely to be uniformly
 *  distributed over time.  The unit of the period is then implicitly
 *  a number (threshold value).
 *
 *  The interrupts period in base unit (which is implicitly seconds or
 *  number, based on the @p mode) is derived out of the @p base, the
 *  @p exponent, and the @p period attributes by this formula:
 *
 *  base-period = period x base^exponent
 *
 *  @param userData               User data as set by
 *                                @eref{OTF2_Reader_RegisterGlobalDefCallbacks}
 *                                or
 *                                @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self                   The unique identifier for this
 *                                @eref{InterruptGenerator} definition.
 *  @param name                   The name of this interrupt generator.
 *                                References a @eref{String} definition.
 *  @param interruptGeneratorMode Mode of the interrupt generator.
 *  @param base                   The base for the period calculation.
 *  @param exponent               The exponent for the period calculation.
 *  @param period                 The period this interrupt generator generates
 *                                interrupts.
 *
 *  @since Version 1.5
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_InterruptGenerator)(
  void* userData,
  OTF2_InterruptGeneratorRef self,
  OTF2_StringRef name,
  OTF2_InterruptGeneratorMode interruptGeneratorMode,
  OTF2_Base base,
  int64_t exponent,
  uint64_t period);

/** @brief Registers the callback for the @eref{InterruptGenerator} definition.
 *
 *  @param globalDefReaderCallbacks   Struct for all callbacks.
 *  @param interruptGeneratorCallback Function which should be called for all
 *                                    @eref{InterruptGenerator} definitions.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetInterruptGeneratorCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_InterruptGenerator interruptGeneratorCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{IoFileProperty} definition record.
 *
 *  Extensible annotation for the polymorphic @eref{IoFile} definition
 *  class.
 *
 *  The tuple (@a ioFile, @a name) must be unique.
 *
 *  @param userData User data as set by
 *                  @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                  @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param ioFile   Parent @eref{IoRegularFile} definition to which this one is a
 *                  supplementary definition. References a
 *                  @eref{IoRegularFile} definition.
 *  @param name     Property name. References a @eref{String} definition.
 *  @param type     The type of this property.
 *  @param value    The value of this property.
 *
 *  @since Version 2.1
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_IoFileProperty)(void* userData,
                                                                         OTF2_IoFileRef ioFile,
                                                                         OTF2_StringRef name,
                                                                         OTF2_Type type,
                                                                         OTF2_AttributeValue value);

/** @brief Registers the callback for the @eref{IoFileProperty} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param ioFilePropertyCallback   Function which should be called for all
 *                                  @eref{IoFileProperty} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoFilePropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoFileProperty ioFilePropertyCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{IoRegularFile} definition record.
 *
 *  Defines a regular file from which an @eref{IoHandle} can be created.
 *
 *  This definition is member of the polymorphic @eref{IoFile} definition
 *  class. All definitions of this polymorphic definition class share
 *  the same global identifier namespace.
 *
 *  @param userData User data as set by
 *                  @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                  @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self     The unique identifier for this @eref{IoRegularFile}
 *                  definition.
 *  @param name     Name of the file. References a @eref{String} definition.
 *  @param scope    Defines the physical scope of this @eref{IoRegularFile} in
 *                  the system tree. E.g., two @eref{IoRegularFile}
 *                  definitions with the same name but different @p scope
 *                  values are physically different, thus I/O operations
 *                  through @eref{IoHandle}s do not operate on the same file.
 *                  References a @eref{SystemTreeNode} definition.
 *
 *  @since Version 2.1
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_IoRegularFile)(void* userData,
                                                                        OTF2_IoFileRef self,
                                                                        OTF2_StringRef name,
                                                                        OTF2_SystemTreeNodeRef scope);

/** @brief Registers the callback for the @eref{IoRegularFile} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param ioRegularFileCallback    Function which should be called for all
 *                                  @eref{IoRegularFile} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoRegularFileCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoRegularFile ioRegularFileCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{IoDirectory} definition record.
 *
 *  Defines a directory from which an @eref{IoHandle} can be created.
 *
 *  This definition is member of the polymorphic @eref{IoFile} definition
 *  class. All definitions of this polymorphic definition class share
 *  the same global identifier namespace.
 *
 *  @param userData User data as set by
 *                  @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                  @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self     The unique identifier for this @eref{IoRegularFile}
 *                  definition.
 *  @param name     Name of the directory. References a @eref{String} definition.
 *  @param scope    Defines the physical scope of this @eref{IoDirectory} in the
 *                  system tree. E.g., two @eref{IoDirectory} definitions
 *                  with the same name but different @p scope values are
 *                  physically different, thus I/O operations through
 *                  @eref{IoHandle}s do not operate on the same directory.
 *                  References a @eref{SystemTreeNode} definition.
 *
 *  @since Version 2.1
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_IoDirectory)(void* userData,
                                                                      OTF2_IoFileRef self,
                                                                      OTF2_StringRef name,
                                                                      OTF2_SystemTreeNodeRef scope);

/** @brief Registers the callback for the @eref{IoDirectory} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param ioDirectoryCallback      Function which should be called for all
 *                                  @eref{IoDirectory} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoDirectoryCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoDirectory ioDirectoryCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{IoHandle} definition record.
 *
 *  Defines an I/O handle which will be used by subsequent I/O operations.
 *  I/O operations can only be applied to @emph{active} I/O handles.
 *  An I/O handle gets @emph{active} either if it was marked with the
 *  @eref{OTF2_IO_HANDLE_FLAG_PRE_CREATED} flag, after it was
 *  referenced in an @eref{IoCreateHandle} event, or it was referenced
 *  in the @a newHandle attribute of an @eref{IoDuplicateHandle}
 *  event. It gets @emph{inactive} if it was referenced in an
 *  @eref{IoDestroyHandle} event. This life cycle can be repeated
 *  indefinitely. Though the @eref{OTF2_IO_HANDLE_FLAG_PRE_CREATED}
 *  flag is unset after a @eref{IoDuplicateHandle} event. All
 *  @eref{Location}s of a @eref{LocationGroup} have access to an
 *  @emph{active} @eref{IoHandle}, regardless which @eref{Location} of
 *  the @eref{LocationGroup} activated the @eref{IoHandle}.
 *
 *  @param userData      User data as set by
 *                       @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                       @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self          The unique identifier for this @eref{IoHandle}
 *                       definition.
 *  @param name          Handle name. References a @eref{String} definition.
 *  @param file          File identifier. References a @eref{IoRegularFile}, or a
 *                       @eref{IoDirectory} definition.
 *  @param ioParadigm    The I/O paradigm. References a @eref{IoParadigm}
 *                       definition.
 *  @param ioHandleFlags Special characteristics of this handle.
 *  @param comm          Scope of the file handle. This scope defines which
 *                       process can access this file via this handle and
 *                       also defines the collective context for this handle.
 *                       References a @eref{Comm} definition.
 *  @param parent        Parent, in case this I/O handle was created and operated
 *                       by an higher-level I/O paradigm. References a
 *                       @eref{IoHandle} definition.
 *
 *  @since Version 2.1
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_IoHandle)(void* userData,
                                                                   OTF2_IoHandleRef self,
                                                                   OTF2_StringRef name,
                                                                   OTF2_IoFileRef file,
                                                                   OTF2_IoParadigmRef ioParadigm,
                                                                   OTF2_IoHandleFlag ioHandleFlags,
                                                                   OTF2_CommRef comm,
                                                                   OTF2_IoHandleRef parent);

/** @brief Registers the callback for the @eref{IoHandle} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param ioHandleCallback         Function which should be called for all
 *                                  @eref{IoHandle} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoHandleCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoHandle ioHandleCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{IoPreCreatedHandleState} definition record.
 *
 *  Provide the I/O access mode and status flags for @emph{pre-created}
 *  @eref{IoHandle}s.
 *
 *  Only allowed once for a @eref{IoHandle} definition with the
 *  @eref{OTF2_IO_HANDLE_FLAG_PRE_CREATED} flag set.
 *
 *  @param userData    User data as set by
 *                     @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                     @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param ioHandle    Parent @eref{IoHandle} definition to which this one is a
 *                     supplementary definition. References a @eref{IoHandle}
 *                     definition.
 *  @param mode        The access mode of the @emph{pre-created} @eref{IoHandle}.
 *  @param statusFlags The status flags of the @emph{pre-created}
 *                     @eref{IoHandle}.
 *
 *  @since Version 2.1
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_IoPreCreatedHandleState)(void* userData,
                                                                                  OTF2_IoHandleRef ioHandle,
                                                                                  OTF2_IoAccessMode mode,
                                                                                  OTF2_IoStatusFlag statusFlags);

/** @brief Registers the callback for the @eref{IoPreCreatedHandleState} definition.
 *
 *  @param globalDefReaderCallbacks        Struct for all callbacks.
 *  @param ioPreCreatedHandleStateCallback Function which should be called for
 *                                         all @eref{IoPreCreatedHandleState}
 *                                         definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoPreCreatedHandleStateCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoPreCreatedHandleState ioPreCreatedHandleStateCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{CallpathParameter} definition record.
 *
 *  A parameter for a callpath definition.
 *
 *  @param userData  User data as set by
 *                   @eref{OTF2_Reader_RegisterGlobalDefCallbacks} or
 *                   @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param callpath  Parent @eref{Callpath} definition to which this one is a
 *                   supplementary definition. References a @eref{Callpath}
 *                   definition.
 *  @param parameter The parameter of this callpath. References a
 *                   @eref{Parameter} definition.
 *  @param type      The type of the attribute value. Must match the type of the
 *                   parameter.
 *  @param value     The value of the parameter for this callpath.
 *
 *  @since Version 2.2
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_CallpathParameter)(void* userData,
                                                                            OTF2_CallpathRef callpath,
                                                                            OTF2_ParameterRef parameter,
                                                                            OTF2_Type type,
                                                                            OTF2_AttributeValue value);

/** @brief Registers the callback for the @eref{CallpathParameter} definition.
 *
 *  @param globalDefReaderCallbacks  Struct for all callbacks.
 *  @param callpathParameterCallback Function which should be called for all
 *                                   @eref{CallpathParameter} definitions.
 *
 *  @since Version 2.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCallpathParameterCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CallpathParameter callpathParameterCallback);

/** @brief Function pointer definition for the callback which is triggered by
 *         a @eref{InterComm} definition record.
 *
 *  The inter-communicator definition.
 *
 *  @param userData           User data as set by
 *                            @eref{OTF2_Reader_RegisterGlobalDefCallbacks}
 *                            or @eref{OTF2_GlobalDefReader_SetCallbacks}.
 *  @param self               The unique identifier for this @eref{Comm}
 *                            definition.
 *  @param name               The name given by calling MPI_Comm_set_name on this
 *                            communicator. Or the empty name to indicate
 *                            that no name was given. References a
 *                            @eref{String} definition.
 *  @param groupA             One of the two MPI process groups in the
 *                            intercommunicator.  The group needs to be of
 *                            type @eref{OTF2_GROUP_TYPE_COMM_GROUP} or
 *                            @eref{OTF2_GROUP_TYPE_COMM_SELF}. References a
 *                            @eref{Group} definition.
 *  @param groupB             The other of the two MPI process groups in the
 *                            intercommunicator.  The group needs to be of
 *                            type @eref{OTF2_GROUP_TYPE_COMM_GROUP} or
 *                            @eref{OTF2_GROUP_TYPE_COMM_SELF}. References a
 *                            @eref{Group} definition.
 *  @param commonCommunicator The common peer MPI communicator used to create
 *                            this inter-communicator. Use
 *                            @eref{OTF2_UNDEFINED_COMM} if no such
 *                            communicator was used. References a
 *                            @eref{Comm}, or a @eref{InterComm} definition.
 *  @param flags              Special characteristics of this communicator.
 *
 *  @since Version 3.0
 *
 *  @return @eref{OTF2_CALLBACK_SUCCESS} or @eref{OTF2_CALLBACK_INTERRUPT}.
 */
typedef OTF2_CallbackCode (*OTF2_GlobalDefReaderCallback_InterComm)(void* userData,
                                                                    OTF2_CommRef self,
                                                                    OTF2_StringRef name,
                                                                    OTF2_GroupRef groupA,
                                                                    OTF2_GroupRef groupB,
                                                                    OTF2_CommRef commonCommunicator,
                                                                    OTF2_CommFlag flags);

/** @brief Registers the callback for the @eref{InterComm} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param interCommCallback        Function which should be called for all
 *                                  @eref{InterComm} definitions.
 *
 *  @since Version 3.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetInterCommCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_InterComm interCommCallback);



#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !OTF2_GLOBAL_DEF_READER_CALLBACKS_H */
