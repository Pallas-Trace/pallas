#include <stdio.h>
#include <stdlib.h>

#include "pallas/pallas.h"
#include "otf2/OTF2_SnapReader.h"
#include "otf2/otf2.h"

/** @brief Return the location ID of the reading related location.
 *
 *  @param reader         Reader object which reads the snapshot events from its
 *                        buffer.
 *  @param[out] location  ID of the location.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_SnapReader_GetLocationID(const OTF2_SnapReader* reader, OTF2_LocationRef* location) {
  NOT_IMPLEMENTED;
}

/** @brief Sets the callback functions for the given reader object. Every time
 *         when OTF2 reads a record, a callback function is called and the records
 *         data is passed to this function. Therefore the programmer needs to set
 *         function pointers at the "callbacks" struct for the record type he wants
 *         to read.
 *
 *  These callbacks are ignored, if the snapshot events are read by an global
 *  snapshot event reader.
 *
 *  @param reader          Reader object which reads the snapshot events from
 *                         its buffer.
 *  @param callbacks       Struct which holds a function pointer for each
 *                         record type. @eref{OTF2_SnapReaderCallbacks_New}.
 *  @param userData        Data passed as argument @a userData to the record
 *                         callbacks.
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_SnapReader_SetCallbacks(OTF2_SnapReader* reader,
                                            const OTF2_SnapReaderCallbacks* callbacks,
                                            void* userData) {
  NOT_IMPLEMENTED;
}

/** @brief Seek jumps to start of latest snapshot that was made
 *         before a given time 'req_time'.
 *
 *  @param reader         Reader object which reads the snapshot events from its
 *                        buffer.
 *  @param req_time       Requested time (see above)
 *  @param found          returns if a matching snapshot was found
 *
 *  @since Version 1.2
 *
 *  @return OTF2_ErrorCode with !=OTF2_SUCCESS if there was an error.
 */
OTF2_ErrorCode OTF2_SnapReader_Seek(OTF2_SnapReader* reader, uint64_t req_time, bool* found) {
  NOT_IMPLEMENTED;
}

/** @brief After callback registration, the local snapshot events could be read
 *         with the following function. Reads at most @a recordsToRead records.
 *         The reader indicates that it reached the end of the trace by just
 *         reading less records than requested.
 *
 *  @param reader           Reader object which reads the snapshot events from
 *                          its buffer.
 *  @param recordsToRead    How many records can be read next.
 *  @param[out] recordsRead Return how many records where really read.
 *
 *  @since Version 1.2
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_SnapReader_ReadSnapshots(OTF2_SnapReader* reader, uint64_t recordsToRead, uint64_t* recordsRead) {
  NOT_IMPLEMENTED;
}
