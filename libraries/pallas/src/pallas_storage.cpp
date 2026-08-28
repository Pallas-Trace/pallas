/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <sys/stat.h>
#include <unistd.h>
#include <zstd.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>

#ifdef WITH_ZFP
#include <zfp.h>
#endif
#ifdef WITH_SZ
#include <sz.h>
#endif

#include "pallas/pallas.h"
#include "pallas/pallas_file_format.h"

#ifdef BMARK
#include "pallas/linked_vector/pallas_bmark.h"
#endif

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_parameter_handler.h"
#include "pallas/utils/pallas_storage.h"
#include "pallas/linked_vector/pallas_linked_vector.h"

short STORE_TIMESTAMPS = 1;
static short STORE_HASHING = 0;

void pallas_storage_option_init() {
    // Timestamp storage
    const char* store_timestamps_str = getenv("STORE_TIMESTAMPS");
    if (store_timestamps_str && strcmp(store_timestamps_str, "TRUE") != 0)
        STORE_TIMESTAMPS = 0;

    // Store hash for sequences
    const char* store_hashing_str = getenv("STORE_HASHING");
    if (store_hashing_str && strcmp(store_hashing_str, "FALSE") != 0)
        STORE_HASHING = 1;
}

static int pallasRecursiveMkdir(const char* dir, mode_t mode) {
    char tmp[1024];
    char* p = nullptr;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", dir);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for (p = tmp + 1; *p; p++)
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, mode);
            *p = '/';
        }
    return mkdir(tmp, mode);
}

void pallas::File::pallasMkdir(const char* dirname, mode_t mode) {
    if (pallasRecursiveMkdir(dirname, mode) != 0) {
        if (errno != EEXIST)
            pallas_error("mkdir(%s) failed: %s\n", dirname, strerror(errno));
    }
}



class FileMap : public std::map<const char*, pallas::File*> {
   public:
    ~FileMap() {
        for (auto& it : *this) {
            delete it.second;
        }
    }
};

FileMap fileMap;

pallas::File* pallas::getFirstOpenFile() {
    for (auto& a : fileMap) {
        if (a.second->isOpen) {
            return a.second;
        }
    }
    return nullptr;
}

static void storeEventData(pallas::EventData& event, const File& eventFile, const pallas::ParameterHandler& parameter_handler);
static void storeEvent(pallas::Event& event, const File& eventFile, const File& durationFile, const pallas::ParameterHandler* parameter_handler, bool load_thread);
static void storeSequence(pallas::Sequence& sequence, const File& sequenceFile, const File& durationFile, const pallas::ParameterHandler* parameter_handler, bool load_thread);

static void storeLoop(pallas::Loop& loop, const File& loopFile);

static void storeString(pallas::Definition& definitions, File& file);
static void storeRegions(pallas::Definition& definitions, File& file);
static void storeAttributes(pallas::Definition& definitions, File& file);
static void storeGroups(pallas::Definition& definitions, File& file);
static void storeComms(pallas::Definition& definitions, File& file);
static void storeMetadata(pallas::Metadata& metadata, File& file);

static void storeLocationGroups(std::vector<pallas::LocationGroup>& location_groups, File& file);
static void storeLocations(std::vector<pallas::Location>& locations, File& file);

static void readEventData(pallas::EventData& event, const File& eventFile, const pallas::ParameterHandler& parameter_handler, uint8_t abi_version);
static void readEvent(pallas::Event& event,
                      const File& eventFile,
                      const File& durationFile,
                      pallas::ParameterHandler& parameter_handler,
                      uint8_t abi_version);

static void readLoop(pallas::Loop& loop, const File& loopFile, uint8_t abi_version);
static void readSequence(pallas::Sequence& sequence, const File& sequenceFile, const char* durationFileName, pallas::ParameterHandler& parameter_handler, uint8_t abi_version);

static void readString(pallas::Definition& definitions, File& file, uint8_t abi_version);
static void readRegions(pallas::Definition& definitions, File& file, uint8_t abi_version);
static void readAttributes(pallas::Definition& definitions, File& file, uint8_t abi_version);
static void readGroups(pallas::Definition& definitions, File& file, uint8_t abi_version);
static void readComms(pallas::Definition& definitions, File& file, uint8_t abi_version);
static void readLocationGroups(std::vector<pallas::LocationGroup>& location_groups, File& file, uint8_t abi_version);
static void readLocations(std::vector<pallas::Location>& locations, File& file, uint8_t abi_version);
static void readMetadata(pallas::Metadata& metadata, File& file, uint8_t abi_version);
void pallasLoadThread(pallas::Archive* globalArchive, pallas::ThreadId thread_id);

static pallas::Archive* pallasGetArchive(pallas::GlobalArchive* global_archive, pallas::LocationGroupId archive_id, bool print_warning = true);

/******************* Read/Write/Compression function for vectors and arrays *******************/

/** Compresses the content in src using ZSTD and writes it to dest. Returns the amount of data written.
 *  @param src The source array.
 *  @param size Size of the source array.
 *  @param dest A free array in which the compressed data will be written.
 *  @param destSize Size of the destination array
 *  @returns Number of bytes written in the dest array.
 */
inline static size_t _pallas_zstd_compress(void* src, size_t size, void* dest, size_t destSize, int compression_level) {
    return ZSTD_compress(dest, destSize, src, size, compression_level);
}

/**
 * Decompresses an array that has been compressed by ZSTD. Returns the size of the uncompressed data.
 * @param realSize Size of the uncompressed data.
 * @param compArray The compressed array.
 * @param compSize Size of the compressed array.
 * @returns The uncompressed array.
 */
inline static uint64_t* _pallas_zstd_read(size_t& realSize, void* compArray, size_t compSize) {
    realSize = ZSTD_getFrameContentSize(compArray, compSize);
    auto dest = new byte[realSize];
    ZSTD_decompress(dest, realSize, compArray, compSize);
    return reinterpret_cast<uint64_t*>(dest);
}

#ifdef WITH_ZFP
/**
 * Gives a conservative upper bound for the size of the compressed data.
 * @param src The source array.
 * @param n Number of items in the array.
 * @return Upper bound to compressed array size in bytes.
 */
inline static size_t _pallas_zfp_bound(uint64_t* src, size_t n) {
    zfp_type type = zfp_type_int64;                 // array scalar type
    zfp_field* field = zfp_field_1d(src, type, n);  // array metadata
    zfp_stream* zfp = zfp_stream_open(nullptr);     // compressed stream and parameters
    zfp_stream_set_accuracy(zfp, .1);               // set tolerance for fixed-accuracy mode, this is absolute error
    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    zfp_stream_close(zfp);
    return bufsize;
}
/**
 * Compresses the content in src using the 1D ZFP Algorithm and writes it to dest.
 * Returns the amounts of data written.
 * @param src The source array.
 * @param n Number of items in the source array.
 * @param dest A free array in which the compressed data will be written.
 * @param destSize Size of the destination array.
 * @return Number of bytes written in the dest array.
 */
inline static size_t _pallas_zfp_compress(uint64_t* src, size_t n, void* dest, size_t destSize) {
    zfp_type type = zfp_type_int64;                        // array scalar type
    zfp_field* field = zfp_field_1d(src, type, n);         // array metadata
    zfp_stream* zfp = zfp_stream_open(nullptr);            // compressed stream and parameters
    zfp_stream_set_accuracy(zfp, .1);                      // set tolerance for fixed-accuracy mode, this is absolute error
    size_t bufsize = zfp_stream_maximum_size(zfp, field);  // capacity of compressed buffer (conservative)
    pallas_assert(bufsize <= destSize);
    bitstream* stream = stream_open(dest, bufsize);  // bit stream to compress to
    zfp_stream_set_bit_stream(zfp, stream);          // associate with compressed stream
    zfp_stream_rewind(zfp);                          // rewind stream to beginning
    size_t outSize = zfp_compress(zfp, field);       // return value is byte size of compressed stream
    zfp_stream_close(zfp);
    stream_close(stream);
    return outSize;
}

/**
 * Decompresses the content in src using the 1D ZFP Algorithm and writes it to dest.
 * Returns the amounts of data written.
 * @param n Number of items that should be decompressed.
 * @param compressedArray The compressed array.
 * @param compressedSize Size of the compressed array.
 * @returns Uncompressed array of size uint64 * n.
 */
inline static uint64_t* _pallas_zfp_decompress(size_t n, void* compressedArray, size_t compressedSize) {
    auto dest = new uint64_t[n];
    zfp_type type = zfp_type_int64;                                    // array scalar type
    zfp_field* field = zfp_field_1d(dest, type, n);                    // array metadata
    zfp_stream* zfp = zfp_stream_open(nullptr);                        // compressed stream and parameters
    zfp_stream_set_accuracy(zfp, .1);                                  // set tolerance for fixed-accuracy mode, this is absolute error
    bitstream* stream = stream_open(compressedArray, compressedSize);  // bit stream to read from
    zfp_stream_set_bit_stream(zfp, stream);                            // associate with compressed stream
    zfp_stream_rewind(zfp);                                            // rewind stream to beginning
    size_t outSize = zfp_decompress(zfp, field);                       // return value is byte size of compressed stream
    zfp_stream_close(zfp);
    stream_close(stream);
    return dest;
}
#endif
#ifdef WITH_SZ
/**
 * Compresses the content in src using the 1D SZ Algorithm.
 * @param src The source array.
 * @param n Number of items in the source array.
 * @param compressedSize Size of the compressed array. Passed by ref and modified.
 * @return The compressed array.
 */
inline static byte* _pallas_sz_compress(uint64_t* src, size_t n, size_t& compressedSize) {
    SZ_Init(nullptr);
    byte* compressedArray = reinterpret_cast<byte*>(SZ_compress(SZ_UINT64, src, &compressedSize, 0, 0, 0, 0, n));
    SZ_Finalize();
    return compressedArray;
}

inline static uint64_t* _pallas_sz_decompress(size_t n, byte* compressedArray, size_t compressedSize) {
    return static_cast<uint64_t*>(SZ_decompress(SZ_UINT64, reinterpret_cast<unsigned char*>(compressedArray), compressedSize, 0, 0, 0, 0, n));
};

#endif

#define N_BYTES 1
#define N_BITS (N_BYTES * 8)
#define MAX_BIT ((1 << N_BITS) - 1)

#ifdef DEBUG
inline static void collectHistogramStats(size_t min, size_t max, const uint64_t* array, size_t n) {
    size_t histogram[1 << N_BITS] = {0};
    size_t width = max - min;
    size_t stepSize = (width) / MAX_BIT;
    if (stepSize < 1) {
        std::cout << "Not interesting to print..." << std::endl;
        return;
    }
    for (size_t i = 0; i < n; i++) {
        size_t binNumber = (array[i] - min) / stepSize;
        binNumber = (binNumber > MAX_BIT) ? MAX_BIT : binNumber;
        histogram[binNumber] += 1;
    }
    size_t maxBinSize = 0;
    size_t groupBy = 4;
    auto string = std::stringstream("");
    for (size_t i = 0; i <= MAX_BIT; i += groupBy) {
        auto newValue = 0;
        DOFOR(j, groupBy) {
            newValue += histogram[i + j];
        }
        if (newValue > 0 && newValue < groupBy * 4) {
            string << ".";
        }
        newValue /= groupBy * 4;
        string << std::string(newValue, '#') << "\n";
    }
    std::cout << string.str() << std::endl;
}

#endif

/** Compresses the content in src using the Histogram method and writes it to dest.
 * Returns the amount of data written.
 *  @param src The source array.
 *  @param n Number of elements in src.
 *  @param dest A free array in which the compressed data will be written.
 *  @param destSize Size of the destination array
 *  @returns Number of bytes written in the dest array.
 */
inline static size_t _pallas_histogram_compress(const uint64_t* src, size_t n, byte* dest, size_t destSize) {
    // This method works by filling "bins" between the min and the max of the array.
    // First check that the destination size of enough to write everything in case you can't compress enough
    pallas_assert(destSize >= (N_BYTES * n + 2 * sizeof(uint64_t)));
    // Compute the min and max
    uint64_t min = UINT64_MAX, max = 0;
    for (size_t i = 0; i < n; i++) {
        min = (src[i] < min) ? src[i] : min;
        max = (src[i] > max) ? src[i] : max;
    }
#ifdef DEBUG
    // collectHistogramStats(min, max, src, n);
#endif
  size_t width = max - min;
  // TODO Skip the previous step using the stats from the vector.
  if (width <= MAX_BIT) {
    for (size_t i = 0; i < n; i++) {
      size_t toWrite = src[i] - min;
      // This MUST be <= MAX_BIT
      if (toWrite > MAX_BIT) {
        pallas_warn("Trying to write %lu values using %d byte at most\n", n, N_BYTES);
        pallas_warn("%" PRIu64 " <= value <= %" PRIu64 ". Problematic value is @%lu:%lu > %d", min, max, i, toWrite, MAX_BIT);
        pallas_error();
      }
      memcpy(&dest[i * N_BYTES], &toWrite, N_BYTES);
    }
  } else {
    double stepSize = double(width) / MAX_BIT;

        // Write min/max
        memcpy(dest, &min, sizeof(min));
        dest = &dest[sizeof(min)];  // Offset the address
        memcpy(dest, &max, sizeof(max));
        dest = &dest[sizeof(max)];  // Offset the address

        // Write each bin
        for (size_t i = 0; i < n; i++) {
            size_t binNumber = std::floor((src[i] - min)) / stepSize;
            binNumber = (binNumber > MAX_BIT) ? MAX_BIT : binNumber;
            // This last check is here in the rare cases of overflow.
            // printf("Writing %lu as %lu\n", src[i], temp);
            memcpy(&dest[i * N_BYTES], &binNumber, N_BYTES);
            // TODO This will not work on small endians architectures.
        }
    }
    return N_BYTES * n + 2 * sizeof(uint64_t);
}

/** Decompresses the content in compArray using the Histogram method and writes it to dest.
 * Returns the amount of data written.
 * @param n Number of elements in the dest array.
 * @param compArray The compressed array.
 * @param compSize Size of the compressed array.
 * @returns Array of uncompressed data of size uint64_t * n.
 */
inline static uint64_t* _pallas_histogram_read(size_t n, byte* compArray, size_t compSize) {
    auto dest = new uint64_t[n];
    // Compute the min and max
    uint64_t min, max;
    memcpy(&min, compArray, sizeof(min));
    compArray = &compArray[sizeof(min)];
    memcpy(&max, compArray, sizeof(max));
    compArray = &compArray[sizeof(max)];
    size_t width = max - min;

    // TODO Skip the previous step using the stats from the vector.
    if (width <= MAX_BIT) {
        for (size_t i = 0; i < n; i++) {
            size_t factor = 0;
            memcpy(&factor, &compArray[i * N_BYTES], N_BYTES);
            dest[i] = min + factor;
            //    printf("Reading %lu as %lu\n", factor, dest[i]);
        }
    } else {
        double stepSize = double(width) / MAX_BIT;

        for (size_t i = 0; i < n; i++) {
            size_t factor = 0;
            memcpy(&factor, &compArray[i * N_BYTES], N_BYTES);
            dest[i] = min + std::floor(factor * stepSize);
            //    printf("Reading %lu as %lu\n", factor, dest[i]);
        }
    }
    return dest;
}

/**
 * Encodes the content in src using a Masking technique and writes it to dest.
 * This is done only for 64-bits values.
 * @param src The source array. Contains n elements of 8 bytes (sizeof uint64_t).
 * @param dest The destination array. Same size as src, is an uint8 for convenience (byte counting).
 * @param n Number of elements in source array.
 * @return Number of interesting bytes contained in dest. (0 <= nBytes <= n * sizeof uint64)
 */
inline static size_t _pallas_masking_encode(const uint64_t* src, byte* dest, size_t n) {
    uint64_t mask = 0;
    for (int i = 0; i < n; i++) {
        mask |= src[i];
    }
    short maskSize = 0;
    while (mask != 0) {
        mask >>= 8;
        maskSize += 1;
    }
    // maskSize is the number of bytes needed to write the mask
    // ie the most amount of byte any number in src will need to be written
    if (maskSize && maskSize != sizeof(uint64_t)) {
        for (int i = 0; i < n; i++) {
            // FIXME This works because our LSB is in front (Small-endian)
            memcpy(&dest[maskSize * i], &src[i], maskSize);
        }
        return maskSize * n;
    } else {
        memcpy(dest, src, n * sizeof(uint64_t));
        return n * sizeof(uint64_t);
    }
}

/** De-encodes an array that has been compressed by the Masking technique. Returns the size of the unencoded data.
 * @param n Number of elements in the dest array.
 * @param encodedArray The encoded array.
 * @param encodedSize Size of the encoded array.
 * @returns Decoded array.
 */
inline static uint64_t* _pallas_masking_read(size_t n, byte* encodedArray, size_t encodedSize) {
    auto dest = new uint64_t[n];
    size_t size = n * sizeof(uint64_t);
    if (encodedSize == size) {
        memcpy(dest, encodedArray, size);
        return dest;
    }
    size_t width = encodedSize / n;
    // width is the number of bytes needed to write an element in the encoded array.
    memset(dest, 0, size);
    for (int i = 0; i < n; i++) {
        // FIXME Still only works with Little-Endian architecture.
        memcpy(&dest[i], &encodedArray[width * i], width);
    }
    return dest;
}

size_t numberPreRawBytes = 0;
size_t numberRawBytes = 0;
size_t numberCompressedBytes = 0;

/**
 * Writes the array to the given file, but encodes and compresses it before
 * according to the value of parameterHandler::EncodingAlgorithm and parameterHandler::CompressingAlgorithm.
 * @param src The source array. Contains n elements of 8 bytes (sizeof uint64_t).
 * @param n Number of elements in src.
 * @param file File to write in.
 * @param parameter_handler Handler for the storage options.
 */
void _pallas_compress_write(uint64_t* src, size_t n, pallas::File* file, const pallas::ParameterHandler* parameter_handler) {
    file->begin_block(__func__);
    size_t size = n * sizeof(uint64_t);
    uint64_t* encodedArray = nullptr;
    size_t encodedSize;
    // First we do the encoding
    switch (parameter_handler->getEncodingAlgorithm()) {
    case pallas::EncodingAlgorithm::None:
        break;
    case pallas::EncodingAlgorithm::Masking: {
        encodedArray = new uint64_t[n];
        encodedSize = _pallas_masking_encode(src, reinterpret_cast<byte*>(encodedArray), n);
        break;
    }
    case pallas::EncodingAlgorithm::LeadingZeroes: {
        pallas_error("Not yet implemented\n");
        break;
    }
    default:
        pallas_error("Invalid Encoding algorithm\n");
    }

    byte* compressedArray = nullptr;
    size_t compressedSize;
    switch (parameter_handler->getCompressionAlgorithm()) {
    case pallas::CompressionAlgorithm::None:
        break;
    case pallas::CompressionAlgorithm::ZSTD: {
        compressedSize = ZSTD_compressBound(encodedArray ? encodedSize : size);
        compressedArray = new byte[compressedSize];
        if (encodedArray) {
            compressedSize = _pallas_zstd_compress(encodedArray, encodedSize, compressedArray, compressedSize, parameter_handler->getZstdCompressionLevel());
        } else {
            compressedSize = _pallas_zstd_compress(src, size, compressedArray, compressedSize, parameter_handler->getZstdCompressionLevel());
        }
        break;
    }
    case pallas::CompressionAlgorithm::Histogram: {
        compressedSize = N_BYTES * n + 2 * sizeof(uint64_t);
        // Take into account that we add the min and the max.
        compressedArray = new byte[compressedSize];
        compressedSize = _pallas_histogram_compress(src, n, compressedArray, compressedSize);
        break;
    }
    case pallas::CompressionAlgorithm::ZSTD_Histogram: {
        // We first do the Histogram compress
        auto tempCompressedSize = N_BYTES * n + 2 * sizeof(uint64_t);
        auto tempCompressedArray = new byte[tempCompressedSize];
        tempCompressedSize = _pallas_histogram_compress(src, n, tempCompressedArray, tempCompressedSize);

        // And then the ZSTD compress
        compressedSize = ZSTD_compressBound(tempCompressedSize);
        compressedArray = new byte[compressedSize];
        compressedSize = _pallas_zstd_compress(tempCompressedArray, tempCompressedSize, compressedArray, compressedSize, parameter_handler->getZstdCompressionLevel());
        delete[] tempCompressedArray;
        break;
    }
#ifdef WITH_ZFP
    case pallas::CompressionAlgorithm::ZFP:
        compressedSize = _pallas_zfp_bound(src, n);
        compressedArray = new byte[compressedSize];
        compressedSize = _pallas_zfp_compress(src, n, compressedArray, compressedSize);
        break;
#endif
#ifdef WITH_SZ
    case pallas::CompressionAlgorithm::SZ:
        compressedArray = _pallas_sz_compress(src, n, compressedSize);
        break;
#endif
    default:
        pallas_error("Invalid Compression algorithm\n");
    }

    if (parameter_handler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None) {
        pallas_log(pallas::DebugLevel::Debug, "Compressing %lu bytes as %lu bytes\n", size, compressedSize);
        file->write(&compressedSize, sizeof(compressedSize), 1);
        file->write(compressedArray, compressedSize, 1);
        numberCompressedBytes += sizeof(compressedSize) + compressedSize;
    } else if (parameter_handler->getEncodingAlgorithm() != pallas::EncodingAlgorithm::None) {
        pallas_log(pallas::DebugLevel::Debug, "Encoding %lu bytes as %lu bytes\n", size, encodedSize);
        file->write(&encodedSize, sizeof(encodedSize), 1);
        file->write(encodedArray, encodedSize, 1);
        numberCompressedBytes += sizeof(encodedSize) + encodedSize;
    } else {
        pallas_log(pallas::DebugLevel::Debug, "Writing %lu bytes as is in %p.\n", size, file->file);
        file->write(&size, sizeof(size), 1);
        file->write(src, size, 1);
        numberCompressedBytes += sizeof(size) + size;
    }
    if (parameter_handler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None)
        delete[] compressedArray;
    if (parameter_handler->getEncodingAlgorithm() != pallas::EncodingAlgorithm::None)
        delete[] encodedArray;
    file->end_block(__func__);
}

/**
 * Reads, de-encodes and decompresses an array from the given file,
 * according to the values of parameterHandler::EncodingAlgorithm and parameterHandler::CompressingAlgorithm.
 * @param n Number of elements of 8 bytes dest is supposed to have.
 * @param file File to read from
 * @returns Array of uncompressed data of size uint64_t * n.
 */
uint64_t* _pallas_compress_read(size_t n, pallas::File* file, const pallas::ParameterHandler& parameter_handler) {
    file->begin_block(__func__);
    size_t expectedSize = n * sizeof(uint64_t);
    uint64_t* uncompressedArray = nullptr;

    size_t compressedSize;
    byte* compressedArray = nullptr;

    size_t encodedSize;
    byte* encodedArray = nullptr;

    auto compressionAlgorithm = parameter_handler.getCompressionAlgorithm();
    auto encodingAlgorithm = parameter_handler.getEncodingAlgorithm();
    if (compressionAlgorithm != pallas::CompressionAlgorithm::None) {
        file->read(&compressedSize, sizeof(compressedSize), 1);
        compressedArray = new byte[compressedSize];
        file->read(compressedArray, compressedSize, 1);
    }

    switch (compressionAlgorithm) {
    case pallas::CompressionAlgorithm::None:
        break;
    case pallas::CompressionAlgorithm::ZSTD: {
        if (encodingAlgorithm == pallas::EncodingAlgorithm::None) {
            size_t uncompressedSize;
            uncompressedArray = _pallas_zstd_read(uncompressedSize, compressedArray, compressedSize);
            pallas_assert(uncompressedSize == expectedSize);
        } else {
            encodedArray = reinterpret_cast<byte*>(_pallas_zstd_read(encodedSize, compressedArray, compressedSize));
            pallas_assert(encodedSize <= expectedSize);
        }
        delete[] compressedArray;
        break;
    }
    case pallas::CompressionAlgorithm::Histogram: {
        uncompressedArray = _pallas_histogram_read(n, compressedArray, compressedSize);
        break;
    }
    case pallas::CompressionAlgorithm::ZSTD_Histogram: {
        // First ZSTD Decode
        size_t histogramSize;
        auto tempUncompressedArray = reinterpret_cast<byte*>(_pallas_zstd_read(histogramSize, compressedArray, compressedSize));
        uncompressedArray = _pallas_histogram_read(n, tempUncompressedArray, histogramSize);
        pallas_assert(n * sizeof(uint64_t) == expectedSize);
        break;
    }
#ifdef WITH_ZFP
    case pallas::CompressionAlgorithm::ZFP: {
        uncompressedArray = _pallas_zfp_decompress(n, compressedArray, compressedSize);
        break;
    }
#endif
#ifdef WITH_SZ
    case pallas::CompressionAlgorithm::SZ:
        uncompressedArray = _pallas_sz_decompress(n, compressedArray, compressedSize);
        break;
#endif
    default:
        pallas_error("Invalid Compression algorithm\n");
    }

    switch (encodingAlgorithm) {
    case pallas::EncodingAlgorithm::None:
        break;
    case pallas::EncodingAlgorithm::Masking: {
        if (compressionAlgorithm == pallas::CompressionAlgorithm::None) {
            file->read(&encodedSize, sizeof(encodedSize), 1);
            encodedArray = new byte[encodedSize];  // Too big but don't care
            file->read(encodedArray, encodedSize, 1);
        }
        uncompressedArray = _pallas_masking_read(n, encodedArray, encodedSize);
        delete[] encodedArray;
        break;
    }
    case pallas::EncodingAlgorithm::LeadingZeroes: {
        pallas_error("Not yet implemented\n");
        break;
    }
    default:
        pallas_error("Invalid Encoding algorithm\n");
    }

    if (compressionAlgorithm == pallas::CompressionAlgorithm::None && encodingAlgorithm == pallas::EncodingAlgorithm::None) {
        size_t realSize;
        file->read(&realSize, sizeof(realSize), 1);
        uncompressedArray = new uint64_t[n];
        file->read(uncompressedArray, realSize, 1);
        pallas_assert(realSize == n * sizeof(uint64_t));
    }
    file->end_block(__func__);
    return uncompressedArray;
}


/** Linked-vector storage helpers shared across TimeLinkedVector and DurationLinkedVector. */

#if 0
/**
 * @brief Write the common SubArray header shared by all storage policies.
 *
 * The header records the logical size, packed policy byte, physical payload
 * size, and persisted payload offset needed to rebuild the SubArray later.
 */
void pallas::SubArrayBase::write_common_header(FILE* info_file) const {
    if (info_file == nullptr) {
        return;
    }

    const auto value_count = this->size();
    uint8_t stored_policy = 0;
    const auto physical_size = mem_size();
    const auto file_offest = offset();
    // Persist the subarray scheme in one byte:
    //   - lower 2 bits: StoragePolicy
    //   - upper 6 bits: LossyPolicy variant when StoragePolicy::Lossy is used
    stored_policy = pack_subarray_flags();

    _pallas_fwrite(&value_count, sizeof(value_count), 1, info_file);
    _pallas_fwrite(&stored_policy, sizeof(stored_policy), 1, info_file);
    _pallas_fwrite(&physical_size, sizeof(physical_size), 1, info_file);
    _pallas_fwrite(&file_offest, sizeof(file_offest), 1, info_file);
}

/**
 * @brief Read the common SubArray header and rebuild the matching manager.
 */
void pallas::SubArrayBase::read_common_header(FILE* info_file) {
    uint8_t stored_policy = 0;
    size_t physical_size = 0;

    _pallas_fread(&value_count, sizeof(value_count), 1, info_file);
    _pallas_fread(&stored_policy, sizeof(stored_policy), 1, info_file);
    _pallas_fread(&physical_size, sizeof(physical_size), 1, info_file);
    _pallas_fread(&file_offset, sizeof(file_offset), 1, info_file);

    unpack_subarray_flags(stored_policy);

    rebuild_manager();
    this->physical_size = physical_size;
}
#endif

/**
 * @brief Reconstruct the common linked-vector header from the info stream.
 *
 * This restores the logical size and, for newer ABI versions, the persisted
 * subarray count and preferred storage policy.
 */
pallas::LinkedVectorBase::LinkedVectorBase(pallas::File* summary_file, pallas::File* details_file, ParameterHandler& p,
                       ValueDomain domain, StoragePolicy _policy, uint8_t abi_version)
    : parameter_handler(p),
      value_domain(domain),
      storage_policy(_policy),
      _summary_file(summary_file),
      _details_file(details_file) {

    summary_file->begin_block(__func__);
    summary_file->read(&value_count, sizeof(value_count), 1);

    if (abi_version >= 18) {
        uint8_t stored_policy = 0;
        summary_file->read(&subarray_total, sizeof(subarray_total), 1);
        summary_file->read(&stored_policy, sizeof(stored_policy), 1);
        if (stored_policy <= static_cast<uint8_t>(StoragePolicy::Lossy)) {
            storage_policy = static_cast<StoragePolicy>(stored_policy);
        }
    }
    summary_file->end_block(__func__);
}

/**
 * @brief Write the common linked-vector header shared by `TimeLinkedVector` and `DurationLinkedVector`.
 */
void pallas::LinkedVectorBase::write_common_header(pallas::File* vector_file) const {
    //TODO:  we do not need the summary/details_file parameters here. We should use the fields instead
    if (vector_file == nullptr) {
        return;
    }
    vector_file->begin_block(__func__);
    const auto policy = storage_policy;
    vector_file->write(&value_count, sizeof(value_count), 1);
    vector_file->write(&subarray_total, sizeof(subarray_total), 1);
    vector_file->write(&policy, sizeof(policy), 1);
    vector_file->end_block(__func__);
}

/**
 * @brief Lazily load one SubArray payload from the value stream.
 *
 * This is the read-side hook used by `LinkedVectorBase` when a SubArray is present in the
 * chain but its payload has been evicted from memory.
 */
void pallas::LinkedVectorBase::load_data(SubArrayBase* sub) {
    pallas_log(DebugLevel::Debug, "Loading values from %s @%lu\n", _details_file->path, sub->details_offset());
    if (!_details_file->isOpen) {
        _details_file->open("r");
    }
    int ret = fseek(_details_file->file, sub->details_offset(), 0);
    while (ret == EBADF) {
        _details_file->close();
        _details_file->open("r");
        ret = fseek(_details_file->file, sub->details_offset(), 0);
    }

    sub->read_data(_details_file);

    parameter_handler.loaded_durations_size += sub->mem_size() * sizeof(uint64_t);
    parameter_handler.subvector_queue.emplace_back(sub);
}

/** Time linked-vector storage methods. */

#if 0
/**
 * @brief Write the timestamp-specific SubArray header fields.
 */
void pallas::SubArrayBase::write_header(FILE* info_file) const { // -> renamed write_summary
    if (info_file == nullptr) {
        return;
    }

    const auto first = first_value();
    const auto last = last_value();

    write_common_header(info_file);
    _pallas_fwrite(&first, sizeof(first), 1, info_file);
    _pallas_fwrite(&last, sizeof(last), 1, info_file);
}

#endif

#if 0
/**
 * @brief Read the timestamp-specific SubArray header fields.
 */
void pallas::TimeSubArray::read_header(FILE* info_file) {
    _pallas_fread(&first_timestamp, sizeof(first_timestamp), 1, info_file);
    _pallas_fread(&last_timestamp, sizeof(last_timestamp), 1, info_file);
}

/**
 * @brief Reconstruct one timestamp SubArray from the info stream.
 */
pallas::TimeSubArray::TimeSubArray(FILE* info_file, TimeSubArray* previous)
    : SubArrayBase(info_file, ValueDomain::Timestamp, previous) {
    read_header(info_file);
}

#endif

#if 0
/**
 * @brief Write the `TimeLinkedVector` header that precedes all timestamp subarray headers.
 */
void pallas::TimeLinkedVector::write_header(pallas::File* summary_file) {
    //TODO:  we do not need the summary/details_file parameters here. We should use the fields instead
    write_common_header(summary_file);
}


/**
 * @brief Persist the full timestamp linked vector across the info and value streams.
 */
void pallas::TimeLinkedVector::write_to_file(pallas::File* summary_file, pallas::File* details_file, const ParameterHandler* parameter_handler) {
//TODO:  we do not need the summary/details_file parameters here. We should use the fields instead

#ifdef BMARK
    BmarkScopedTimer timer(get_bmark_family(), BmarkMetric::Write);
    bmark_note_write_call(get_bmark_family(), size());
#endif
    write_header(summary_file);

    for (auto* base_subarray = first; base_subarray != nullptr; base_subarray = base_subarray->next_subarray()) {
        auto* subarray = base_subarray;
        subarray->write_data(details_file);
        subarray->write_summary(summary_file);
    }
}
#endif
/**
 * @brief Reconstruct a `TimeLinkedVector` from the persisted info stream.
 *
 * For newer ABI versions the stored subarray count is trusted directly;
 * otherwise the constructor walks subarray headers until the logical size is
 * fully covered.
 */
pallas::TimeLinkedVector::TimeLinkedVector(pallas::File* summary_file, pallas::File* details_file, ParameterHandler& p, uint8_t abi_version)
    : LinkedVectorBase(summary_file, details_file, p, ValueDomain::Timestamp, p.getStoragePolicy(), abi_version) {
    if (value_count == 0) {
        return;
    }

    if (abi_version >= 18) {
        for (size_t i = 0; i < subarray_total; ++i) {
            last = SubArrayBase::load_subarray(details_file, last);
            if (first == nullptr) {
                first = last;
            }
        }
        rebuild_subarray_index();
        return;
    }

    size_t loaded_values = 0;
    subarray_total = 0;
    while (loaded_values < value_count) {
        last = SubArrayBase::load_subarray(details_file, last);
        if (first == nullptr) {
            first = last;
        }
        loaded_values += last->size();
        subarray_total++;
    }
    rebuild_subarray_index();
}

/** Duration linked-vector storage methods. */

#if 0
/**
 * @brief Write the duration-specific SubArray header fields.
 */
void pallas::DurationSubArray::write_header(FILE* info_file) const {
    if (info_file == nullptr) {
        return;
    }

    const auto min = min_value();
    const auto max = max_value();
    const auto mean = mean_value();

    write_common_header(info_file);
    _pallas_fwrite(&min, sizeof(min), 1, info_file);
    _pallas_fwrite(&max, sizeof(max), 1, info_file);
    _pallas_fwrite(&mean, sizeof(mean), 1, info_file);
    pallas_assert_inferior_equal(mean, max);
    pallas_assert_inferior_equal(min, mean);
}

/**
 * @brief Read the duration-specific SubArray header fields.
 */
void pallas::DurationSubArray::read_header(FILE* info_file) {
    _pallas_fread(&min_duration, sizeof(min_duration), 1, info_file);
    _pallas_fread(&max_duration, sizeof(max_duration), 1, info_file);
    _pallas_fread(&mean_duration, sizeof(mean_duration), 1, info_file);
    mean_duration_is_finalized = true;
    if (max_duration < mean_duration) {
        static bool show_warning = true;
        if (show_warning) {
            pallas_warn("This trace is malformed ( see 36daaa9ed0fd0517bbc42e6f78ca7627cea30b82 ). You should update Pallas and regenerate it.\n");
            show_warning = false;
        }
        mean_duration /= value_count;
    }
    pallas_assert_inferior_equal(mean_duration, max_duration);
    pallas_assert_inferior_equal(min_duration, mean_duration);
}
#endif

#if 0
/**
 * @brief Reconstruct one duration SubArray from the info stream.
 */
pallas::DurationSubArray::DurationSubArray(FILE* info_file, DurationSubArray* previous)
    : SubArrayBase(info_file, ValueDomain::Duration, previous) {
    read_header(info_file);
}

#endif

#if 0
/**
 * @brief Write the `DurationLinkedVector` header that precedes all duration subarray headers.
 */
void pallas::DurationLinkedVector::write_header(pallas::File* summary_file) {
    summary_file->begin_block(__func__);
    write_common_header(summary_file);
    if (value_count == 0) {
        return;
    }

    for (auto* base_subarray = first; base_subarray != nullptr; base_subarray = base_subarray->next_subarray()) {
        base_subarray->finalize_block();
//        static_cast<DurationSubArray*>(base_subarray)->final_update_mean();
    }
    final_update_mean();

    summary_file->write(&min_duration, sizeof(min_duration), 1);
    summary_file->write(&max_duration, sizeof(max_duration), 1);
    summary_file->write(&mean_duration, sizeof(mean_duration), 1);
    pallas_assert_inferior_equal(mean_duration, max_duration);
    pallas_assert_inferior_equal(min_duration, mean_duration);
    summary_file->end_block(__func__);
}
#endif
/**
 * @brief Persist the full duration linked vector across the info and value streams.
 */
void pallas::DurationLinkedVector::write_to_file(pallas::File* summary_file, pallas::File* details_file, const ParameterHandler* parameter_handler) {
    //TODO:  we do not need the summary/details_file parameters here. We should use the fields instead
#ifdef BMARK
    BmarkScopedTimer timer(get_bmark_family(), BmarkMetric::Write);
    bmark_note_write_call(get_bmark_family(), size());
#endif
    //write_header(summary_file);
    if (value_count == 0) {
        return;
    }

    for (auto* subarray = first; subarray != nullptr; subarray = subarray->next_subarray()) {
        subarray->write_details(details_file, nullptr, nullptr);
//        subarray->write_summary(summary_file);
    }
}

/**
 * @brief Reconstruct a `DurationLinkedVector` from the persisted info stream.
 *
 * This restores the vector-level duration statistics first, then rebuilds the
 * SubArray chain using either the stored subarray count or the legacy
 * size-driven loop depending on the ABI version.
 */
pallas::DurationLinkedVector::DurationLinkedVector(pallas::File* summary_file, pallas::File* details_file, ParameterHandler& p, uint8_t abi_version)
    : LinkedVectorBase(summary_file, details_file, p, ValueDomain::Duration, p.getStoragePolicy(), abi_version) {
    if (value_count == 0) {
        return;
    }
    size_t loaded_values = 0;
    summary_file->begin_block(__func__);
    summary_file->read(&min_duration, sizeof(min_duration), 1);
    summary_file->read(&max_duration, sizeof(max_duration), 1);
    summary_file->read(&mean_duration, sizeof(mean_duration), 1);
    mean_duration_is_finalized = true;
    if (max_duration < mean_duration) {
        static bool show_warning = true;
        if (show_warning) {
            pallas_warn("This trace is malformed ( see 36daaa9ed0fd0517bbc42e6f78ca7627cea30b82 ). You should update Pallas and regenerate it.\n");
            show_warning = false;
        }
        mean_duration /= value_count;
    }
    pallas_assert_inferior_equal(mean_duration, max_duration);
    pallas_assert_inferior_equal(min_duration, mean_duration);

    if (abi_version >= 18) {
        for (size_t i = 0; i < subarray_total; ++i) {
            last = SubArrayBase::load_subarray(details_file, last, &p, this);
            if (first == nullptr) {
                first = last;
            }
        }
        rebuild_subarray_index();
        goto out;
    }


    subarray_total = 0;
    while (loaded_values < value_count) {
        last = SubArrayBase::load_subarray(details_file, last, &p, this);
        if (first == nullptr) {
            first = last;
        }
        loaded_values += last->size();
        subarray_total++;
    }
    rebuild_subarray_index();
out:
    summary_file->end_block(__func__);
}

/**************** Storage Functions ****************/

void pallas_storage_init(const char* dir_name) {
    pallas::File::pallasMkdir(dir_name, 0777);
    pallas_storage_option_init();
}

static const char* base_dirname(pallas::Archive* a) {
    return a->dir_name;
}

static const char* getThreadPath(pallas::Thread* th) {
    char* folderPath = new char[1024];
    snprintf(folderPath, 1024, "archive_%u/thread_%u", th->archive->id, th->id);
    //  snprintf(folderPath, 1024, "thread_%u", th->id);
    return folderPath;
}

static const char* pallasGetEventDurationFilename(const char* base_dirname, pallas::Thread* th) {
    char* filename = new char[1024];
    const char* threadPath = getThreadPath(th);
    snprintf(filename, 1024, "%s/%s/event_durations.dat", base_dirname, threadPath);
    delete[] threadPath;
    return filename;
}

static void _pallas_store_attribute_values(pallas::Event* e, pallas::File& file, const pallas::ParameterHandler& parameter_handler) {
    file.begin_block(__func__);
    file.write(&e->attribute_pos, sizeof(e->attribute_pos), 1);
    if (e->attribute_pos > 0) {
        pallas_log(pallas::DebugLevel::Debug, "\t\tStore %lu attributes\n", e->attribute_pos);
        if (parameter_handler.getCompressionAlgorithm() != pallas::CompressionAlgorithm::None) {
            size_t compressedSize = ZSTD_compressBound(e->attribute_pos);
            byte* compressedArray = new byte[compressedSize];
            compressedSize = _pallas_zstd_compress(e->attribute_buffer, e->attribute_pos, compressedArray, compressedSize, parameter_handler.getZstdCompressionLevel());
            file.write(&compressedSize, sizeof(compressedSize), 1);
            file.write(compressedArray, compressedSize, 1);
            delete[] compressedArray;
        } else {
            file.write(e->attribute_buffer, e->attribute_pos, 1);
        }
    }
    file.end_block(__func__);
}

static void _pallas_read_attribute_values(pallas::Event* e, pallas::File& file, const pallas::ParameterHandler& parameter_handler, uint8_t abi_version) {
    file.begin_block(__func__);
    file.read(&e->attribute_pos, sizeof(e->attribute_pos), 1);
    e->attribute_buffer_size = e->attribute_pos;
    e->attribute_pos = 0;
    e->attribute_buffer = nullptr;

    if (e->attribute_buffer_size > 0) {
        e->attribute_buffer = new byte[e->attribute_buffer_size];
        if (e->attribute_buffer == nullptr) {
            pallas_error("Cannot allocate memory\n");
        }
        if (parameter_handler.getCompressionAlgorithm() != pallas::CompressionAlgorithm::None) {
            size_t compressedSize;
            file.read(&compressedSize, sizeof(compressedSize), 1);
            byte* compressedArray = new byte[compressedSize];
            file.read(e->attribute_buffer, compressedSize, 1);
            e->attribute_buffer = reinterpret_cast<byte*>(_pallas_zstd_read(e->attribute_buffer_size, compressedArray, compressedSize));
            delete[] compressedArray;
        } else {
            file.read(e->attribute_buffer, e->attribute_buffer_size, 1);
        }
    }
    file.end_block(__func__);
}
static void storeEventData(pallas::EventData& event,
                           pallas::File& eventFile,
                           const pallas::ParameterHandler& parameter_handler) {
    eventFile.begin_block(__func__);
#if 0
    eventFile.write(&event.record, sizeof(event.record), 1);
    eventFile.write(&event.event_size, sizeof(event.event_size), 1);
    size_t payload_size = event.event_size - offsetof(pallas::EventData, event_data);
    if (payload_size > 0) {
        eventFile.write(event.event_data, payload_size, 1);
    }
#endif
    eventFile.write(&event, sizeof(event), 1);
    eventFile.end_block(__func__);
}

static void readEventData(pallas::EventData& event,
                          pallas::File& eventFile,
                          const pallas::ParameterHandler& parameter_handler,
                          uint8_t abi_version) {
    eventFile.begin_block(__func__);
#if 0
    eventFile.read(&event.record, sizeof(event.record), 1);
    if (abi_version <= 18 && event.record == 0) {
        event.record = pallas::PALLAS_EVENT_BUFFER_FLUSH;
    }
    eventFile.read(&event.event_size, sizeof(event.event_size), 1);
    std::memset(event.event_data, 0, sizeof(event.event_data));
    auto size = event.event_size - offsetof(pallas::EventData, event_data);
    size_t payload_size = event.event_size - offsetof(pallas::EventData, event_data);
    if (payload_size > 0) {
        eventFile.read(event.event_data, payload_size, 1);
    }
#endif
    eventFile.read(&event, sizeof(event), 1);
    eventFile.end_block(__func__);
};


static void readEvent(pallas::Event& event,
                      pallas::File& summary_file,
                      pallas::File& details_file,
                      pallas::ParameterHandler& parameter_handler,
                      uint8_t abi_version) {
    summary_file.begin_block(__func__);
    readEventData(event.data, summary_file, parameter_handler, abi_version);

    size_t serialized_attr_size = 0;
    summary_file.read(&serialized_attr_size, sizeof(serialized_attr_size), 1);

    event.attribute_buffer = nullptr;
    event.attribute_buffer_size = serialized_attr_size;
    event.attribute_pos = serialized_attr_size;

    if (serialized_attr_size > 0) {
        event.attribute_buffer = new byte[serialized_attr_size];
        summary_file.read(event.attribute_buffer, sizeof(byte), serialized_attr_size);
    }

    if (event.data.record == pallas::PALLAS_EVENT_MAX_ID) {
        if (STORE_TIMESTAMPS) {
            size_t size = 0;
            summary_file.read(&size, sizeof(size), 1);
            if (abi_version >= 18) {
                size_t n_sub_array = 0;
                summary_file.read(&n_sub_array, sizeof(n_sub_array), 1);
            }
        }
        event.timestamps = nullptr;
        event.nb_occurrences = 0;
        pallas_log(pallas::DebugLevel::Debug, "\tLoaded invalid event %d\n", event.id);
        goto out;
    }
    event.timestamps = new pallas::TimeLinkedVector(&summary_file, &details_file, parameter_handler, abi_version);
    event.nb_occurrences = event.timestamps->size();
    pallas_log(pallas::DebugLevel::Debug, "\tLoaded event %d {.nb_events=%zu}\n", event.id, event.timestamps->size());
out:
    summary_file.end_block(__func__);
}

static const char* pallasGetSequenceDurationFilename(const char* base_dirname, pallas::Thread* th) {
    char* filename = new char[1024];
    const char* threadPath = getThreadPath(th);
    snprintf(filename, 1024, "%s/%s/sequence_durations.dat", base_dirname, threadPath);
    delete[] threadPath;
    return filename;
}


static void readSequence(pallas::Sequence& sequence, pallas::File& summary_file, pallas::File& details_file, pallas::ParameterHandler& parameter_handler, uint8_t abi_version) {
    summary_file.begin_block(__func__);
    summary_file.read(&sequence.type, sizeof(sequence.type), 1);
    size_t size;
    summary_file.read(&size, sizeof(size), 1);

    // catch empty sequence
    if (size == 0) {
        sequence.id = pallas::Token();
        sequence.tokens.clear();
        sequence.durations = nullptr;
        sequence.exclusive_durations = nullptr;
        sequence.timestamps = nullptr;
        pallas_log(pallas::DebugLevel::Debug, "\\tLoaded invalid sequence\\n");
        goto out;
    }

    sequence.tokens.resize(size);
    summary_file.read(sequence.tokens.data(), sizeof(pallas::Token), size);
    if (STORE_TIMESTAMPS) {
        sequence.durations = new pallas::DurationLinkedVector(&summary_file, &details_file, parameter_handler, abi_version);
        sequence.exclusive_durations = new pallas::DurationLinkedVector(&summary_file, &details_file, parameter_handler, abi_version);
        sequence.timestamps = new pallas::TimeLinkedVector(&summary_file, &details_file, parameter_handler, abi_version);
#ifdef BMARK
        sequence.durations->set_bmark_family(pallas::BmarkFamily::SequenceDurations);
        sequence.exclusive_durations->set_bmark_family(pallas::BmarkFamily::SequenceExclusiveDurations);
        sequence.timestamps->set_bmark_family(pallas::BmarkFamily::SequenceTimestamps);
#endif
    }
    pallas_log(pallas::DebugLevel::Debug, "\tLoaded sequence %d {.size=%zu, .nb_ts=%zu}\n", sequence.id.id, sequence.size(), sequence.durations->size());
out:
    summary_file.end_block(__func__);
}

#if 0
static void storeLoop(pallas::Loop& loop, pallas::File& loopFile) {
    if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
        pallas_log(pallas::DebugLevel::Debug, "\tStore loop %d {.repeated_token=%d.%d, .nb_iterations: %u\n", loop.self_id.id, loop.repeated_token.type, loop.repeated_token.id,
                   loop.nb_iterations);
        std::cout << "}" << std::endl;
    }
    loopFile.begin_block(__func__);
    loopFile.write(&loop.repeated_token, sizeof(loop.repeated_token), 1);
    loopFile.write(&loop.nb_iterations, sizeof(loop.nb_iterations), 1);
    loopFile.write(&loop.nb_occurrences, sizeof(loop.nb_occurrences), 1);
    loopFile.end_block(__func__);
}
#endif


static void readLoop(pallas::Loop& loop, pallas::File& loopFile, uint8_t abi_version) {
    loopFile.begin_block(__func__);
    loopFile.read(&loop.repeated_token, sizeof(loop.repeated_token), 1);
    loopFile.read(&loop.nb_iterations, sizeof(loop.nb_iterations), 1);
    if (abi_version <= 18) {
        loop.nb_occurrences = -1;
    } else {
        loopFile.read(&loop.nb_occurrences, sizeof(loop.nb_occurrences), 1);
    }
    pallas_log(pallas::DebugLevel::Debug, "\tLoad loop %d {.repeated_token=%d.%d, .nb_iterations: %u, .nb_occurrences: %" PRIu64 "}\n",
               loop.self_id.id, loop.repeated_token.type, loop.repeated_token.id, loop.nb_iterations, loop.nb_occurrences);

    if (loop.repeated_token.type == pallas::TypeInvalid) {
      loop.self_id.type = pallas::TypeInvalid;
    }
    loopFile.end_block(__func__);
}

static void storeString(pallas::Definition& definitions, pallas::File& file) {
    file.begin_block(__func__);
    size_t size = definitions.strings.size();
    file.write(&size, sizeof(size), 1);
    for (auto& it : definitions.strings) {
        auto& ref = it.first;
        auto& s = it.second;
        pallas_log(pallas::DebugLevel::Debug, "\tStore String {.ref=%d, .length=%d, .str='%s'}\n", s.string_ref, s.length, s.str);
        file.write(&s.string_ref, sizeof(s.string_ref), 1);
        file.write(&s.length, sizeof(s.length), 1);
        file.write(s.str, sizeof(char), s.length);
    }
    file.end_block(__func__);
}

static void readString(pallas::Definition& definitions, pallas::File& file, uint8_t abi_version) {
    file.begin_block(__func__);
    size_t size;
    file.read(&size, sizeof(size), 1);
    for (size_t i = 0; i < size; i++) {
        pallas::StringRef ref;
        file.read(&ref, sizeof(ref), 1);
        pallas::String& string = definitions.strings[ref];
        string.string_ref = ref;
        file.read(&string.length, sizeof(string.length), 1);
        string.str = (char*)calloc(string.length, sizeof(char));
        pallas_assert(string.str);
        file.read(string.str, sizeof(char), string.length);
        pallas_log(pallas::DebugLevel::Debug, "\tLoad String {.ref=%d, .length=%d, .str='%s'}\n", string.string_ref, string.length, string.str);
    }
    file.end_block(__func__);
}

static void storeRegions(pallas::Definition& definitions, pallas::File& file) {
    file.begin_block(__func__);
    size_t size = definitions.regions.size();
    file.write(&size, sizeof(size), 1);
    if (definitions.regions.empty())
        return;

    pallas_log(pallas::DebugLevel::Debug, "\tStore %zu Regions\n", definitions.regions.size());
    for (auto& region : definitions.regions) {
        file.write(&region.second, sizeof(pallas::Region), 1);
    }
    file.end_block(__func__);
}

static void readRegions(pallas::Definition& definitions, pallas::File& file, uint8_t abi_version) {
    file.begin_block(__func__);
    size_t size;
    file.read(&size, sizeof(size), 1);
    pallas::Region tempRegion;
    for (size_t i = 0; i < size; i++) {
        file.read(&tempRegion, sizeof(pallas::Region), 1);
        definitions.regions[tempRegion.region_ref] = tempRegion;
        pallas_log(pallas::DebugLevel::Debug, "\tLoad Region {.ref=%u}\n", tempRegion.region_ref);
    }

    pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu regions\n", definitions.regions.size());
    file.end_block(__func__);
}

static void storeAttributes(pallas::Definition& definitions, pallas::File& file) {
    file.begin_block(__func__);
    size_t size = definitions.attributes.size();
    file.write(&size, sizeof(size), 1);
    pallas_log(pallas::DebugLevel::Debug, "\tStore %zu Attributes\n", definitions.attributes.size());
    for (int i = 0; i < definitions.attributes.size(); i++) {
        pallas_log(pallas::DebugLevel::Debug, "\t\t[%d] {ref=%d, name=%d, type=%d}\n", i, definitions.attributes[i].attribute_ref, definitions.attributes[i].name,
                   definitions.attributes[i].type);
    }

    for (auto& attribute : definitions.attributes) {
        file.write(&attribute.second, sizeof(pallas::Attribute), 1);
    }
    file.end_block(__func__);
}

static void readAttributes(pallas::Definition& definitions, pallas::File& file, uint8_t abi_version) {
    file.begin_block(__func__);
    size_t size;
    file.read(&size, sizeof(size), 1);
    pallas::Attribute tempAttribute;
    for (size_t i = 0; i < size; i++) {
        file.read(&tempAttribute, sizeof(pallas::Attribute), 1);
        definitions.attributes[tempAttribute.attribute_ref] = tempAttribute;
    }

    pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu attributes\n", definitions.attributes.size());
    file.end_block(__func__);
}

static void storeGroups(pallas::Definition& definitions, pallas::File& file) {
    file.begin_block(__func__);
    size_t size = definitions.groups.size();
    file.write(&size, sizeof(size), 1);
    for (auto& [ref, g] : definitions.groups) {
        pallas_log(pallas::DebugLevel::Debug, "\tStore Group {.ref=%d, .name=%d, .nb_members=%d}\n", g.group_ref, g.name, g.numberOfMembers);

        file.write(&g.group_ref, sizeof(g.group_ref), 1);
        file.write(&g.name, sizeof(g.name), 1);
        file.write(&g.group_type, sizeof(g.group_type), 1);
        file.write(&g.paradigm, sizeof(g.paradigm), 1);
        file.write(&g.numberOfMembers, sizeof(g.numberOfMembers), 1);
        file.write(g.members, sizeof(uint32_t), g.numberOfMembers);
    }
    file.end_block(__func__);
}

static void readGroups(pallas::Definition& definitions, pallas::File& file, uint8_t abi_version) {
    file.begin_block(__func__);
    size_t size;
    file.read(&size, sizeof(size), 1);
    for (size_t i = 0; i < size; i++) {
        pallas::GroupRef ref;
        file.read(&ref, sizeof(ref), 1);
        pallas::Group& g = definitions.groups[ref];
        g.group_ref = ref;
        file.read(&g.name, sizeof(g.name), 1);
        if (abi_version >= 17) {
            file.read(&g.group_type, sizeof(g.group_type), 1);
            file.read(&g.paradigm, sizeof(g.paradigm), 1);
        }
        file.read(&g.numberOfMembers, sizeof(g.numberOfMembers), 1);
        g.members = new uint32_t[g.numberOfMembers];
        pallas_assert(g.members);
        if (abi_version == 16) {
            auto temp = new uint64_t[g.numberOfMembers];
            file.read(temp, sizeof(uint64_t), g.numberOfMembers);
            for (size_t i = 0; i < g.numberOfMembers; i++) {
                g.members[i] = temp[i];
            }
            delete[] temp;
        } else {
            file.read(g.members, sizeof(uint32_t), g.numberOfMembers);
        }
        pallas_log(pallas::DebugLevel::Debug, "\tLoad Group {.ref=%d, .name=%d, .nb_members=%d}\n", g.group_ref, g.name, g.numberOfMembers);
    }
    file.end_block(__func__);
}

static void storeComms(pallas::Definition& definitions, pallas::File& file) {
    file.begin_block(__func__);
    size_t size = definitions.comms.size();
    file.write(&size, sizeof(size), 1);
    if (definitions.comms.empty())
        goto out;

    pallas_log(pallas::DebugLevel::Debug, "\tStore %zu Comms\n", definitions.comms.size());
    for (auto& comm : definitions.comms) {
        file.write(&comm.second, sizeof(pallas::Comm), 1);
    }
out:
    file.end_block(__func__);
}

static void readComms(pallas::Definition& definitions, pallas::File& file, uint8_t abi_version) {
    file.begin_block(__func__);
    size_t size;
    file.read(&size, sizeof(size), 1);
    pallas::Comm tempComm;
    for (size_t i = 0; i < size; i++) {
        file.read(&tempComm, sizeof(pallas::Comm), 1);
        definitions.comms[tempComm.comm_ref] = tempComm;
    }

    pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu comms\n", definitions.comms.size());
    file.end_block(__func__);
}

static void storeDefinitions(pallas::Definition& def, pallas::File& file) {
    storeString(def, file);
    storeRegions(def, file);
    storeAttributes(def, file);
    storeGroups(def, file);
    storeComms(def, file);
}

static void readDefinitions(pallas::Definition& def, pallas::File& file, uint8_t abi_version) {
    readString(def, file, abi_version);
    readRegions(def, file, abi_version);
    readAttributes(def, file, abi_version);
    readGroups(def, file, abi_version);
    readComms(def, file, abi_version);
}

static void storeLocationGroups(std::vector<pallas::LocationGroup>& location_groups, pallas::File& file) {
    file.begin_block(__func__);
    pallas_log(pallas::DebugLevel::Debug, "\tStore %zu location groups\n", location_groups.size());
    size_t size = location_groups.size();
    file.write(&size, sizeof(size), 1);
    file.write(location_groups.data(), sizeof(pallas::LocationGroup), location_groups.size());
    file.end_block(__func__);
}

static void readLocationGroups(std::vector<pallas::LocationGroup>& location_groups, pallas::File& file, uint8_t abi_version) {
    file.begin_block(__func__);
    size_t size;
    file.read(&size, sizeof(size), 1);
    location_groups.resize(size);
    if (location_groups.empty())
        goto out;

    file.read(location_groups.data(), sizeof(pallas::LocationGroup), location_groups.size());
    std::sort(location_groups.begin(), location_groups.end(), [](pallas::LocationGroup a, pallas::LocationGroup b) { return a.id < b.id; });
    pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu location_groups\n", location_groups.size());
out:
    file.end_block(__func__);
}

static void storeLocations(std::vector<pallas::Location>& locations, pallas::File& file) {
    file.begin_block(__func__);
    pallas_log(pallas::DebugLevel::Debug, "\tStore %zu locations\n", locations.size());
    for (auto& l : locations) {
        pallas_assert(l.id != PALLAS_THREAD_ID_INVALID);
    }
    size_t size = locations.size();
    file.write(&size, sizeof(size), 1);
    file.write(locations.data(), sizeof(pallas::Location), locations.size());
    file.end_block(__func__);
}

static void readLocations(std::vector<pallas::Location>& locations, pallas::File& file, uint8_t abi_version) {
    file.begin_block(__func__);
    size_t size;
    file.read(&size, sizeof(size), 1);
    locations.resize(size);
    if (locations.empty())
        goto out;
    file.read(locations.data(), sizeof(pallas::Location), locations.size());
    std::sort(locations.begin(), locations.end(), [](pallas::Location a, pallas::Location b) { return a.id < b.id; });
    pallas_log(pallas::DebugLevel::Debug, "\tLoad %lu locations\n", locations.size());
out:
    file.end_block(__func__);
}

static void storeMetadata(pallas::Metadata& metadata, pallas::File& file) {
    file.begin_block(__func__);
    pallas_log(pallas::DebugLevel::Debug, "\tStoring metadata.\n");
    auto size = metadata.size();
    file.write(&size, sizeof(size), 1);
    for (auto& [key, value] : metadata) {
        file.writeString(key);
        file.writeString(value);
    }
    file.end_block(__func__);
}

static void readMetadata(pallas::Metadata& metadata, pallas::File& file, uint8_t abi_version) {
    file.begin_block(__func__);
    pallas_log(pallas::DebugLevel::Debug, "\tReading metadata.\n");
    size_t size;
    file.read(&size, sizeof(size), 1);
    if (abi_version == 16) {
        fseek(file.file, sizeof(size) + size, SEEK_CUR);
        static bool metadata_warning_set = true;
        if (metadata_warning_set) {
            pallas_warn("Could not read Thread metadata, ABI too low: %d < 17\n", abi_version);
            metadata_warning_set = false;
        }
        goto out;
    }
    for (size_t i = 0; i < size; i++) {
        auto key = file.readString();
        auto value = file.readString();
        metadata[key] = value;
    }
out:
    file.end_block(__func__);
}

static pallas::File pallasGetThreadFile(const char* dir_name, pallas::Thread* thread, const char* mode) {
    char filename[1024];
    const char* threadPath = getThreadPath(thread);
    snprintf(filename, 1024, "%s/%s/thread.pallas", dir_name, threadPath);
    delete[] threadPath;
    return pallas::File(filename, mode);
}



void pallasStoreThread(const char* base_dir, pallas::Thread* thread, const pallas::ParameterHandler* parameter_handler, bool load_thread) {
    pallas::File thread_summary_file = pallasGetThreadFile(base_dir, thread, "w");
    
    const char* sequenceDurationFilename = pallasGetSequenceDurationFilename(base_dir, thread);
    pallas::File sequenceDurationFile = pallas::File(sequenceDurationFilename, "w");

    const char* eventDurationFilename = pallasGetEventDurationFilename(base_dir, thread);
    pallas::File eventDurationFile = pallas::File(eventDurationFilename, "w");

    

    pallas_log(pallas::DebugLevel::Verbose, "\tThread %u {.nb_events=%lu, .nb_sequences=%lu, .nb_loops=%lu}\n", 
        thread->id, thread->nb_events, thread->nb_sequences, thread->nb_loops);

    pallas::file_format::storeThread(thread_summary_file,
		   eventDurationFile,
		   sequenceDurationFile,
		   thread,
		   parameter_handler,
		   load_thread);

    sequenceDurationFile.close();
    eventDurationFile.close();

#if 0
           threadFile.begin_block(__func__);

    threadFile.write(&th->id, sizeof(th->id), 1);
    threadFile.write(&th->archive->id, sizeof(th->archive->id), 1);

  threadFile.write(&th->nb_events, sizeof(th->nb_events), 1);
  threadFile.write(&th->nb_sequences, sizeof(th->nb_sequences), 1);
  threadFile.write(&th->nb_loops, sizeof(th->nb_loops), 1);

  threadFile.write(&th->sequence_root, sizeof(th->sequence_root), 1);

    threadFile.write(&th->first_timestamp, sizeof(th->first_timestamp), 1);

  const char* eventDurationFilename = pallasGetEventDurationFilename(path, th);
  pallas::File eventDurationFile = pallas::File(eventDurationFilename, "w");
  delete[] eventDurationFilename;
  for (int i = 0; i < th->nb_events; i++) {
    storeEvent(th->events[i], threadFile, eventDurationFile, parameter_handler, load_thread);
  }
  eventDurationFile.close();

  // write event indirection map
  size_t event_map_size = th->event_id_map.size();
  threadFile.write(&event_map_size, sizeof(size_t), 1);
  if (event_map_size > 0) {
    threadFile.write(th->event_id_map.data(), sizeof(uint32_t), event_map_size);
  }

    

  // write sequence indirection map
  size_t seq_map_size = th->sequence_id_map.size();
  threadFile.write(&seq_map_size, sizeof(size_t), 1);
  if (seq_map_size > 0) {
    threadFile.write(th->sequence_id_map.data(), sizeof(uint32_t), seq_map_size);
  }

    for (int i = 0; i < th->nb_loops; i++) {
        storeLoop(th->loops[i], threadFile);
  }

  // write loop indirection map
  size_t loop_map_size = th->loop_id_map.size();
  threadFile.write(&loop_map_size, sizeof(size_t), 1);
  if (loop_map_size > 0) {
    threadFile.write(th->loop_id_map.data(), sizeof(uint32_t), loop_map_size);
  }

  threadFile.end_block(__func__);
    threadFile.close();

    double effective_ratio = numberCompressedBytes ? (numberRawBytes + .0) / numberCompressedBytes : 0.0;
    double true_ratio = numberCompressedBytes ? (numberPreRawBytes + .0) / numberCompressedBytes : 0.0;
    pallas_log(pallas::DebugLevel::Error,
               "Storage bytes: pre_raw=%lu raw=%lu compressed=%lu effective_ratio=%.2f true_ratio=%.2f\n",
               numberPreRawBytes,
               numberRawBytes,
               numberCompressedBytes,
               effective_ratio,
               true_ratio);

    #endif
}

void pallas::Thread::store(const char* path, const ParameterHandler* parameter_handler, bool load_thread) {
    pallasStoreThread(path, this, parameter_handler, load_thread);
}

void pallas::Archive::store(const char* path, const ParameterHandler* parameter_handler) {
    pallasStoreArchive(this, path, parameter_handler);
}

void pallas::GlobalArchive::store(const char* path, const ParameterHandler* parameter_handler) {
    pallasStoreGlobalArchive(this, path, parameter_handler);
}

static void readThread(pallas::GlobalArchive* global_archive, pallas::Thread* th, pallas::ThreadId thread_id, uint8_t abi_version) {
    th->id = thread_id;
    pallas::File threadFile = pallasGetThreadFile(global_archive->dir_name, th, "r");
    if (!threadFile.is_open()) {
        return;
    }
    threadFile.begin_block(__func__);
    threadFile.read(&th->id, sizeof(th->id), 1);
    pallas::LocationGroupId archive_id;
    threadFile.read(&archive_id, sizeof(archive_id), 1);
    // This used to be used for something, but not anymore.

    threadFile.read(&th->nb_events, sizeof(th->nb_events), 1);
    th->nb_allocated_events = th->nb_events;
    th->events = new pallas::Event[th->nb_allocated_events];

    threadFile.read(&th->nb_sequences, sizeof(th->nb_sequences), 1);
    th->nb_allocated_sequences = th->nb_sequences;
    th->sequences = new pallas::Sequence[th->nb_allocated_sequences];

    threadFile.read(&th->nb_loops, sizeof(th->nb_loops), 1);
    th->nb_allocated_loops = th->nb_loops;
    th->loops = new pallas::Loop[th->nb_allocated_loops];

  if (abi_version >= 20) {
    threadFile.read(&th->sequence_root, sizeof(th->sequence_root), 1);
  } else {
    th->sequence_root = 0;
  }

    threadFile.read(&th->first_timestamp, sizeof(th->first_timestamp), 1);

  pallas_log(pallas::DebugLevel::Verbose, "Reading %lu events\n", th->nb_events);
  const char* eventDurationFilename = pallasGetEventDurationFilename(global_archive->dir_name, th);
  if (fileMap.find(eventDurationFilename) == fileMap.end()) {
    fileMap[eventDurationFilename] = new pallas::File(eventDurationFilename);;
  }
  for (size_t i = 0; i < th->nb_events; i++) {
    th->events[i].id = i;
    readEvent(th->events[i], threadFile, *fileMap[eventDurationFilename], *global_archive->parameter_handler, abi_version);
  }

  // read events with indirection map if supported
  if (abi_version >= 20) {
    size_t event_map_size;
    threadFile.read(&event_map_size, sizeof(size_t), 1);
    th->event_id_map.resize(event_map_size);
    if (event_map_size > 0) {
      threadFile.read(th->event_id_map.data(), sizeof(uint32_t), event_map_size);
    }
    for (size_t logi_id = 0; logi_id < th->event_id_map.size(); logi_id++) {
      uint32_t phys_id = th->event_id_map[logi_id];
      if (phys_id != PALLAS_INDEX_INVALID) {
        th->events[phys_id].id = logi_id;
      }
    }
  } else {
    th->event_id_map.resize(th->nb_events);
    for (size_t i = 0; i < th->nb_events; i++) {
      th->events[i].id = i;
      th->event_id_map[i] = i;
    }
  }

  pallas_log(pallas::DebugLevel::Verbose, "Reading %lu sequences\n", th->nb_sequences);
  const char* sequenceDurationFilename = pallasGetSequenceDurationFilename(global_archive->dir_name, th);
  if (fileMap.find(sequenceDurationFilename) == fileMap.end()) {
    fileMap[sequenceDurationFilename] = new pallas::File(sequenceDurationFilename);;
  }
  for (size_t i = 0; i < th->nb_sequences; i++) {
    th->sequences[i].id = PALLAS_SEQUENCE_ID(i);
    readSequence(th->sequences[i], threadFile, *fileMap[sequenceDurationFilename], *global_archive->parameter_handler, abi_version);
  }

  // read sequences with indirection map if supported
  if (abi_version >= 20) {
    size_t seq_map_size;
    threadFile.read(&seq_map_size, sizeof(size_t), 1);
    th->sequence_id_map.resize(seq_map_size);
    if (seq_map_size > 0) {
      threadFile.read(th->sequence_id_map.data(), sizeof(uint32_t), seq_map_size);
    }
    for (size_t logi_id = 0; logi_id < th->sequence_id_map.size(); logi_id++) {
      uint32_t phys_id = th->sequence_id_map[logi_id];
      if (phys_id != PALLAS_INDEX_INVALID) {
        th->sequences[phys_id].id = PALLAS_SEQUENCE_ID(logi_id);
      }
    }
  } else {
    th->sequence_id_map.resize(th->nb_sequences);
    for (size_t i = 0; i < th->nb_sequences; i++) {
      th->sequences[i].id = PALLAS_SEQUENCE_ID(i);
      th->sequence_id_map[i] = i;
    }
  }

  pallas_log(pallas::DebugLevel::Verbose, "Reading %lu loops\n", th->nb_loops);
  for (size_t i = 0; i < th->nb_loops; i++) {
    th->loops[i].self_id = PALLAS_LOOP_ID(i);
    readLoop(th->loops[i], threadFile, abi_version);
  }

  // read loops with indirection map if supported
  if (abi_version >= 20) {
    size_t loop_map_size;
    threadFile.read(&loop_map_size, sizeof(size_t), 1);
    th->loop_id_map.resize(loop_map_size);
    if (loop_map_size > 0) {
      threadFile.read(th->loop_id_map.data(), sizeof(uint32_t), loop_map_size);
    }
    for (size_t logi_id = 0; logi_id < th->loop_id_map.size(); logi_id++) {
      uint32_t phys_id = th->loop_id_map[logi_id];
      if (phys_id != PALLAS_INDEX_INVALID) {
        th->loops[phys_id].self_id = PALLAS_LOOP_ID(logi_id);
      }
    }
  } else {
    th->loop_id_map.resize(th->nb_loops);
    for (size_t i = 0; i < th->nb_loops; i++) {
      th->loops[i].self_id = PALLAS_LOOP_ID(i);
      th->loop_id_map[i] = i;
    }
  }
  threadFile.end_block(__func__);
  threadFile.close();

    pallas_log(pallas::DebugLevel::Verbose, "\tThread %u: {.nb_events=%lu, .nb_sequences=%lu, .nb_loops=%lu}\n", th->id, th->nb_events, th->nb_sequences, th->nb_loops);
}

void pallasStoreGlobalArchive(pallas::GlobalArchive* archive, const char* path, const pallas::ParameterHandler* parameter_handler) {
    pallas_log(pallas::DebugLevel::Debug, "Storing global archive\n");
    if (!archive)
        return;

    std::filesystem::path fullpath(std::string(path) + "/" + std::string(archive->trace_name));
    if (fullpath.extension() != ".pallas") {
        fullpath += ".pallas";
    }

    pallas::File file = pallas::File(fullpath.c_str(), "w");
    if (!file.is_open())
        pallas_abort();
    file.begin_block(__func__);
    uint8_t version = PALLAS_ABI_VERSION;
    file.write(&version, sizeof(version), 1);
    parameter_handler->writeToFile(&file);

    storeDefinitions(archive->definitions, file);
    storeLocationGroups(archive->location_groups, file);
    storeLocations(archive->locations, file);
    storeMetadata(archive->metadata, file);
    file.end_block(__func__);
    file.close();
}

char* pallas_archive_fullpath(pallas::Archive* a, const char* path) {
    int len = strlen(path) + 32;
    char* fullpath = new char[len];
    snprintf(fullpath, len, "%s/archive_%u/archive.pallas", path, a->id);
    return fullpath;
}

void pallasStoreArchive(pallas::Archive* archive, const char* path, const pallas::ParameterHandler* parameter_handler) {
    pallas_log(pallas::DebugLevel::Debug, "Storing archive %d\n", archive->id);
    if (!archive)
        return;

    char* fullpath = pallas_archive_fullpath(archive, path);
    pallas::File file = pallas::File(fullpath, "w");
    if (!file.is_open())
        pallas_abort();
    delete[] fullpath;
    file.begin_block(__func__);
    file.write(&archive->id, sizeof(pallas::LocationGroupId), 1);
#ifdef DEBUG
    if (archive->locations.size() != archive->nb_threads) {
        pallas_warn("archive.locations (%lu) != archive.nb_threads (%lu)\n", archive->locations.size(), archive->nb_threads);
    }
#endif
    pallas_log(pallas::DebugLevel::Verbose, "Archive %d has %lu threads\n", archive->id, archive->nb_threads);
    file.write(&archive->nb_threads, sizeof(int), 1);
    storeDefinitions(archive->definitions, file);
    storeLocationGroups(archive->location_groups, file);
    storeLocations(archive->locations, file);
    storeMetadata(archive->metadata, file);

    file.end_block(__func__);
    file.close();

#ifdef BMARK
    bmark_write_archive_csv(archive, path);
#endif
}

static char* pallas_archive_filename(pallas::GlobalArchive* archive, pallas::LocationGroupId id) {
    size_t tracename_len = strlen(archive->trace_name) + 1;
    pallas_assert(tracename_len >= 8);
    size_t extension_index = tracename_len - 8;
    pallas_assert(strcmp(&archive->trace_name[extension_index], ".pallas") == 0);

    char* trace_basename = new char[tracename_len];
    strncpy(trace_basename, archive->trace_name, extension_index);
    trace_basename[extension_index] = '\0';

    int len = strlen("archive") * 2 + 32 + 5;
    char* result = new char[len];
    snprintf(result, len, "archive_%d/archive.pallas", id);
    return result;
}

void pallas::ParameterHandler::writeToFile(pallas::File* file) const {
    file->begin_block(__func__);
    file->write(&compressionAlgorithm, sizeof(compressionAlgorithm), 1);
    file->write(&encodingAlgorithm, sizeof(encodingAlgorithm), 1);
    file->write(&zstdCompressionLevel, sizeof(zstdCompressionLevel), 1);
    file->write(&loopFindingAlgorithm, sizeof(loopFindingAlgorithm), 1);
    file->write(&maxLoopLength, sizeof(maxLoopLength), 1);
    file->write(&overrideLoopDetection, sizeof(overrideLoopDetection), 1);
    file->write(&timestampStorage, sizeof(timestampStorage), 1);
    file->write(&storagePolicy, sizeof(storagePolicy), 1);
    file->write(&timeLossyPolicy, sizeof(timeLossyPolicy), 1);
    file->write(&durationLossyPolicy, sizeof(durationLossyPolicy), 1);
    file->end_block(__func__);
}

pallas::ParameterHandler::ParameterHandler(pallas::File* file) {
    readFromFile(file);
}

void pallas::ParameterHandler::readFromFile(pallas::File* file) {
    file->begin_block(__func__);
    pallas_log(pallas::DebugLevel::Debug, "Reading configuration from trace.\n");
    file->read(&compressionAlgorithm, sizeof(compressionAlgorithm), 1);
    file->read(&encodingAlgorithm, sizeof(encodingAlgorithm), 1);
    file->read(&zstdCompressionLevel, sizeof(zstdCompressionLevel), 1);
    file->read(&loopFindingAlgorithm, sizeof(loopFindingAlgorithm), 1);
    file->read(&maxLoopLength, sizeof(maxLoopLength), 1);
    file->read(&overrideLoopDetection, sizeof(overrideLoopDetection), 1);
    file->read(&timestampStorage, sizeof(timestampStorage), 1);
    file->read(&storagePolicy, sizeof(storagePolicy), 1);
    file->read(&timeLossyPolicy, sizeof(timeLossyPolicy), 1);
    file->read(&durationLossyPolicy, sizeof(durationLossyPolicy), 1);
    file->end_block(__func__);
    pallas_log(pallas::DebugLevel::Debug, "%s\n", this->to_string().c_str());
}

pallas::Archive* pallas::GlobalArchive::getArchive(pallas::LocationGroupId archive_id) {
  /* check if archive_id is already known */
  for (int i = 0; i < nb_archives; i++) {
    if (archive_list[i] != nullptr && archive_list[i]->id == archive_id) {
      return archive_list[i];
    }
  }


    auto* archive = new Archive(*this, archive_id);

    const char* fullpath = pallas_archive_fullpath(archive, archive->dir_name);

    pallas_log(pallas::DebugLevel::Debug, "Reading archive @ %s\n", fullpath);

    auto file = File(fullpath, "r");
    delete[] fullpath;
    if (!file.is_open()) {
        pallas_warn("I can't read %s: %s\n", file.path, strerror(errno));
        return nullptr;
    }
    file.begin_block(__func__);
    file.read(&archive->id, sizeof(pallas::LocationGroupId), 1);
    file.read(&archive->nb_threads, sizeof(int), 1);
    archive->threads = new pallas::Thread*[archive->nb_threads]();
    archive->nb_allocated_threads = archive->nb_threads;
    readDefinitions(archive->definitions, file, abi_version);
    readLocationGroups(archive->location_groups, file, abi_version);
    readLocations(archive->locations, file, abi_version);
    for (auto& l : locations) {
        if (l.parent == archive->id)
            archive->locations.emplace_back(l);
    }
    readMetadata(archive->metadata, file, abi_version);
    file.end_block(__func__);
    file.close();

    int index = 0;
    while (archive_list[index] != nullptr) {
        index++;
        if (index >= nb_archives) {
            pallas_error("Tried to load more archives than there are.\n");
        }
    }
    archive_list[index] = archive;

    return archive;
}

void pallas::GlobalArchive::freeArchive(pallas::LocationGroupId archiveId) {
    for (int i = 0; i < nb_archives; i++) {
        if (archive_list[i] != nullptr && archive_list[i]->id == archiveId) {
            delete archive_list[i];
            archive_list[i] = nullptr;
            return;
        }
    }
};

/**
 * Getter for a Thread from its id. Loads it from a file if need be.
 * @returns First Thread matching the given pallas::ThreadId, or nullptr if it doesn't have a match.
 */
pallas::Thread* pallas::Archive::getThread(ThreadId thread_id) {
    for (int i = 0; i < nb_threads; i++) {
        if (threads[i] && threads[i]->id == thread_id)
            return threads[i];
    }
    pallas_log(pallas::DebugLevel::Verbose, "Loading Thread %d in Archive %d\n", thread_id, id);
    auto* thread = new Thread();
    auto location = getLocation(thread_id);
    if (location == nullptr) {
        pallas_warn("Archive::getThread(%u): could not find matching Location\n", thread_id);
        return nullptr;
    }
    auto parent = global_archive->getLocationGroup(location->parent);
    if (id == parent->id) {
        thread->archive = this;
        readThread(global_archive, thread, location->id, global_archive->abi_version);
        auto index = thread_id - locations[0].id;
        threads[index] = thread;
        return thread;
    }
    pallas_warn("Archive::getThread(%u): Location's parent isn't us: %u != %u\n", thread_id, id, parent->id);
    return nullptr;
}

pallas::Thread* pallas::Archive::getThreadAt(size_t index) {
    if (index >= nb_threads) {
        return nullptr;
    }
    return getThread(locations[index].id);
}

void pallas::Archive::freeThread(pallas::ThreadId thread_id) {
    pallas_log(DebugLevel::Debug, "{%p}.freeThread(%d)\n", this, thread_id);
    for (int i = 0; i < nb_threads; i++) {
        if (threads[i] && threads[i]->id == thread_id) {
            delete threads[i];
            threads[i] = nullptr;
        }
    }
};

void pallas::Archive::freeThreadAt(size_t i) {
    pallas_log(DebugLevel::Debug, "{%p}.freeThreadAt(%lu)\n", this, i);
    if (i < nb_threads) {
        delete threads[i];
        threads[i] = nullptr;
    }
};

pallas::GlobalArchive* pallas_open_trace(const char* trace_filename) {
    std::filesystem::path path(trace_filename);
    path = std::filesystem::absolute(path);
    std::string trace_name = path.filename();
    std::string dir_name = path.parent_path();

    pallas::File file = pallas::File(path.c_str(), "r");
    if (!file.is_open())
        return nullptr;
    uint8_t abi_version;
    file.begin_block(__func__);
    file.read(&abi_version, sizeof(abi_version), 1);
    auto minimum_compatible_abi = 16;
    if (abi_version != PALLAS_ABI_VERSION) {
        if (abi_version < minimum_compatible_abi) {
            pallas_error("This trace uses Pallas ABI version %d, but the current installation (%d) only supports version over %d\n", abi_version, PALLAS_ABI_VERSION,
                         minimum_compatible_abi);
        }
        pallas_warn("This trace uses Pallas ABI version %d, which is compatible with current version %d\n", abi_version, PALLAS_ABI_VERSION);
    }
    auto* trace = new pallas::GlobalArchive(dir_name.c_str(), trace_name.c_str());
    trace->abi_version = abi_version;
    trace->parameter_handler = new pallas::ParameterHandler(&file);
    trace->parameter_handler->does_stats_need_compute = false;
    pallas_log(pallas::DebugLevel::Debug, "Reading GlobalArchive {.dir_name='%s', .trace='%s'}\n", trace->dir_name, trace->trace_name);

    readDefinitions(trace->definitions, file, abi_version);
    readLocationGroups(trace->location_groups, file, abi_version);
    readLocations(trace->locations, file, abi_version);
    readMetadata(trace->metadata, file, abi_version);
    trace->nb_archives = trace->location_groups.size();
    trace->nb_allocated_archives = trace->location_groups.size();
    if (trace->location_groups.size()) {
        delete[] trace->archive_list;
        trace->archive_list = new pallas::Archive*[trace->location_groups.size()]();
    } else
        trace->archive_list = nullptr;

    file.end_block(__func__);
    file.close();
    return trace;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
