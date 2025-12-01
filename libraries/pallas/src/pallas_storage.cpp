/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zstd.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>

#ifdef WITH_ZFP
#include <zfp.h>
#endif
#ifdef WITH_SZ
#include <sz.h>
#endif
#include "pallas/pallas.h"
#include "pallas/pallas_dbg.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_parameter_handler.h"
#include "pallas/pallas_storage.h"

#include <algorithm>
#include <iostream>
#include <filesystem>

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

static void pallasMkdir(const char* dirname, mode_t mode) {
  if (pallasRecursiveMkdir(dirname, mode) != 0) {
    if (errno != EEXIST)
      pallas_error("mkdir(%s) failed: %s\n", dirname, strerror(errno));
  }
}

static FILE* pallasFileOpen(const char* filename, const char* mode) {
  pallas_log(pallas::DebugLevel::Debug, "Open %s with mode %s\n", filename, mode);
  char* filename_copy = strdup(filename);
  pallasMkdir(dirname(filename_copy), 0777);
  free(filename_copy);

  FILE* file = fopen(filename, mode);
  if (file == nullptr) {
    pallas_warn("Cannot open %s: %s\n", filename, strerror(errno));
  }
  return file;
}

#define _pallas_fread(ptr, size, nmemb, stream)            \
  do {                                                     \
    size_t ret = fread(ptr, size, nmemb, stream);          \
    if (ret != (nmemb))                                    \
      pallas_error("fread failed: %d %s\n", errno, strerror(errno)); \
  } while (0)

#define _pallas_fwrite(ptr, size, nmemb, stream)   \
  do {                                             \
    size_t ret = fwrite(ptr, size, nmemb, stream); \
    if (ret != (nmemb))                            \
      pallas_error("fwrite failed\n");             \
  } while (0)

size_t numberOpenFiles = 0;
size_t maxNumberFilesOpen = 32;
class File;

File* getFirstOpenFile();

class File {
public:
    FILE* file = nullptr;
    char* path = nullptr;
    bool isOpen = false;
    bool is_open() const { return isOpen; }
    // TODO Add the file mode to the File class
    void open(const char* mode) {
        if (isOpen) {
            pallas_log(pallas::DebugLevel::Verbose, "Trying to open file that is already open: %s\n", path);
            // store();
            return;
        }
        while (numberOpenFiles >= maxNumberFilesOpen) {
            auto* openedFilePath = getFirstOpenFile();
            if (!openedFilePath) {
                pallas_warn("Could not find any more duration files to store: %lu files opened.\n", numberOpenFiles);
                break;
            }
            openedFilePath->close();
        }
        file = pallasFileOpen(path, mode);
        if (file) {
            numberOpenFiles++;
            isOpen = true;
        }
    };

    void close() {
        // TODO grab the lock
        if (!isOpen) {
            pallas_log(pallas::DebugLevel::Debug, "Trying to store file that is already closed: %s\n", path);
        }
        isOpen = false;
        fclose(file);
        if (numberOpenFiles)
            numberOpenFiles--;
    };

    void read(void* ptr, size_t size, size_t n) const {
        if (size > 0)
            _pallas_fread(ptr, size, n, file);
    }

    void write(void* ptr, size_t size, size_t n) const {
        if (size > 0)
            _pallas_fwrite(ptr, size, n, file);
    }

    explicit File(const char* path, const char* mode = nullptr) {
        this->path = strdup(path);
        if (mode) {
            open(mode);
        }
    }

    ~File() {
        if (isOpen) {
            close();
        }
        // delete file;
        free(path);
    }
};

class FileMap : public std::map<const char*, File*> {
public:
    ~FileMap() {
        for (auto& it : *this) {
            delete it.second;
        }
    }
};

FileMap fileMap;

File* getFirstOpenFile() {
    for (auto& a : fileMap) {
        if (a.second->isOpen) {
            return a.second;
        }
    }
    return nullptr;
}


static void storeEventData(pallas::EventData& event,
                           const File& eventFile,
                           const pallas::ParameterHandler& parameter_handler);
static void storeEvent(pallas::Event& event,
                             const File& eventFile,
                             const File& durationFile,
                             const pallas::ParameterHandler* parameter_handler,
                             bool load_thread);
static void storeSequence(pallas::Sequence& sequence,
                                const File& sequenceFile,
                                const File& durationFile,
                                const pallas::ParameterHandler* parameter_handler,
                                bool load_thread);

static void storeLoop(pallas::Loop& loop, const File& loopFile);

static void storeString(pallas::Definition& definitions, File& file);
static void storeRegions(pallas::Definition& definitions, File& file);
static void storeAttributes(pallas::Definition& definitions, File& file);
static void storeGroups(pallas::Definition& definitions, File& file);
static void storeComms(pallas::Definition& definitions, File& file);
static void storeAdditionalContent(pallas::AdditionalContent<void> *additional_content, File& file);

static void storeLocationGroups(std::vector<pallas::LocationGroup>& location_groups, File& file);
static void storeLocations(std::vector<pallas::Location>& locations, File& file);

static void readEventData(pallas::EventData& event,
    const File& eventFile,
    const pallas::ParameterHandler& parameter_handler);
static void readEvent(pallas::Event& event,
                            const File& eventFile,
                            const File& durationFile,
                            const char* durationFileName,
                            pallas::ParameterHandler& parameter_handler);

static void readLoop(pallas::Loop& loop, const File& loopFile);
static void readSequence(pallas::Sequence& sequence,
                               const File& sequenceFile,
                               const char* durationFileName,
                               pallas::ParameterHandler& parameter_handler);

static void readString(pallas::Definition& definitions, File& file);
static void readRegions(pallas::Definition& definitions, File& file);
static void readAttributes(pallas::Definition& definitions, File& file);
static void readGroups(pallas::Definition& definitions, File& file);
static void readComms(pallas::Definition& definitions, File& file);
static void readLocationGroups(std::vector<pallas::LocationGroup>& location_groups, File& file);
static void readLocations(std::vector<pallas::Location>& locations, File& file);
static void readAdditionalContent(pallas::AdditionalContent<void> *additional_content, File& file);
void pallasLoadThread(pallas::Archive* globalArchive, pallas::ThreadId thread_id);

static pallas::Archive* pallasGetArchive(pallas::GlobalArchive* global_archive,
                                         pallas::LocationGroupId archive_id,
                                         bool print_warning = true);

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
  zfp_type type = zfp_type_int64;                 // array scalar type
  zfp_field* field = zfp_field_1d(src, type, n);  // array metadata
  zfp_stream* zfp = zfp_stream_open(nullptr);     // compressed stream and parameters
  zfp_stream_set_accuracy(zfp, .1);               // set tolerance for fixed-accuracy mode, this is absolute error
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
  zfp_type type = zfp_type_int64;                  // array scalar type
  zfp_field* field = zfp_field_1d(dest, type, n);  // array metadata
  zfp_stream* zfp = zfp_stream_open(nullptr);      // compressed stream and parameters
  zfp_stream_set_accuracy(zfp, .1);                // set tolerance for fixed-accuracy mode, this is absolute error
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
  byte* compressedArray = SZ_compress(SZ_UINT64, src, &compressedSize, 0, 0, 0, 0, n);
  SZ_Finalize();
  return compressedArray;
}

inline static uint64_t* _pallas_sz_decompress(size_t n, byte* compressedArray, size_t compressedSize) {
  return static_cast<uint64_t*>(SZ_decompress(SZ_UINT64, compressedArray, compressedSize, 0, 0, 0, 0, n));
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
        pallas_warn("%lu <= value <= %lu. Problematic value is @%lu:%lu > %d", min, max, i, toWrite, MAX_BIT);
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
inline static void _pallas_compress_write(uint64_t* src, size_t n, FILE* file, const pallas::ParameterHandler* parameter_handler) {
    size_t size = n * sizeof(uint64_t);
    uint64_t* encodedArray = nullptr;
    size_t encodedSize;
    // First we do the encoding
    switch (parameter_handler->getEncodingAlgorithm()) {
    case pallas::EncodingAlgorithm::None:
        break;
    case pallas::EncodingAlgorithm::Masking: {
        encodedArray = new uint64_t[n];
        encodedSize = _pallas_masking_encode(src, (uint8_t*)encodedArray, n);
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
        compressedArray = new uint8_t[compressedSize];
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
        _pallas_fwrite(&compressedSize, sizeof(compressedSize), 1, file);
        _pallas_fwrite(compressedArray, compressedSize, 1, file);
        numberRawBytes += size;
        numberCompressedBytes += compressedSize;
    } else if (parameter_handler->getEncodingAlgorithm() != pallas::EncodingAlgorithm::None) {
        pallas_log(pallas::DebugLevel::Debug, "Encoding %lu bytes as %lu bytes\n", size, encodedSize);
        _pallas_fwrite(&encodedSize, sizeof(encodedSize), 1, file);
        _pallas_fwrite(encodedArray, encodedSize, 1, file);
    } else {
        size_t offset = ftell(file);
        pallas_log(pallas::DebugLevel::Debug, "Writing %lu bytes as is @%lu in %p.\n", size, offset, file);
        _pallas_fwrite(&size, sizeof(size), 1, file);
        _pallas_fwrite(src, size, 1, file);
    }
    if (parameter_handler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None)
        delete[] compressedArray;
    if (parameter_handler->getEncodingAlgorithm() != pallas::EncodingAlgorithm::None)
        delete[] encodedArray;
}

/**
 * Reads, de-encodes and decompresses an array from the given file,
 * according to the values of parameterHandler::EncodingAlgorithm and parameterHandler::CompressingAlgorithm.
 * @param n Number of elements of 8 bytes dest is supposed to have.
 * @param file File to read from
 * @returns Array of uncompressed data of size uint64_t * n.
 */
inline static uint64_t* _pallas_compress_read(size_t n, FILE* file, const pallas::ParameterHandler& parameter_handler) {
  size_t expectedSize = n * sizeof(uint64_t);
  uint64_t* uncompressedArray = nullptr;

  size_t compressedSize;
  byte* compressedArray = nullptr;

  size_t encodedSize;
  byte* encodedArray = nullptr;

  auto compressionAlgorithm = parameter_handler.getCompressionAlgorithm();
  auto encodingAlgorithm = parameter_handler.getEncodingAlgorithm();
  if (compressionAlgorithm != pallas::CompressionAlgorithm::None) {
    _pallas_fread(&compressedSize, sizeof(compressedSize), 1, file);
    compressedArray = new byte[compressedSize];
    _pallas_fread(compressedArray, compressedSize, 1, file);
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
    auto tempUncompressedArray =
      reinterpret_cast<byte*>(_pallas_zstd_read(histogramSize, compressedArray, compressedSize));
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
      _pallas_fread(&encodedSize, sizeof(encodedSize), 1, file);
      encodedArray = new byte[encodedSize];  // Too big but don't care
      _pallas_fread(encodedArray, encodedSize, 1, file);
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

  if (compressionAlgorithm == pallas::CompressionAlgorithm::None &&
      encodingAlgorithm == pallas::EncodingAlgorithm::None) {
    size_t realSize;
    _pallas_fread(&realSize, sizeof(realSize), 1, file);
    uncompressedArray = new uint64_t[n];
    _pallas_fread(uncompressedArray, realSize, 1, file);
    pallas_assert(realSize == n * sizeof(uint64_t));
  }
  return uncompressedArray;
}

void pallas::LinkedVector::SubArray::write_to_file(FILE* file,  const ParameterHandler* parameter_handler) {
    first_value = array[0];
    last_value = array[size-1];
    offset = ftell(file);
    _pallas_compress_write(array, size, file, parameter_handler);
    delete[] array;
    array = nullptr;
}

void pallas::LinkedDurationVector::SubArray::write_to_file(FILE* file,  const ParameterHandler* parameter_handler) {
    offset = ftell(file);
    _pallas_compress_write(array, size, file, parameter_handler);
    delete [] array;
    array = nullptr;
}

void pallas::LinkedVector::write_to_file(FILE* infoFile, FILE* dataFile, const ParameterHandler* parameter_handler) {
    _pallas_fwrite(&size, sizeof(size), 1, infoFile);
    if (size == 0)
        return;
    // Write the Subarrays statistics
    auto* sub_array = first;
    while (sub_array) {
        if (sub_array->array != nullptr) {
            sub_array->write_to_file(dataFile, parameter_handler);
        }
        _pallas_fwrite(&sub_array->size, sizeof(sub_array->size), 1, infoFile);
        _pallas_fwrite(&sub_array->first_value, sizeof(sub_array->first_value), 1, infoFile);
        _pallas_fwrite(&sub_array->last_value, sizeof(sub_array->last_value), 1, infoFile);
        _pallas_fwrite(&sub_array->offset, sizeof(sub_array->offset), 1, infoFile);
        sub_array = sub_array->next;
    }
    free_data();
}

pallas::LinkedVector::SubArray::SubArray(FILE* file, SubArray* previous) {
    _pallas_fread(&size, sizeof(size), 1, file);
    _pallas_fread(&first_value, sizeof(first_value), 1, file);
    _pallas_fread(&last_value, sizeof(last_value), 1, file);
    _pallas_fread(&offset, sizeof(offset), 1, file);
    allocated = 0;
    this->previous = previous;
    if (previous) {
        previous->next = this;
        starting_index = previous->starting_index + previous->size;
    }
}

pallas::LinkedVector::LinkedVector(FILE* vectorFile, const char* valueFilePath, ParameterHandler& parameter_handler) : parameter_handler(parameter_handler) {
    filePath = valueFilePath;
    first = nullptr;
    last = nullptr;
    _pallas_fread(&size, sizeof(size), 1, vectorFile);
    if (size == 0) {
        return;
    }
    size_t temp_size = 0;
    while (temp_size < size) {
        last = new SubArray(vectorFile, last);
        if (first == nullptr) {
            first = last;
        }
        temp_size += last->size;
    }
}

void pallas::LinkedDurationVector::write_to_file(FILE* vectorFile, FILE* valueFile, const ParameterHandler* parameter_handler) {
    _pallas_fwrite(&size, sizeof(size), 1, vectorFile);
    if (size == 0)
        return;
    final_update_mean();
    // Write the statistics to the vectorFile
    _pallas_fwrite(&min, sizeof(min), 1, vectorFile);
    _pallas_fwrite(&max, sizeof(max), 1, vectorFile);
    _pallas_fwrite(&mean, sizeof(mean), 1, vectorFile);

    // Then write the statistics for all the sub_arrays.
    auto* sub_array = first;
    while (sub_array) {
        if (sub_array->array != nullptr) {
            sub_array->write_to_file(valueFile, parameter_handler);
        }
        _pallas_fwrite(&sub_array->size, sizeof(sub_array->size), 1, vectorFile);
        _pallas_fwrite(&sub_array->min, sizeof(sub_array->min), 1, vectorFile);
        _pallas_fwrite(&sub_array->max, sizeof(sub_array->max), 1, vectorFile);
        _pallas_fwrite(&sub_array->mean, sizeof(sub_array->mean), 1, vectorFile);
        _pallas_fwrite(&sub_array->offset, sizeof(sub_array->offset), 1, vectorFile);
        sub_array = sub_array->next;
    }
    free_data();
}

 pallas::LinkedDurationVector::SubArray::SubArray(FILE* file, SubArray* previous) {
    _pallas_fread(&size, sizeof(size), 1, file);
    _pallas_fread(&min, sizeof(min), 1, file);
    _pallas_fread(&max, sizeof(max), 1, file);
    _pallas_fread(&mean, sizeof(mean), 1, file);
    _pallas_fread(&offset, sizeof(offset), 1, file);
    allocated = 0;
    this->previous = previous;
    if (previous) {
        previous->next = this;
        starting_index = previous->starting_index + previous->size;
    }
}


pallas::LinkedDurationVector::LinkedDurationVector(FILE* vectorFile, const char* valueFilePath, ParameterHandler& parameter_handler): parameter_handler(parameter_handler) {
    filePath = valueFilePath;
    first = nullptr;
    last = nullptr;
    _pallas_fread(&size, sizeof(size), 1, vectorFile);
    if (size == 0) {
        return;
    }
    // Load the statistics from the vectorFile
    _pallas_fread(&min, sizeof(min), 1, vectorFile);
    _pallas_fread(&max, sizeof(max), 1, vectorFile);
    _pallas_fread(&mean, sizeof(mean), 1, vectorFile);
    size_t temp_size = 0;
    while (temp_size < size) {
        last = new SubArray(vectorFile, last);
        if (first == nullptr) {
            first = last;
        }
        temp_size += last->size;
    }

}

void pallas::LinkedVector::load_data(SubArray* sub) {
  pallas_log(DebugLevel::Debug, "Loading timestamps from %s @ %lu\n", filePath, sub->offset);
  File& f = *fileMap[filePath];
  if (!f.isOpen) {
    f.open("r");
  }
  int ret = fseek(f.file, sub->offset, 0);
  while (ret == EBADF) {
    f.close();
    f.open("r");
    ret = fseek(f.file, sub->offset, 0);
  }
  sub->array = _pallas_compress_read(sub->size, f.file, parameter_handler);
    parameter_handler.loaded_durations_size += sub->size * sizeof(uint64_t);
    parameter_handler.subvector_queue.emplace_back(sub);
}


void pallas::LinkedDurationVector::load_data(SubArray* sub) {
    pallas_log(DebugLevel::Debug, "Loading timestamps from %s @ %lu\n", filePath, sub->offset);
    File& f = *fileMap[filePath];
    if (!f.isOpen) {
        f.open("r");
    }
    int ret = fseek(f.file, sub->offset, 0);
    while (ret == EBADF) {
        f.close();
        f.open("r");
        ret = fseek(f.file, sub->offset, 0);
    }
    sub->array = _pallas_compress_read(sub->size, f.file, parameter_handler);
    parameter_handler.loaded_durations_size += sub->size * sizeof(uint64_t);
    parameter_handler.subvector_queue.emplace_back(sub);
}

/**************** Storage Functions ****************/

void pallas_storage_init(const char* dir_name) {
  pallasMkdir(dir_name, 0777);
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

static void _pallas_store_attribute_values(pallas::Event* e, const File& file, const pallas::ParameterHandler& parameter_handler) {
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
}

static void _pallas_read_attribute_values(pallas::Event* e, const File& file, const pallas::ParameterHandler& parameter_handler) {
  file.read(&e->attribute_pos, sizeof(e->attribute_pos), 1);
  e->attribute_buffer_size = e->attribute_pos;
  e->attribute_pos = 0;
  e->attribute_buffer = nullptr;

  if (e->attribute_buffer_size > 0) {
    e->attribute_buffer = new uint8_t[e->attribute_buffer_size];
    if (e->attribute_buffer == nullptr) {
      pallas_error("Cannot allocate memory\n");
    }
    if (parameter_handler.getCompressionAlgorithm() != pallas::CompressionAlgorithm::None) {
      size_t compressedSize;
      file.read(&compressedSize, sizeof(compressedSize), 1);
      byte* compressedArray = new byte[compressedSize];
      file.read(e->attribute_buffer, compressedSize, 1);
      e->attribute_buffer =
        reinterpret_cast<uint8_t*>(_pallas_zstd_read(e->attribute_buffer_size, compressedArray, compressedSize));
      delete[] compressedArray;
    } else {
      file.read(e->attribute_buffer, e->attribute_buffer_size, 1);
    }
  }
}
static void storeEventData(pallas::EventData& event,
                             const File& eventFile,
                             const pallas::ParameterHandler& parameter_handler) {
    eventFile.write(&event, event.event_size, 1);
}

static void readEventData(pallas::EventData& event,
                            const File& eventFile,
                            const pallas::ParameterHandler& parameter_handler) {
    eventFile.read(&event.record, sizeof(event.record), 1);
    eventFile.read(&event.event_size, sizeof(event.event_size), 1);
    auto size = event.event_size - offsetof(pallas::EventData, event_data);
    eventFile.read(event.event_data, size, 1);
};

static void storeEvent(pallas::Event& event,
                                    const File& eventFile,
                                    const File& durationFile,
                                    const pallas::ParameterHandler* parameter_handler,
                                    bool load_thread) {
    pallas_log(pallas::DebugLevel::Debug, "\tStore event %d {.nb_events=%zu}\n", event.id, event.timestamps->size);
    pallas_log(pallas::DebugLevel::Debug, "%s\n", event.timestamps->to_string().c_str());
    storeEventData(event.data, eventFile, *parameter_handler);
    eventFile.write(&event.attribute_pos, sizeof(event.attribute_pos), 1);
    if (event.attribute_pos > 0) {
        pallas_log(pallas::DebugLevel::Debug, "\t\tStore %lu attributes\n", event.attribute_pos);
        eventFile.write(event.attribute_buffer, sizeof(byte), event.attribute_pos);
    }
    if (STORE_TIMESTAMPS) {
        if (load_thread) {
            event.timestamps->load_all_data();
            event.timestamps->reset_offsets();
        }
        event.timestamps->write_to_file(eventFile.file, durationFile.file, parameter_handler);
    }
}

static void readEvent(pallas::Event& event,
                                   const File& eventFile,
                                   const File& durationFile,
                                   const char* durationFileName,
                                   pallas::ParameterHandler& parameter_handler) {
    readEventData(event.data, eventFile, parameter_handler);
    eventFile.read(&event.attribute_buffer_size, sizeof(event.attribute_buffer_size), 1);
    event.attribute_pos = 0;
    event.attribute_buffer = nullptr;
    if (event.attribute_buffer_size > 0) {
        event.attribute_buffer = new byte[event.attribute_buffer_size];
        eventFile.read(event.attribute_buffer, sizeof(byte), event.attribute_buffer_size);
    }
    event.timestamps = new pallas::LinkedVector(eventFile.file, durationFileName, parameter_handler);
    event.nb_occurrences = event.timestamps->size;
    pallas_log(pallas::DebugLevel::Debug, "\tLoaded event %d {.nb_events=%zu}\n", event.id, event.timestamps->size);
}

static const char* pallasGetSequenceDurationFilename(const char* base_dirname, pallas::Thread* th) {
    char* filename = new char[1024];
    const char* threadPath = getThreadPath(th);
    snprintf(filename, 1024, "%s/%s/sequence_durations.dat", base_dirname, threadPath);
    delete[] threadPath;
    return filename;
}

static void storeSequence(pallas::Sequence& sequence,
                                const File& sequenceFile,
                                const File& durationFile,
                                const pallas::ParameterHandler* parameter_handler,
                                bool load_thread
        ) {
    pallas_log(pallas::DebugLevel::Debug, "\tStore sequence %d {.size=%zu, .nb_ts=%zu}\n",
               sequence.id.id, sequence.size(), sequence.durations->size);
    if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
        //    th->printSequence(sequence);
        std::cout << "Durations: " << sequence.durations->to_string() << "\n"
                << "Exclusive Durations:" << sequence.exclusive_durations->to_string() << "\n"
                << "Timestamps: " << sequence.timestamps->to_string() << std::endl;
    }
    sequenceFile.write(&sequence.type, sizeof(sequence.type), 1);
    size_t size = sequence.size();
    sequenceFile.write(&size, sizeof(size), 1);
    sequenceFile.write(sequence.tokens.data(), sizeof(sequence.tokens[0]), sequence.size());
#ifdef DEBUG
    for (const auto& t : sequence.tokens) {
        pallas_assert(t.isValid());
    }
#endif
    if (STORE_TIMESTAMPS) {
        if (load_thread) {
            sequence.durations->load_all_data();
            sequence.durations->reset_offsets();
        }
        sequence.durations->write_to_file(sequenceFile.file, durationFile.file, parameter_handler);
        if (load_thread) {
            sequence.exclusive_durations->load_all_data();
            sequence.exclusive_durations->reset_offsets();
        }
        sequence.exclusive_durations->write_to_file(sequenceFile.file, durationFile.file, parameter_handler);
        if (load_thread) {
            sequence.timestamps->load_all_data();
            sequence.timestamps->reset_offsets();
        }
        sequence.timestamps->write_to_file(sequenceFile.file, durationFile.file, parameter_handler);
    }
}

static void readSequence(pallas::Sequence& sequence,
                               const File& sequenceFile,
                               const char* durationFileName,
                               pallas::ParameterHandler& parameter_handler) {
    sequenceFile.read(&sequence.type, sizeof(sequence.type), 1);
    size_t size;
    sequenceFile.read(&size, sizeof(size), 1);
    sequence.tokens.resize(size);
    sequenceFile.read(sequence.tokens.data(), sizeof(pallas::Token), size);
    if (STORE_TIMESTAMPS) {
        sequence.durations = new pallas::LinkedDurationVector(sequenceFile.file, durationFileName, parameter_handler);
        sequence.exclusive_durations = new pallas::LinkedDurationVector(sequenceFile.file, durationFileName, parameter_handler);
        sequence.timestamps = new pallas::LinkedVector(sequenceFile.file, durationFileName, parameter_handler);
    }
    pallas_log(pallas::DebugLevel::Debug, "\tLoaded sequence %d {.size=%zu, .nb_ts=%zu}\n", sequence.id.id, sequence.size(),
               sequence.durations->size);
}

static void storeLoop(pallas::Loop& loop, const File& loopFile) {
  if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
    pallas_log(pallas::DebugLevel::Debug, "\tStore loop %d {.repeated_token=%d.%d, .nb_iterations: %u\n",
               loop.self_id.id, loop.repeated_token.type, loop.repeated_token.id, loop.nb_iterations);
    std::cout << "}" << std::endl;
  }
  loopFile.write(&loop.repeated_token, sizeof(loop.repeated_token), 1);
  loopFile.write(&loop.nb_iterations, sizeof(loop.nb_iterations), 1);
}

static void readLoop(pallas::Loop& loop, const File& loopFile) {
  loopFile.read(&loop.repeated_token, sizeof(loop.repeated_token), 1);
  loopFile.read(&loop.nb_iterations, sizeof(loop.nb_iterations), 1);
  if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
    pallas_log(pallas::DebugLevel::Debug, "\tLoad loop %d {.repeated_token=%d.%d, .nb_iterations: %u\n",
               loop.self_id.id, loop.repeated_token.type, loop.repeated_token.id, loop.nb_iterations);
  }
}

static void storeString(pallas::Definition& definitions, File& file) {
  size_t size = definitions.strings.size();
  file.write(&size, sizeof(size), 1);
  for (auto& it : definitions.strings) {
    auto& ref = it.first;
    auto& s = it.second;
    pallas_log(pallas::DebugLevel::Debug, "\tStore String {.ref=%d, .length=%d, .str='%s'}\n", s.string_ref, s.length,
               s.str);
    file.write(&s.string_ref, sizeof(s.string_ref), 1);
    file.write(&s.length, sizeof(s.length), 1);
    file.write(s.str, sizeof(char), s.length);
  }
}

static void readString(pallas::Definition& definitions, File& file) {
  size_t size;
  file.read(&size, sizeof(size), 1);
  for (size_t i = 0; i < size; i++) {
    pallas::StringRef ref;
    file.read(&ref, sizeof(ref), 1);
    pallas::String& string = definitions.strings[ref];
    string.string_ref = ref;
    file.read(&string.length, sizeof(string.length), 1);
    string.str = (char*) calloc(string.length, sizeof(char));
    pallas_assert(string.str);
    file.read(string.str, sizeof(char), string.length);
    pallas_log(pallas::DebugLevel::Debug, "\tLoad String {.ref=%d, .length=%d, .str='%s'}\n", string.string_ref,
               string.length, string.str);
  }
}

static void storeRegions(pallas::Definition& definitions, File& file) {
  size_t size = definitions.regions.size();
  file.write(&size, sizeof(size), 1);
  if (definitions.regions.empty())
    return;

  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu Regions\n", definitions.regions.size());
  for (auto& region : definitions.regions) {
    file.write(&region.second, sizeof(pallas::Region), 1);
  }
}

static void readRegions(pallas::Definition& definitions, File& file) {
  size_t size;
  file.read(&size, sizeof(size), 1);
  pallas::Region tempRegion;
  for (size_t i = 0; i < size; i++) {
    file.read(&tempRegion, sizeof(pallas::Region), 1);
    definitions.regions[tempRegion.region_ref] = tempRegion;
  }

  pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu regions\n", definitions.regions.size());
}

static void storeAttributes(pallas::Definition& definitions, File& file) {
  size_t size = definitions.attributes.size();
  file.write(&size, sizeof(size), 1);
  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu Attributes\n", definitions.attributes.size());
  for (int i = 0; i < definitions.attributes.size(); i++) {
    pallas_log(pallas::DebugLevel::Debug, "\t\t[%d] {ref=%d, name=%d, type=%d}\n", i,
               definitions.attributes[i].attribute_ref, definitions.attributes[i].name,
               definitions.attributes[i].type);
  }

  for (auto& attribute : definitions.attributes) {
    file.write(&attribute.second, sizeof(pallas::Attribute), 1);
  }
}

static void readAttributes(pallas::Definition& definitions, File& file) {
  size_t size;
  file.read(&size, sizeof(size), 1);
  pallas::Attribute tempAttribute;
  for (size_t i = 0; i < size; i++) {
    file.read(&tempAttribute, sizeof(pallas::Attribute), 1);
    definitions.attributes[tempAttribute.attribute_ref] = tempAttribute;
  }

  pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu attributes\n", definitions.attributes.size());
}

static void storeGroups(pallas::Definition& definitions, File& file) {
  size_t size = definitions.groups.size();
  file.write(&size, sizeof(size), 1);
  for (auto& [ref, g] : definitions.groups) {
    pallas_log(pallas::DebugLevel::Debug, "\tStore Group {.ref=%d, .name=%d, .nb_members=%d}\n", g.group_ref, g.name,
               g.numberOfMembers);

    file.write(&g.group_ref, sizeof(g.group_ref), 1);
    file.write(&g.name, sizeof(g.name), 1);
    file.write(&g.numberOfMembers, sizeof(g.numberOfMembers), 1);
    file.write(g.members, sizeof(uint64_t), g.numberOfMembers);
  }
}

static void readGroups(pallas::Definition& definitions, File& file) {
  size_t size;
  file.read(&size, sizeof(size), 1);
  for (size_t i = 0; i < size; i++) {
    pallas::GroupRef ref;
    file.read(&ref, sizeof(ref), 1);
    pallas::Group& tempGroup = definitions.groups[ref];
    tempGroup.group_ref = ref;
    file.read(&tempGroup.name, sizeof(tempGroup.name), 1);
    file.read(&tempGroup.numberOfMembers, sizeof(tempGroup.numberOfMembers), 1);
    tempGroup.members = new uint64_t[tempGroup.numberOfMembers];
    pallas_assert(tempGroup.members);
    file.read(tempGroup.members, sizeof(uint64_t), tempGroup.numberOfMembers);
    pallas_log(pallas::DebugLevel::Debug, "\tLoad Group {.ref=%d, .name=%d, .nb_members=%d}\n", tempGroup.group_ref,
               tempGroup.name, tempGroup.numberOfMembers);
  }
}

static void storeComms(pallas::Definition& definitions, File& file) {
  size_t size = definitions.comms.size();
  file.write(&size, sizeof(size), 1);
  if (definitions.comms.empty())
    return;

  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu Comms\n", definitions.comms.size());
  for (auto& comm : definitions.comms) {
    file.write(&comm.second, sizeof(pallas::Comm), 1);
  }
}

static void readComms(pallas::Definition& definitions, File& file) {
  size_t size;
  file.read(&size, sizeof(size), 1);
  pallas::Comm tempComm;
  for (size_t i = 0; i < size; i++) {
    file.read(&tempComm, sizeof(pallas::Comm), 1);
    definitions.comms[tempComm.comm_ref] = tempComm;
  }

  pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu comms\n", definitions.comms.size());
}

static void storeDefinitions(pallas::Definition& def, File& file) {
  storeString(def, file);
  storeRegions(def, file);
  storeAttributes(def, file);
  storeGroups(def, file);
  storeComms(def, file);
}

static void readDefinitions(pallas::Definition& def, File& file) {
  readString(def, file);
  readRegions(def, file);
  readAttributes(def, file);
  readGroups(def, file);
  readComms(def, file);
}

static void storeLocationGroups(std::vector<pallas::LocationGroup>& location_groups, File& file) {
  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu location groups\n", location_groups.size());
  size_t size = location_groups.size();
  file.write(&size, sizeof(size), 1);
  file.write(location_groups.data(), sizeof(pallas::LocationGroup), location_groups.size());
}

static void readLocationGroups(std::vector<pallas::LocationGroup>& location_groups, File& file) {
  size_t size;
  file.read(&size, sizeof(size), 1);
  location_groups.resize(size);
  if (location_groups.empty())
    return;

  file.read(location_groups.data(), sizeof(pallas::LocationGroup), location_groups.size());
  std::sort(location_groups.begin(), location_groups.end(), [](pallas::LocationGroup a, pallas::LocationGroup b) { return a.id < b.id; });
  pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu location_groups\n", location_groups.size());
}

static void storeLocations(std::vector<pallas::Location> &locations, File& file) {
  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu locations\n", locations.size());
  for (auto& l : locations) {
    pallas_assert(l.id != PALLAS_THREAD_ID_INVALID);
  }
  size_t size = locations.size();
  file.write(&size, sizeof(size), 1);
  file.write(locations.data(), sizeof(pallas::Location), locations.size());
}

static void readLocations(std::vector<pallas::Location>& locations, File& file) {
  size_t size;
  file.read(&size, sizeof(size), 1);
  locations.resize(size);
  if (locations.empty())
    return;
  file.read(locations.data(), sizeof(pallas::Location), locations.size());
  std::sort(locations.begin(), locations.end(), [](pallas::Location a, pallas::Location b) { return a.id < b.id; });
  pallas_log(pallas::DebugLevel::Debug, "\tLoad %lu locations\n", locations.size());
}

static void storeAdditionalContent(pallas::AdditionalContent<void> *additional_content, File& file) {
    pallas_log(pallas::DebugLevel::Debug, "\tStoring additional content.\n");
    // We have to start by leaving enough space to later write the number of content and bytes we wrote
    size_t original_position = ftell(file.file);
    fseek(file.file, sizeof(size_t) * 2, SEEK_CUR);
    size_t sum = 0;
    size_t count = 0;
    while (additional_content != nullptr) {
        size_t before_position = ftell(file.file);
        size_t n_bytes_written = additional_content->write_content(additional_content->content, file.file);
        size_t actual_bytes_written = ftell( file.file ) - before_position;
        if (n_bytes_written == actual_bytes_written) {
            pallas_warn("Mismatch in # of bytes written and # of bytes user write_content returns: %lu != %lu\n", n_bytes_written, actual_bytes_written);
        }
        sum += actual_bytes_written;
        count ++;
        additional_content = additional_content->next;
    }
    // Then go back to the start to write the data.
    fseek(file.file, original_position, SEEK_SET);
    file.write(&sum, sizeof(size_t), 1);
    file.write(&count, sizeof(size_t), 1);
    fseek(file.file, sum, SEEK_CUR);
    pallas_log(pallas::DebugLevel::Debug, "\tStored %lu additional contents for %lu bytes + %lu for padding\n", count, sum, 2* sizeof(size_t));
}

static void readAdditionalContent(pallas::AdditionalContent<void> *additional_content, File& file) {
    pallas_log(pallas::DebugLevel::Verbose, "\tReading additional content\n");
    size_t original_position = ftell(file.file);
    size_t theo_sum;
    size_t theo_count;
    file.read(&theo_sum, sizeof(size_t), 1);
    file.read(&theo_count, sizeof(size_t), 1);
    size_t sum = 0;
    size_t count = 0;
    while (additional_content != nullptr && (sum < theo_sum && count < theo_count)) {
        sum += additional_content->read_content(additional_content->content, file.file);
        count ++;
    }
    if (theo_count != count) {
        pallas_warn("Mismatch in # of data and # of data user described: %lu != %lu\n", theo_count, count);
    }
    if (theo_sum != sum) {
        pallas_warn("Mismatch in # of data written and # of data user defined read_content returns: %lu != %lu\n", theo_sum, sum);
    }
    fseek(file.file, original_position + sizeof(size_t) * 2 + theo_sum, SEEK_SET);
    pallas_log(pallas::DebugLevel::Verbose, "\tRead %lu additional contents for %lu bytes\n", count, sum);
}

static File pallasGetThreadFile(const char* dir_name, pallas::Thread* thread, const char* mode) {
  char filename[1024];
  const char* threadPath = getThreadPath(thread);
  snprintf(filename, 1024, "%s/%s/thread.pallas", dir_name, threadPath);
  delete[] threadPath;
  return File(filename, mode);
}

void pallasStoreThread(const char* path, pallas::Thread* th, const pallas::ParameterHandler* parameter_handler, bool load_thread) {
  File threadFile = pallasGetThreadFile(path, th, "w");
  if(!threadFile.is_open())
    return;

  pallas_log(pallas::DebugLevel::Verbose, "\tThread %u {.nb_events=%lu, .nb_sequences=%lu, .nb_loops=%lu}\n", th->id,
             th->nb_events, th->nb_sequences, th->nb_loops);

  threadFile.write(&th->id, sizeof(th->id), 1);
  threadFile.write(&th->archive->id, sizeof(th->archive->id), 1);

  threadFile.write(&th->nb_events, sizeof(th->nb_events), 1);
  threadFile.write(&th->nb_sequences, sizeof(th->nb_sequences), 1);
  threadFile.write(&th->nb_loops, sizeof(th->nb_loops), 1);

  threadFile.write(&th->first_timestamp, sizeof(th->first_timestamp), 1);

  const char* eventDurationFilename = pallasGetEventDurationFilename(path, th);
  File eventDurationFile = File(eventDurationFilename, "w");
  delete[] eventDurationFilename;
  for (int i = 0; i < th->nb_events; i++) {
    storeEvent(th->events[i], threadFile, eventDurationFile, parameter_handler, load_thread);
  }
  eventDurationFile.close();

  const char* sequenceDurationFilename = pallasGetSequenceDurationFilename(path, th);
  File sequenceDurationFile = File(sequenceDurationFilename, "w");
  delete[] sequenceDurationFilename;
  for (int i = 0; i < th->nb_sequences; i++) {
    storeSequence(th->sequences[i], threadFile, sequenceDurationFile, parameter_handler, load_thread);
  }
  sequenceDurationFile.close();

  for (int i = 0; i < th->nb_loops; i++)
    storeLoop(th->loops[i], threadFile);
  threadFile.close();
  pallas_log(pallas::DebugLevel::Debug, "Average compression ratio: %.2f\n",
             (numberRawBytes + .0) / numberCompressedBytes);
}

void pallas::Thread::store(const char *path, const ParameterHandler* parameter_handler, bool load_thread) {
  pallasStoreThread(path, this, parameter_handler, load_thread);
}

void pallas::Archive::store(const char *path, const ParameterHandler* parameter_handler) {
    pallasStoreArchive(this, path, parameter_handler);
}

void pallas::GlobalArchive::store(const char *path, const ParameterHandler* parameter_handler) {
    pallasStoreGlobalArchive(this, path, parameter_handler);
}

static void readThread(pallas::GlobalArchive* global_archive, pallas::Thread* th, pallas::ThreadId thread_id) {
  th->id = thread_id;
  File threadFile = pallasGetThreadFile(global_archive->dir_name, th, "r");
  if (! threadFile.is_open()) {
    return;
  }
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

  threadFile.read(&th->first_timestamp, sizeof(th->first_timestamp), 1);

  pallas_log(pallas::DebugLevel::Verbose, "Reading %lu events\n", th->nb_events);
  const char* eventDurationFilename = pallasGetEventDurationFilename(global_archive->dir_name, th);
  if (fileMap.find(eventDurationFilename) == fileMap.end()) {
    fileMap[eventDurationFilename] = new File(eventDurationFilename);;
  }
  for (int i = 0; i < th->nb_events; i++) {
    th->events[i].id = i;
    readEvent(th->events[i], threadFile, *fileMap[eventDurationFilename], eventDurationFilename, *global_archive->parameter_handler);
  }

  pallas_log(pallas::DebugLevel::Verbose, "Reading %lu sequences\n", th->nb_sequences);
  const char* sequenceDurationFilename = pallasGetSequenceDurationFilename(global_archive->dir_name, th);
  if (fileMap.find(sequenceDurationFilename) == fileMap.end()) {
    fileMap[sequenceDurationFilename] = new File(sequenceDurationFilename);;
  }
  for (int i = 0; i < th->nb_sequences; i++) {
    th->sequences[i].id = PALLAS_SEQUENCE_ID(i);
    readSequence(th->sequences[i], threadFile, sequenceDurationFilename, *global_archive->parameter_handler);
  }

  pallas_log(pallas::DebugLevel::Verbose, "Reading %lu loops\n", th->nb_loops);
  for (int i = 0; i < th->nb_loops; i++) {
    th->loops[i].self_id = PALLAS_LOOP_ID(i);
    readLoop(th->loops[i], threadFile);
  }
  threadFile.close();

  pallas_log(pallas::DebugLevel::Verbose, "\tThread %u: {.nb_events=%lu, .nb_sequences=%lu, .nb_loops=%lu}\n", th->id,
             th->nb_events, th->nb_sequences, th->nb_loops);
}

void pallasStoreGlobalArchive(pallas::GlobalArchive* archive, const char* path, const pallas::ParameterHandler* parameter_handler) {
    pallas_log(pallas::DebugLevel::Debug, "Storing global archive\n");
    if (!archive)
        return;

    std::filesystem::path fullpath(std::string(path) + "/" + std::string(archive->trace_name));
    if (fullpath.extension() != ".pallas") {
        fullpath += ".pallas";
    }

    File file = File(fullpath.c_str(), "w");
    if (!file.is_open())
        pallas_abort();

    uint8_t version = PALLAS_ABI_VERSION;
    file.write(&version, sizeof(version), 1);
    parameter_handler->writeToFile(file.file);

    storeDefinitions(archive->definitions, file);
    storeLocationGroups(archive->location_groups, file);
    storeLocations(archive->locations, file);
    storeAdditionalContent(archive->additional_content, file);

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
    File file = File(fullpath, "w");
    if (!file.is_open())
        pallas_abort();
    delete[] fullpath;
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
    storeAdditionalContent(archive->additional_content, file);
    file.close();
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



void pallas::ParameterHandler::writeToFile(FILE* file) const {
  _pallas_fwrite(&compressionAlgorithm, sizeof(compressionAlgorithm), 1, file);
  _pallas_fwrite(&encodingAlgorithm, sizeof(encodingAlgorithm), 1, file);
  _pallas_fwrite(&zstdCompressionLevel, sizeof(zstdCompressionLevel), 1, file);
  _pallas_fwrite(&loopFindingAlgorithm, sizeof(loopFindingAlgorithm), 1, file);
  _pallas_fwrite(&maxLoopLength, sizeof(maxLoopLength), 1, file);
  _pallas_fwrite(&timestampStorage, sizeof(timestampStorage), 1, file);
}

pallas::ParameterHandler::ParameterHandler(FILE* file) {
  readFromFile(file);
}


void pallas::ParameterHandler::readFromFile(FILE* file) {
  pallas_log(pallas::DebugLevel::Debug, "Reading configuration from trace.\n");
  _pallas_fread(&compressionAlgorithm, sizeof(compressionAlgorithm), 1, file);
  _pallas_fread(&encodingAlgorithm, sizeof(encodingAlgorithm), 1, file);
  _pallas_fread(&zstdCompressionLevel, sizeof(zstdCompressionLevel), 1, file);
  _pallas_fread(&loopFindingAlgorithm, sizeof(loopFindingAlgorithm), 1, file);
  _pallas_fread(&maxLoopLength, sizeof(maxLoopLength), 1, file);
  _pallas_fread(&timestampStorage, sizeof(timestampStorage), 1, file);
  pallas_log(pallas::DebugLevel::Debug, "%s\n", this->to_string().c_str());
}

pallas::Archive* pallas::GlobalArchive::getArchive(pallas::LocationGroupId archive_id, bool print_warning) {
  /* check if archive_id is already known */
  for (int i = 0; i < nb_archives; i++) {
    if (archive_list[i] != nullptr && archive_list[i]->id == archive_id) {
      return archive_list[i];
    }
  }


  auto* archive = new Archive(*this, archive_id);

  const char* fullpath = pallas_archive_fullpath(archive, archive->dir_name);

  pallas_log(pallas::DebugLevel::Debug, "Reading archive @ %s\n", fullpath );

  auto file = File(fullpath, "r");
  delete[]fullpath;
  if( !file.is_open() ) {
    pallas_warn("I can't read %s: %s\n", file.path, strerror(errno));
    return nullptr;
  }

  file.read(&archive->id, sizeof(pallas::LocationGroupId), 1);
  file.read(&archive->nb_threads, sizeof(int), 1);
  delete[] archive->threads;
  archive->threads = new pallas::Thread*[archive->nb_threads]();
  archive->nb_allocated_threads = archive->nb_threads;
  readDefinitions(archive->definitions, file);
  readLocationGroups(archive->location_groups, file);
  readLocations(archive->locations, file);
  readAdditionalContent(archive->additional_content, file);
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
    readThread(global_archive, thread, location->id);
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
    pallas_log(DebugLevel::Debug, "{%p}.freeThread(%d)\n",this, thread_id);
  for (int i = 0; i < nb_threads; i++) {
    if (threads[i] && threads[i]->id == thread_id) {
      delete threads[i];
      threads[i] = nullptr;
    }
  }
};

void pallas::Archive::freeThreadAt(size_t i) {
    pallas_log(DebugLevel::Debug, "{%p}.freeThreadAt(%lu)\n",this, i);
  if (i < nb_threads) {
    delete threads[i];
    threads[i] = nullptr;
  }
};

pallas::GlobalArchive* pallas_open_trace(const char* trace_filename) {
    std::filesystem::path path (trace_filename);
    path = std::filesystem::absolute(path);
    std::string trace_name = path.filename();
    std::string dir_name = path.parent_path();

    File file = File(path.c_str(), "r");
    if (!file.is_open())
        return nullptr;
    uint8_t abi_version;
    file.read(&abi_version, sizeof(abi_version), 1);
    if (abi_version != PALLAS_ABI_VERSION) {
        pallas_error("This trace uses Pallas ABI version %d, but the current installation only supports version %d\n", abi_version, PALLAS_ABI_VERSION);
    }
    auto* trace = new pallas::GlobalArchive(dir_name.c_str(), trace_name.c_str());
    trace->parameter_handler = new pallas::ParameterHandler(file.file);
    pallas_log(pallas::DebugLevel::Debug, "Reading GlobalArchive {.dir_name='%s', .trace='%s'}\n", trace->dir_name, trace->trace_name);

    readDefinitions(trace->definitions, file);
    readLocationGroups(trace->location_groups, file);
    readLocations(trace->locations, file);
    readAdditionalContent(trace->additional_content, file);
    trace->nb_archives = trace->location_groups.size();
    trace->nb_allocated_archives = trace->location_groups.size();
    if (trace->location_groups.size()) {
        delete[] trace->archive_list;
        trace->archive_list = new pallas::Archive*[trace->location_groups.size()]();
    } else
        trace->archive_list = nullptr;

    file.close();

    for (auto& locationGroup : trace->location_groups) {
        auto* archive = trace->getArchive(locationGroup.id);
        std::copy_if(trace->locations.begin(), trace->locations.end(), std::back_inserter(archive->locations),
                     [locationGroup](pallas::Location l) { return l.parent == locationGroup.id; });
    }
    trace->locations.clear();
    // This weird bit of code with the location is just to make sure that they stay local
    return trace;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
