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

#include <sys/syscall.h>  //Debug -> à retirer ensuite
#include <inttypes.h>   //Debug -> à retirer ensuite
#include <sys/types.h>  //Debug -> à retirer ensuite

#ifdef WITH_ZFP
#include <zfp.h>
#endif
#ifdef WITH_SZ
#include <SZ3/api/sz.hpp>
#endif

#include "pallas/pallas.h"
#include "pallas/pallas_dbg.h"
#include "pallas/pallas_parameter_handler.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"

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

static int pallasRecursiveMkdir(const char* dir, __mode_t mode) {
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
    pallas_error("Cannot open %s: %s\n", filename, strerror(errno));
  }
  return file;
}

#define _pallas_fread(ptr, size, nmemb, stream)            \
  do {                                                     \
    size_t ret = fread(ptr, size, nmemb, stream);          \
    if (ret != (nmemb))                                    \
      pallas_error("fread failed: %s\n", strerror(errno)); \
  } while (0)

#define _pallas_fwrite(ptr, size, nmemb, stream)   \
  do {                                             \
    size_t ret = fwrite(ptr, size, nmemb, stream); \
    if (ret != (nmemb))                            \
      pallas_error("fwrite failed\n");             \
  } while (0)

size_t numberOpenFiles = 0;
size_t maxNumberFilesOpen = 32;
void* getFirstOpenFile();
namespace pallas {
class File {
 public:
  FILE* file = nullptr;
  char* path = nullptr;
  bool isOpen = false;
  void open(const char* mode) {
    if (isOpen) {
      pallas_log(DebugLevel::Verbose, "Trying to open file that is already open: %s\n", path);
      // close();
      return;
    }
    while (numberOpenFiles >= maxNumberFilesOpen) {
      auto* openedFilePath = static_cast<pallas::File*>(getFirstOpenFile());
      if (!openedFilePath) {
        pallas_warn("Could not find any more duration files to close: %lu files opened.\n", numberOpenFiles);
        break;
      }
      openedFilePath->close();
    }
    file = pallasFileOpen(path, mode);
    numberOpenFiles++;
    isOpen = true;
  };
  void close() {
    // TODO grab the lock
    if (!isOpen) {
      pallas_log(DebugLevel::Debug, "Trying to close file that is already closed: %s\n", path);
    }
    isOpen = false;
    fclose(file);
    if (numberOpenFiles)
      numberOpenFiles--;
  };
  void read(void* ptr, size_t size, size_t n) const { _pallas_fread(ptr, size, n, file); }
  void write(void* ptr, size_t size, size_t n) const { _pallas_fwrite(ptr, size, n, file); }
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
}  // namespace pallas

std::map<const char*, pallas::File*> fileMap;
void* getFirstOpenFile() {
  for (auto& a : fileMap) {
    if (a.second->isOpen) {
      return a.second;
    }
  }
  return nullptr;
}

static void pallasStoreEvent(pallas::EventSummary& event,
                             const pallas::File& eventFile,
                             const pallas::File& durationFile,
                             pallas::Thread* th);
static void pallasStoreSequence(pallas::Sequence& sequence,
                                const pallas::File& sequenceFile,
                                const pallas::File& durationFile,
                                pallas::Thread* th);

static void pallasStoreLoop(pallas::Loop& loop, const pallas::File& loopFile);

static void pallasStoreString(pallas::Archive* a);
static void pallasStoreRegions(pallas::Archive* a);
static void pallasStoreAttributes(pallas::Archive* a);

static void pallasStoreLocationGroups(pallas::Archive* a);
static void pallasStoreLocations(pallas::Archive* a);

static void pallasReadEvent(pallas::EventSummary& event,
                            const pallas::File& eventFile,
                            const pallas::File& durationFile,
                            const char* durationFileName);
static void pallasReadSequence(pallas::Sequence& sequence,
                               const pallas::File& sequenceFile,
                               const pallas::File& durationFile,
                               const char* durationFileName);
static void pallasReadLoop(pallas::Loop& loop, const pallas::File& loopFile);

static void pallasReadString(pallas::Archive* a);
static void pallasReadRegions(pallas::Archive* a);
static void pallasReadAttributes(pallas::Archive* a);
static void pallasReadLocationGroups(pallas::Archive* a);
static void pallasReadLocations(pallas::Archive* a);

void pallasLoadThread(pallas::Archive* archive, pallas::ThreadId thread_id);

static pallas::Archive* pallasGetArchive(pallas::Archive* global_archive,
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
inline static size_t _pallas_zstd_compress(void* src, size_t size, void* dest, size_t destSize) {
  return ZSTD_compress(dest, destSize, src, size, pallas::parameterHandler->getZstdCompressionLevel());
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
 * @param destSize Size of the compressed array.
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
 * Compresses the content in src using the 1D Lorenzo interpolation
 * @param src The source array.
 * @param n Number of items in the source array.
 * @param compressedSize Size of the compressed array. Passed by ref and modified.
 * @return The compressed array.
 */

inline static byte* _pallas_sz_compress(uint64_t* src, size_t n, size_t& compressedSize) {
  SZ3::Config conf({n});  //usage : SZ3::Config conf({size_dimension_1, size_dimension_2, size_dimension_3})
  conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
  conf.errorBoundMode = SZ3::EB_ABS;  // absolute error
  conf.absErrorBound = 10;            // absolute error bound (in nanosecond)


  byte* compressedArray = reinterpret_cast<byte*>(SZ_compress(conf, src, compressedSize));
  return compressedArray;
}


inline static void _pallas_sz_decompress(size_t n, byte* compressedArray, size_t compressedSize,uint64_t* decompressedArray) {
  SZ3::Config conf({n});  //usage : SZ3::Config conf({size_dimension_1, size_dimension_2, size_dimension_3})
  conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
  conf.errorBoundMode = SZ3::EB_ABS;  // absolute error
  conf.absErrorBound = 10;            // absolute error bound (in nanosecond)

  SZ_decompress(conf, reinterpret_cast<char*>(compressedArray), compressedSize, decompressedArray);
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
  //collectHistogramStats(min, max, src, n);
#endif
  size_t width = max - min;
  // TODO Skip the previous step using the stats from the vector.
  size_t stepSize = (width) / MAX_BIT;

  // Write min/max
  memcpy(dest, &min, sizeof(min));
  dest = &dest[sizeof(min)];  // Offset the address
  memcpy(dest, &max, sizeof(max));
  dest = &dest[sizeof(max)];  // Offset the address

  if (stepSize > 1) {
    // Write each bin
    for (size_t i = 0; i < n; i++) {
      size_t binNumber = (src[i] - min) / stepSize;
      binNumber = (binNumber > MAX_BIT) ? MAX_BIT : binNumber;
      // This last check is here in the rare cases of overflow.
      // printf("Writing %lu as %lu\n", src[i], temp);
      memcpy(&dest[i * N_BYTES], &binNumber, N_BYTES);
      // TODO This will not work on small endians architectures.
    }
  } else {
    for (size_t i = 0; i < n; i++) {
      size_t toWrite = src[i] - min;
      // This MUST be < MAX_BIT
      pallas_assert(toWrite < MAX_BIT);
      memcpy(&dest[i * N_BYTES], &toWrite, N_BYTES);
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
  size_t stepSize = width / MAX_BIT;

  //  printf("Min: %lu; Max: %lu\n", min, max);

  if (stepSize > 1) {
    for (size_t i = 0; i < n; i++) {
      size_t factor = 0;
      memcpy(&factor, &compArray[i * N_BYTES], N_BYTES);
      dest[i] = min + factor * stepSize;
      //    printf("Reading %lu as %lu\n", factor, dest[i]);
    }
  } else {
    for (size_t i = 0; i < n; i++) {
      size_t factor = 0;
      memcpy(&factor, &compArray[i * N_BYTES], N_BYTES);
      dest[i] = min + factor;
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
  * Copy the content of the thread buffer (of timestamps) in the file, following the format 
  * number of uint64_t then the array of timestamps. Compress the buffer if needed.
  * @param file The destination file.
  * @param th The current thread.
  */
inline static void _pallas_flush_durations_buffer(FILE* file, pallas::Thread* th){
  printf("\n==%d=%d== [DEBUG] _pallas_flush_durations_buffer \n", (int)getpid(), (int)gettid());
  size_t n = th->bufferTimestamps->usedSpace;  
  //assert(n >= 0);
  if(n == 0){
    printf("==== [DEBUG] /!\\ n == 0");
    return;
  }

  if (pallas::parameterHandler->getEncodingAlgorithm() != pallas::EncodingAlgorithm::None){
    // Buffer not working with encoding for the moment 
    return;
  }
  uint64_t* src = th->bufferTimestamps->arrayTimestamps;  

  // Init like this in case there no compression
  byte* compressedArray = (byte*)th->bufferTimestamps->arrayTimestamps;  
  size_t compressedSize = n;
  //compressedArray = _pallas_sz_compress(th->bufferTimestamps->arrayTimestamps
  //                    ,n/sizeof(uint64_t), compressedSize);
  //////////////////

  // Compress the buffer

  switch (pallas::parameterHandler->getCompressionAlgorithm()) {
    case pallas::CompressionAlgorithm::None:
      printf("==== [DEBUG] Pas de compression\n");
      break;
    case pallas::CompressionAlgorithm::ZSTD: {
      compressedArray = new byte[compressedSize];
      compressedSize = _pallas_zstd_compress(src, n, compressedArray, compressedSize);
      break;
    }
    case pallas::CompressionAlgorithm::Histogram: {
      compressedSize = N_BYTES * n + 2 * sizeof(uint64_t);
      ;  // Take into account that we add the min and the max.
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
      compressedSize = _pallas_zstd_compress(tempCompressedArray, tempCompressedSize, compressedArray, compressedSize);
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
      printf("==== [DEBUG] _pallas_sz_compress\n");
      compressedArray = _pallas_sz_compress(src, n, compressedSize);
      break;
  #endif
    default:
      pallas_error("Invalid Compression algorithm\n");
  }

  printf("==== [DEBUG] _pallas_fwrite\n");
  _pallas_fwrite(&compressedSize, sizeof(compressedSize), 1, file);
  _pallas_fwrite(compressedArray, compressedSize, 1, file);
  th->bufferTimestamps->usedSpace = 0;
  numberRawBytes += n;
  if(pallas::parameterHandler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None){
    numberCompressedBytes += compressedSize;
  }
  printf("==== [DEBUG] numberRawBytes : %ld (+%ld)\n",numberRawBytes,n);
  printf("==== [DEBUG] numberCompressedBytes : %ld (+%ld)\n\n",numberCompressedBytes,compressedSize);
}

/**
 * Writes the array to the given file, but encodes and compresses it before
 * according to the value of parameterHandler::EncodingAlgorithm and parameterHandler::CompressingAlgorithm.
 * @param src The source array. Contains n elements of 8 bytes (sizeof uint64_t).
 * @param n Number of elements in src.
 * @param file File to write in.
 */

// TODO : change documentation according to changes
inline static void _pallas_compress_write(uint64_t* src, size_t n, FILE* file,pallas::Thread* th) {
  //printf("==== [DEBUG] _pallas_compress_write\n");
  size_t size = n * sizeof(uint64_t);
  uint64_t* encodedArray = nullptr;
  size_t encodedSize = n;


  // First we do the encoding
  switch (pallas::parameterHandler->getEncodingAlgorithm()) {
  case pallas::EncodingAlgorithm::None:
    //printf("==== [DEBUG] Pas d'encodage\n");
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

  // TODO : mettre ça dans une fonction a part ?

  //Buffer not working with encoding

  // Check if enougth space in the buffer
  //printf("==== [DEBUG] Check buffer\n");
  size_t bufferNumberElements = th->bufferTimestamps->usedSpace/sizeof(uint64);
  if (SIZE_BUFFER_TIMESTAMP - th->bufferTimestamps->usedSpace >= n){
    memcpy(th->bufferTimestamps->arrayTimestamps+(uint64_t)bufferNumberElements
          ,src,bufferNumberElements);

    th->bufferTimestamps->usedSpace += n;
    assert(th->bufferTimestamps->usedSpace <= SIZE_BUFFER_TIMESTAMP);
    pallas_log(pallas::DebugLevel::Debug, "buffered %lu bytes\n", n);
    return;
  }

  printf("==== [DEBUG] Manque de place dans buffer (libre %ld, need : %ld)\n",SIZE_BUFFER_TIMESTAMP - th->bufferTimestamps->usedSpace,n);
  uint64_t* toBufferArray = src;
  size_t toBufferSize = n;
  src = th->bufferTimestamps->arrayTimestamps;
  n = th->bufferTimestamps->usedSpace;
  printf("==== [DEBUG] toBufferArray : %p, toBufferSize : %ld , src : %p, n : %ld",toBufferArray,
          toBufferSize,src,n);

  th->bufferTimestamps->usedSpace = 0;  // Buffer flushed

  byte* compressedArray = nullptr;
  size_t compressedSize;
  switch (pallas::parameterHandler->getCompressionAlgorithm()) {
    case pallas::CompressionAlgorithm::None:
      break;
    case pallas::CompressionAlgorithm::ZSTD: {
      compressedSize = ZSTD_compressBound(encodedArray ? encodedSize : size);
      compressedArray = new byte[compressedSize];
      if (encodedArray) {
        compressedSize = _pallas_zstd_compress(encodedArray, encodedSize, compressedArray, compressedSize);
      } else {
        compressedSize = _pallas_zstd_compress(src, size, compressedArray, compressedSize);
      }
      break;
    }
    case pallas::CompressionAlgorithm::Histogram: {
      compressedSize = N_BYTES * n + 2 * sizeof(uint64_t);
      ;  // Take into account that we add the min and the max.
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
      compressedSize = _pallas_zstd_compress(tempCompressedArray, tempCompressedSize, compressedArray, compressedSize);
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

  if (pallas::parameterHandler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None) {
    //TODO add pallas_logs for buffer
    pallas_log(pallas::DebugLevel::Debug, "Compressing %lu bytes as %lu bytes\n", size, compressedSize);
    _pallas_fwrite(&compressedSize, sizeof(compressedSize), 1, file);
    _pallas_fwrite(compressedArray, compressedSize, 1, file);
    numberRawBytes += encodedSize;
    numberCompressedBytes += compressedSize;
  } else if (pallas::parameterHandler->getEncodingAlgorithm() != pallas::EncodingAlgorithm::None) {
    pallas_log(pallas::DebugLevel::Debug, "Encoding %lu bytes as %lu bytes\n", size, encodedSize);
    _pallas_fwrite(&encodedSize, sizeof(encodedSize), 1, file);
    _pallas_fwrite(encodedArray, encodedSize, 1, file);
  } else {
    pallas_log(pallas::DebugLevel::Debug, "Writing %lu bytes as is.\n", size);
    _pallas_fwrite(&size, sizeof(size), 1, file);
    _pallas_fwrite(src, size, 1, file);
  }

  // Then add the element to the buffer
  memcpy(th->bufferTimestamps->arrayTimestamps+th->bufferTimestamps->usedSpace,toBufferArray,toBufferSize);
  th->bufferTimestamps->usedSpace += toBufferSize;

  if (th->bufferTimestamps->usedSpace >= SIZE_BUFFER_TIMESTAMP) {
    printf("Buffer too small: usedSpace = %lu, SIZE_BUFFER_TIMESTAMP = %lu\n",
           th->bufferTimestamps->usedSpace, SIZE_BUFFER_TIMESTAMP);
  }
  assert(th->bufferTimestamps->usedSpace < SIZE_BUFFER_TIMESTAMP);  // Too small buffer

  if (pallas::parameterHandler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None)
    delete[] compressedArray;
  if (pallas::parameterHandler->getEncodingAlgorithm() != pallas::EncodingAlgorithm::None)
    delete[] encodedArray;
}

/**
 * Reads, de-encodes and decompresses an array from the given file,
 * according to the values of parameterHandler::EncodingAlgorithm and parameterHandler::CompressingAlgorithm.
 * @param n Number of elements of 8 bytes dest is supposed to have.
 * @param file File to read from
 * @returns Array of uncompressed data of size uint64_t * n.
 */
inline static uint64_t* _pallas_compress_read(size_t n, FILE* file) {
  size_t expectedSize = n * sizeof(uint64_t);
  uint64_t* uncompressedArray = nullptr;

  size_t compressedSize;
  byte* compressedArray = nullptr;

  size_t encodedSize;
  byte* encodedArray = nullptr;

  auto compressionAlgorithm = pallas::parameterHandler->getCompressionAlgorithm();
  auto encodingAlgorithm = pallas::parameterHandler->getEncodingAlgorithm();
  if (compressionAlgorithm != pallas::CompressionAlgorithm::None) {
    _pallas_fread(&compressedSize, sizeof(compressedSize), 1, file);
    compressedArray = new byte[compressedSize];
    _pallas_fread(compressedArray, compressedSize, 1, file);
  }

  switch (compressionAlgorithm) {
  case pallas::CompressionAlgorithm::None:
    break;
  case pallas::CompressionAlgorithm::ZSTD: {
    if (pallas::parameterHandler->getEncodingAlgorithm() == pallas::EncodingAlgorithm::None) {
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
    uncompressedArray = new uint64_t[n];
    _pallas_sz_decompress(n, compressedArray, compressedSize,uncompressedArray);
    break;
#endif
  default:
    pallas_error("Invalid Compression algorithm\n");
  }

  switch (encodingAlgorithm) {
  case pallas::EncodingAlgorithm::None:
    break;
  case pallas::EncodingAlgorithm::Masking: {
    if (pallas::parameterHandler->getCompressionAlgorithm() == pallas::CompressionAlgorithm::None) {
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

void pallas::LinkedVector::writeToFile(FILE* vectorFile, FILE* valueFile,pallas::Thread* th) {
  if (first)
    finalUpdateStats();
  // Write the statistics to the vectorFile
  _pallas_fwrite(&size, sizeof(size), 1, vectorFile);
  if (size <= 3) {
    _pallas_fwrite(first->array, sizeof(size_t), size, vectorFile);
  } else if (size >= 4) {
    _pallas_fwrite(&min, sizeof(min), 1, vectorFile);
    _pallas_fwrite(&max, sizeof(max), 1, vectorFile);
    _pallas_fwrite(&mean, sizeof(mean), 1, vectorFile);
    offset = ftell(valueFile);
    _pallas_fwrite(&offset, sizeof(offset), 1, vectorFile);

    // And write the timestamps to the valueFile
    auto* buffer = new uint64_t[size];
    uint cur_index = 0;
    SubVector* sub_vec = first;
    while (sub_vec) {
      sub_vec->copyToArray(&buffer[sub_vec->starting_index]);
      cur_index += sub_vec->size;
      sub_vec = sub_vec->next;
    }
    pallas_assert(cur_index == size);
    _pallas_compress_write(buffer, size, valueFile,th);
    delete[] buffer;
    sub_vec = first;
    while (sub_vec) {
      delete[] sub_vec->array;
      auto temp = sub_vec;
      sub_vec = sub_vec->next;
      delete temp;
    }
    first = nullptr;
    last = nullptr;
  }
}

pallas::LinkedVector::LinkedVector(FILE* vectorFile, const char* valueFilePath) {
  filePath = valueFilePath;
  first = nullptr;
  last = nullptr;
  _pallas_fread(&size, sizeof(size), 1, vectorFile);
  if (size == 0) {
    min, max, mean = 0;
  }
  if (size <= 3) {
    auto temp = new size_t[size];
    _pallas_fread(temp, sizeof(size_t), size, vectorFile);
    last = new SubVector(size, temp);
    first = last;
    if (size == 1) {
      max = min;
      mean = min;
    } else if (size == 2) {
      min = std::min(temp[0], temp[1]);
      max = std::max(temp[0], temp[1]);
      mean = (temp[0] + temp[1]) / 2;
    } else {
      min = std::min(temp[0], std::min(temp[1], temp[2]));
      max = std::max(temp[0], std::max(temp[1], temp[2]));
      mean = (temp[0] + temp[1] + temp[2]) / 3;
    }
  } else if (size >= 4) {
    _pallas_fread(&min, sizeof(min), 1, vectorFile);
    _pallas_fread(&max, sizeof(max), 1, vectorFile);
    _pallas_fread(&mean, sizeof(mean), 1, vectorFile);
    offset = 0;
    _pallas_fread(&offset, sizeof(offset), 1, vectorFile);
  }
}

void pallas::LinkedVector::load_timestamps() {
  pallas_log(DebugLevel::Debug, "Loading timestamps from %s\n", filePath);
  pallas::File& f = *fileMap[filePath];
  if (!f.isOpen) {
    f.open("r");
  }
  int ret = fseek(f.file, offset, 0);
  while (ret == EBADF) {
    f.close();
    f.open("r");
    ret = fseek(f.file, offset, 0);
  }
  auto temp = _pallas_compress_read(size, f.file);
  last = new SubVector(size, temp);
  first = last;
}

/**************** Storage Functions ****************/

void pallas_storage_init(pallas::Archive* archive) {
  pallasMkdir(archive->dir_name, 0777);
  pallas_storage_option_init();
}

static const char* base_dirname(pallas::Archive* a) {
  return a->dir_name;
}

static const char* getThreadPath(pallas::Thread* th) {
  char* folderPath = new char[1024];
  //  snprintf(folderPath, 1024, "archive_%u/thread_%u", th->archive->id, th->id);
  snprintf(folderPath, 1024, "thread_%u", th->id);
  return folderPath;
}

static const char* pallasGetEventFilename(const char* base_dirname, pallas::Thread* th) {
  char* filename = new char[1024];
  const char* threadPath = getThreadPath(th);
  snprintf(filename, 1024, "%s/%s/thread.pallas", base_dirname, threadPath);
  delete[] threadPath;
  return filename;
}

static const char* pallasGetEventDurationFilename(const char* base_dirname, pallas::Thread* th) {
  char* filename = new char[1024];
  const char* threadPath = getThreadPath(th);
  snprintf(filename, 1024, "%s/%s/event_durations.dat", base_dirname, threadPath);
  delete[] threadPath;
  return filename;
}

static void _pallas_store_attribute_values(pallas::EventSummary* e, const pallas::File& file) {
  file.write(&e->attribute_pos, sizeof(e->attribute_pos), 1);
  if (e->attribute_pos > 0) {
    pallas_log(pallas::DebugLevel::Debug, "\t\tStore %lu attributes\n", e->attribute_pos);
    if (pallas::parameterHandler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None) {
      size_t compressedSize = ZSTD_compressBound(e->attribute_pos);
      byte* compressedArray = new byte[compressedSize];
      compressedSize = _pallas_zstd_compress(e->attribute_buffer, e->attribute_pos, compressedArray, compressedSize);
      file.write(&compressedSize, sizeof(compressedSize), 1);
      file.write(compressedArray, compressedSize, 1);
      delete[] compressedArray;
    } else {
      file.write(e->attribute_buffer, e->attribute_pos, 1);
    }
  }
}

static void _pallas_read_attribute_values(pallas::EventSummary* e, const pallas::File& file) {
  file.read(&e->attribute_pos, sizeof(e->attribute_pos), 1);
  e->attribute_buffer_size = e->attribute_pos;
  e->attribute_pos = 0;
  e->attribute_buffer = nullptr;

  if (e->attribute_buffer_size > 0) {
    e->attribute_buffer = new uint8_t[e->attribute_buffer_size];
    if (e->attribute_buffer == nullptr) {
      pallas_error("Cannot allocate memory\n");
    }
    if (pallas::parameterHandler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None) {
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

static void pallasStoreEvent(pallas::EventSummary& event,
                             const pallas::File& eventFile,
                             const pallas::File& durationFile,
                             pallas::Thread* th) {
  pallas_log(pallas::DebugLevel::Debug, "\tStore event %d {.nb_events=%zu}\n", event.id, event.durations->size);
  if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
    event.durations->print();
    std::cout << "\n";
  }
  eventFile.write(&event.event, sizeof(pallas::Event), 1);
  eventFile.write(&event.attribute_pos, sizeof(event.attribute_pos), 1);
  if (event.attribute_pos > 0) {
    pallas_log(pallas::DebugLevel::Debug, "\t\tStore %lu attributes\n", event.attribute_pos);
    eventFile.write(event.attribute_buffer, sizeof(byte), event.attribute_pos);
  }
  if (STORE_TIMESTAMPS) {
    event.durations->writeToFile(eventFile.file, durationFile.file,th);
  }
}

static void pallasReadEvent(pallas::EventSummary& event,
                            const pallas::File& eventFile,
                            const pallas::File& durationFile,
                            const char* durationFileName) {
  eventFile.read(&event.event, sizeof(pallas::Event), 1);
  eventFile.read(&event.attribute_buffer_size, sizeof(event.attribute_buffer_size), 1);
  event.attribute_pos = 0;
  event.attribute_buffer = nullptr;
  if (event.attribute_buffer_size > 0) {
    event.attribute_buffer = new byte[event.attribute_buffer_size];
    eventFile.read(event.attribute_buffer, sizeof(byte), event.attribute_buffer_size);
  }
  event.durations = new pallas::LinkedVector(eventFile.file, durationFileName);
}

static const char* pallasGetSequenceFilename(const char* base_dirname, pallas::Thread* th) {
  char* filename = new char[1024];
  const char* threadPath = getThreadPath(th);
  snprintf(filename, 1024, "%s/%s/thread.pallas", base_dirname, threadPath);
  return filename;
}

static const char* pallasGetSequenceDurationFilename(const char* base_dirname, pallas::Thread* th) {
  char* filename = new char[1024];
  const char* threadPath = getThreadPath(th);
  snprintf(filename, 1024, "%s/%s/sequence_durations.dat", base_dirname, threadPath);
  delete[] threadPath;
  return filename;
}

static void pallasStoreSequence(pallas::Sequence& sequence,
                                const pallas::File& sequenceFile,
                                const pallas::File& durationFile,
                                pallas::Thread* th) {
  pallas_log(pallas::DebugLevel::Debug, "\tStore sequence %d {.size=%zu, .nb_ts=%zu}\n", sequence.id, sequence.size(),
             sequence.durations->size);
  if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
    //    th->printSequence(sequence);
    sequence.durations->print();
    std::cout << "\n";
  }
  size_t size = sequence.size();
  sequenceFile.write(&size, sizeof(size), 1);
  sequenceFile.write(sequence.tokens.data(), sizeof(sequence.tokens[0]), sequence.size());
  if (STORE_TIMESTAMPS) {
    sequence.durations->writeToFile(sequenceFile.file, durationFile.file,th);
  }
}

static void pallasReadSequence(pallas::Sequence& sequence,
                               const pallas::File& sequenceFile,
                               const char* durationFileName) {
  size_t size;
  sequenceFile.read(&size, sizeof(size), 1);
  sequence.tokens.resize(size);
  sequenceFile.read(sequence.tokens.data(), sizeof(pallas::Token), size);
  if (STORE_TIMESTAMPS) {
    delete sequence.durations;  // durations is created when making a new Sequence
    sequence.durations = new pallas::LinkedVector(sequenceFile.file, durationFileName);
  }
  pallas_log(pallas::DebugLevel::Debug, "\tLoaded sequence %d {.size=%zu, .nb_ts=%zu}\n", sequence.id, sequence.size(),
             sequence.durations->size);
}

static pallas::File pallasGetLoopFile(const char* base_dirname, pallas::Thread* th, const char* mode) {
  char filename[1024];
  const char* threadPath = getThreadPath(th);
  snprintf(filename, 1024, "%s/%s/thread.pallas", base_dirname, threadPath);
  delete[] threadPath;
  return pallas::File(filename, mode);
}

static void pallasStoreLoop(pallas::Loop& loop, const pallas::File& loopFile) {
  if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
    pallas_log(pallas::DebugLevel::Debug,
               "\tStore loops %d {.nb_loops=%zu, .repeated_token=%d.%d, .nb_iterations:", loop.self_id.id,
               loop.nb_iterations.size(), loop.repeated_token.type, loop.repeated_token.id);
    std::cout << "[";
    for (const auto& i : loop.nb_iterations) {
      std::cout << i << ((&i != &loop.nb_iterations.back()) ? ", " : "]");
    }
    std::cout << "}" << std::endl;
  }
  loopFile.write(&loop.repeated_token, sizeof(loop.repeated_token), 1);
  size_t size = loop.nb_iterations.size();
  loopFile.write(&size, sizeof(size), 1);
  loopFile.write(loop.nb_iterations.data(), sizeof(uint), loop.nb_iterations.size());
}

static void pallasReadLoop(pallas::Loop& loop, const pallas::File& loopFile) {
  loopFile.read(&loop.repeated_token, sizeof(loop.repeated_token), 1);
  size_t size;
  loopFile.read(&size, sizeof(size), 1);
  loop.nb_iterations.resize(size);
  loopFile.read(loop.nb_iterations.data(), sizeof(uint), size);
  if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
    pallas_log(pallas::DebugLevel::Debug,
               "\tLoad loops %d {.nb_loops=%zu, .repeated_token=%d.%d, .nb_iterations: ", loop.self_id.id,
               loop.nb_iterations.size(), loop.repeated_token.type, loop.repeated_token.id);
    std::cout << "[";
    for (const auto& i : loop.nb_iterations) {
      std::cout << i << ((&i != &loop.nb_iterations.back()) ? ", " : "]");
    }
    std::cout << "}" << std::endl;
  }
}

static pallas::File pallasGetStringFile(pallas::Archive* a, const char* mode) {
  char filename[1024];
  if (a->id == PALLAS_MAIN_LOCATION_GROUP_ID)
    snprintf(filename, 1024, "%s/string.dat", base_dirname(a));
  else
    snprintf(filename, 1024, "%s/archive_%u/string.dat", base_dirname(a), a->id);
  return pallas::File(filename, mode);
}

static void pallasStoreStringGeneric(const pallas::File& file, pallas::String* s) {
  pallas_log(pallas::DebugLevel::Debug, "\tStore String {.ref=%d, .length=%d, .str='%s'}\n", s->string_ref, s->length,
             s->str);

  file.write(&s->string_ref, sizeof(s->string_ref), 1);
  file.write(&s->length, sizeof(s->length), 1);
  file.write(s->str, sizeof(char), s->length);
}

static void pallasStoreString(pallas::Archive* a) {
  pallas::File file = pallasGetStringFile(a, "w");
  size_t size = a->definitions.strings.size();
  file.write(&size, sizeof(size), 1);
  for (auto& s : a->definitions.strings) {
    pallasStoreStringGeneric(file, &s.second);
  }
  file.close();
}

static void pallasReadStringGeneric(const pallas::File& file, pallas::String* s) {
  file.read(&s->string_ref, sizeof(s->string_ref), 1);
  file.read(&s->length, sizeof(s->length), 1);
  s->str = new char[s->length];
  pallas_assert(s->str);
  file.read(s->str, sizeof(char), s->length);
  pallas_log(pallas::DebugLevel::Debug, "\tLoad String {.ref=%d, .length=%d, .str='%s'}\n", s->string_ref, s->length,
             s->str);
}

static void pallasReadString(pallas::Archive* a) {
  pallas::File file = pallasGetStringFile(a, "r");
  size_t size;
  file.read(&size, sizeof(size), 1);
  pallas::String tempString;
  for (size_t i = 0; i < size; i++) {
    file.read(&tempString.string_ref, sizeof(tempString.string_ref), 1);
    file.read(&tempString.length, sizeof(tempString.length), 1);
    tempString.str = new char[tempString.length];
    pallas_assert(tempString.str);
    file.read(tempString.str, sizeof(char), tempString.length);
    pallas_log(pallas::DebugLevel::Debug, "\tLoad String {.ref=%d, .length=%d, .str='%s'}\n", tempString.string_ref,
               tempString.length, tempString.str);
    a->definitions.strings[tempString.string_ref] = tempString;
  }
  file.close();
}

static pallas::File pallasGetRegionsFile(pallas::Archive* a, const char* mode) {
  char filename[1024];
  if (a->id == PALLAS_MAIN_LOCATION_GROUP_ID)
    snprintf(filename, 1024, "%s/regions.dat", base_dirname(a));
  else
    snprintf(filename, 1024, "%s/archive_%u/regions.dat", base_dirname(a), a->id);
  return pallas::File(filename, mode);
}

static void pallasStoreRegionsGeneric(const pallas::File& file, pallas::Definition* d) {
  if (d->regions.empty())
    return;

  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu Regions\n", d->regions.size());
  for (auto& region : d->regions) {
    file.write(&region.second, sizeof(pallas::Region), 1);
  }
}

static void pallasStoreRegions(pallas::Archive* a) {
  pallas::File file = pallasGetRegionsFile(a, "w");
  size_t size = a->definitions.regions.size();
  file.write(&size, sizeof(size), 1);
  pallasStoreRegionsGeneric(file, &a->definitions);
  file.close();
}

static void pallasReadRegions(pallas::Archive* a) {
  pallas::File file = pallasGetRegionsFile(a, "r");
  size_t size;
  file.read(&size, sizeof(size), 1);
  pallas::Region tempRegion;
  for (size_t i = 0; i < size; i++) {
    file.read(&tempRegion, sizeof(pallas::Region), 1);
    a->definitions.regions[tempRegion.region_ref] = tempRegion;
  }

  pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu regions\n", a->definitions.regions.size());
  file.close();
}

static pallas::File pallasGetAttributesFile(pallas::Archive* a, const char* mode) {
  char filename[1024];
  if (a->id == PALLAS_MAIN_LOCATION_GROUP_ID)
    snprintf(filename, 1024, "%s/attributes.dat", base_dirname(a));
  else
    snprintf(filename, 1024, "%s/archive_%u/attributes.dat", base_dirname(a), a->id);
  return pallas::File(filename, mode);
}

static void pallasStoreAttributesGeneric(const pallas::File& file, pallas::Definition* d) {
  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu Attributes\n", d->attributes.size());
  for (int i = 0; i < d->attributes.size(); i++) {
    pallas_log(pallas::DebugLevel::Debug, "\t\t[%d] {ref=%d, name=%d, type=%d}\n", i, d->attributes[i].attribute_ref,
               d->attributes[i].name, d->attributes[i].type);
  }

  for (auto& attribute : d->attributes) {
    file.write(&attribute.second, sizeof(pallas::Attribute), 1);
  }
}

static void pallasStoreAttributes(pallas::Archive* a) {
  pallas::File file = pallasGetAttributesFile(a, "w");
  size_t size = a->definitions.attributes.size();
  file.write(&size, sizeof(size), 1);
  pallasStoreAttributesGeneric(file, &a->definitions);
  file.close();
}

static void pallasReadAttributes(pallas::Archive* a) {
  pallas::File file = pallasGetAttributesFile(a, "r");
  size_t size;
  file.read(&size, sizeof(size), 1);
  pallas::Attribute tempAttribute;
  for (size_t i = 0; i < size; i++) {
    file.read(&tempAttribute, sizeof(pallas::Attribute), 1);
    a->definitions.attributes[tempAttribute.attribute_ref] = tempAttribute;
  }

  pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu attributes\n", a->definitions.attributes.size());
  file.close();
}

static pallas::File pallasGetLocationGroupsFile(pallas::Archive* a, const char* mode) {
  char filename[1024];
  if (a->id == PALLAS_MAIN_LOCATION_GROUP_ID)
    snprintf(filename, 1024, "%s/location_groups.dat", base_dirname(a));
  else
    snprintf(filename, 1024, "%s/archive_%u/location_groups.dat", base_dirname(a), a->id);
  return pallas::File(filename, mode);
}

static void pallasStoreLocationGroups(pallas::Archive* a) {
  if (a->location_groups.empty())
    return;

  pallas::File file = pallasGetLocationGroupsFile(a, "w");
  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu location groupds\n", a->location_groups.size());

  file.write(a->location_groups.data(), sizeof(pallas::LocationGroup), a->location_groups.size());
  file.close();
}

static void pallasReadLocationGroups(pallas::Archive* a) {
  if (a->location_groups.empty())
    return;

  pallas::File file = pallasGetLocationGroupsFile(a, "r");

  file.read(a->location_groups.data(), sizeof(pallas::LocationGroup), a->location_groups.size());
  file.close();

  pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu location_groups\n", a->location_groups.size());
}

static pallas::File pallasGetLocationsFile(pallas::Archive* a, const char* mode) {
  char filename[1024];
  if (a->id == PALLAS_MAIN_LOCATION_GROUP_ID)
    snprintf(filename, 1024, "%s/locations.dat", base_dirname(a));
  else
    snprintf(filename, 1024, "%s/archive_%u/locations.dat", base_dirname(a), a->id);
  return pallas::File(filename, mode);
}

static void pallasStoreLocations(pallas::Archive* a) {
  if (a->locations.empty())
    return;

  pallas::File file = pallasGetLocationsFile(a, "w");
  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu locations\n", a->locations.size());

  for (auto& l : a->locations) {
    pallas_assert(l.id != PALLAS_THREAD_ID_INVALID);
  }

  file.write(a->locations.data(), sizeof(pallas::Location), a->locations.size());
  file.close();
}

static void pallasReadLocations(pallas::Archive* a) {
  if (a->locations.empty())
    return;

  pallas::File file = pallasGetLocationsFile(a, "r");

  file.read(a->locations.data(), sizeof(pallas::Location), a->locations.size());
  file.close();

  pallas_log(pallas::DebugLevel::Debug, "\tLoad %lu locations\n", a->locations.size());
}

static pallas::File pallasGetThreadFile(const char* dir_name, pallas::Thread* thread, const char* mode) {
  char filename[1024];
  const char* threadPath = getThreadPath(thread);
  snprintf(filename, 1024, "%s/%s/thread.pallas", dir_name, threadPath);
  delete[] threadPath;
  return pallas::File(filename, mode);
}

static void pallasStoreThread(const char* dir_name, pallas::Thread* th) {
  if (th->nb_events == 0) {
    pallas_log(pallas::DebugLevel::Verbose, "\tSkipping Thread %u {.nb_events=%d, .nb_sequences=%d, .nb_loops=%d}\n",
               th->id, th->nb_events, th->nb_sequences, th->nb_loops);
    abort();
  }

  pallas::File threadFile = pallasGetThreadFile(dir_name, th, "w");

  pallas_log(pallas::DebugLevel::Verbose, "\tThread %u {.nb_events=%d, .nb_sequences=%d, .nb_loops=%d}\n", th->id,
             th->nb_events, th->nb_sequences, th->nb_loops);

  threadFile.write(&th->id, sizeof(th->id), 1);
  threadFile.write(&th->archive->id, sizeof(th->archive->id), 1);

  threadFile.write(&th->nb_events, sizeof(th->nb_events), 1);
  threadFile.write(&th->nb_sequences, sizeof(th->nb_sequences), 1);
  threadFile.write(&th->nb_loops, sizeof(th->nb_loops), 1);

  const char* eventDurationFilename = pallasGetEventDurationFilename(dir_name, th);
  pallas::File eventDurationFile = pallas::File(eventDurationFilename, "w");
  for (int i = 0; i < th->nb_events; i++) {
    pallasStoreEvent(th->events[i], threadFile, eventDurationFile,th);
  }
  _pallas_flush_durations_buffer(eventDurationFile.file,th);
  eventDurationFile.close();

  const char* sequenceDurationFilename = pallasGetSequenceDurationFilename(dir_name, th);
  pallas::File sequenceDurationFile = pallas::File(sequenceDurationFilename, "w");
  for (int i = 0; i < th->nb_sequences; i++) {
    pallasStoreSequence(*th->sequences[i], threadFile, sequenceDurationFile,th);
  }
  _pallas_flush_durations_buffer(sequenceDurationFile.file,th);
  sequenceDurationFile.close();

  for (int i = 0; i < th->nb_loops; i++)
    pallasStoreLoop(th->loops[i], threadFile);
  threadFile.close();
  pallas_log(pallas::DebugLevel::Debug, "Average compression ratio: %.2f\n",
             (numberRawBytes + .0) / numberCompressedBytes);
}

void pallas::Thread::finalizeThread() {
  pallasStoreThread(archive->dir_name, this);
}

static void pallasReadThread(pallas::Archive* global_archive, pallas::Thread* th, pallas::ThreadId thread_id) {
  th->id = thread_id;
  pallas::File threadFile = pallasGetThreadFile(global_archive->dir_name, th, "r");
  threadFile.read(&th->id, sizeof(th->id), 1);
  pallas::LocationGroupId archive_id;
  threadFile.read(&archive_id, sizeof(archive_id), 1);
  th->archive = pallasGetArchive(global_archive, archive_id);

  threadFile.read(&th->nb_events, sizeof(th->nb_events), 1);
  th->nb_allocated_events = th->nb_events;
  th->events = new pallas::EventSummary[th->nb_allocated_events];

  threadFile.read(&th->nb_sequences, sizeof(th->nb_sequences), 1);
  th->nb_allocated_sequences = th->nb_sequences;
  th->sequences = new pallas::Sequence*[th->nb_allocated_sequences];
  for (int i = 0; i < th->nb_sequences; i++) {
    th->sequences[i] = new pallas::Sequence;
  }

  threadFile.read(&th->nb_loops, sizeof(th->nb_loops), 1);
  th->nb_allocated_loops = th->nb_loops;
  th->loops = new pallas::Loop[th->nb_allocated_loops];

  pallas_log(pallas::DebugLevel::Verbose, "Reading %d events\n", th->nb_events);
  const char* eventDurationFilename = pallasGetEventDurationFilename(global_archive->dir_name, th);
  pallas::File& eventDurationFile = *new pallas::File(eventDurationFilename);
  fileMap[eventDurationFilename] = &eventDurationFile;
  for (int i = 0; i < th->nb_events; i++) {
    th->events[i].id = i;
    pallasReadEvent(th->events[i], threadFile, eventDurationFile, eventDurationFilename);
  }

  pallas_log(pallas::DebugLevel::Verbose, "Reading %d sequences\n", th->nb_sequences);
  const char* sequenceDurationFilename = pallasGetSequenceDurationFilename(global_archive->dir_name, th);
  pallas::File& sequenceDurationFile = *new pallas::File(sequenceDurationFilename);
  fileMap[sequenceDurationFilename] = &sequenceDurationFile;
  for (int i = 0; i < th->nb_sequences; i++) {
    th->sequences[i]->id = i;
    pallasReadSequence(*th->sequences[i], threadFile, sequenceDurationFilename);
  }

  pallas_log(pallas::DebugLevel::Verbose, "Reading %d loops\n", th->nb_loops);
  for (int i = 0; i < th->nb_loops; i++) {
    th->loops[i].self_id = PALLAS_LOOP_ID(i);
    pallasReadLoop(th->loops[i], threadFile);
  }
  threadFile.close();

  pallas_log(pallas::DebugLevel::Verbose, "\tThread %u: {.nb_events=%d, .nb_sequences=%d, .nb_loops=%d}\n", th->id,
             th->nb_events, th->nb_sequences, th->nb_loops);
}

void pallas_storage_finalize_thread(pallas::Thread* thread) {
  if (!thread)
    return;
  pallasStoreThread(thread->archive->dir_name, thread);
}

void pallas_storage_finalize(pallas::Archive* archive) {
  if (!archive)
    return;

  char* fullpath;
  size_t fullpath_len;
  if (archive->id == PALLAS_MAIN_LOCATION_GROUP_ID) {
    fullpath_len = strlen(archive->dir_name) + strlen(archive->trace_name) + strlen("%s/%s.pallas");
    fullpath = new char[fullpath_len];
    snprintf(fullpath, fullpath_len, "%s/%s.pallas", archive->dir_name, archive->trace_name);
  } else {
    fullpath_len = strlen(archive->dir_name) + 32 + strlen("%s/archive_%u/archive.pallas");
    fullpath = new char[fullpath_len];
    snprintf(fullpath, fullpath_len, "%s/archive_%u/archive.pallas", archive->dir_name, archive->id);
  }

  pallas::File file = pallas::File(fullpath, "w");
  delete[] fullpath;
  file.write(&archive->id, sizeof(pallas::LocationGroupId), 1);
  if (archive->id == PALLAS_MAIN_LOCATION_GROUP_ID) {
    uint8_t version = PALLAS_ABI_VERSION;
    file.write(&version, sizeof(version), 1);
    pallas::parameterHandler->writeToFile(file.file);
  }
  size_t size;
  size = archive->location_groups.size();
  file.write(&size, sizeof(size), 1);
  size = archive->locations.size();
  file.write(&size, sizeof(size), 1);
  file.write(&archive->nb_threads, sizeof(int), 1);
  //  file.write(&COMPRESSION_OPTIONS, sizeof(COMPRESSION_OPTIONS), 1, f);
  file.write(&STORE_HASHING, sizeof(STORE_HASHING), 1);
  file.write(&STORE_TIMESTAMPS, sizeof(STORE_TIMESTAMPS), 1);

  pallasStoreString(archive);
  pallasStoreRegions(archive);
  pallasStoreAttributes(archive);

  pallasStoreLocationGroups(archive);
  pallasStoreLocations(archive);

  file.close();
}

static char* _archive_filename(pallas::Archive* global_archive, pallas::LocationGroupId id) {
  if (id == PALLAS_MAIN_LOCATION_GROUP_ID)
    return strdup(global_archive->trace_name);

  int tracename_len = strlen(global_archive->trace_name) + 1;
  int extension_index = tracename_len - 8;
  pallas_assert(strcmp(&global_archive->trace_name[extension_index], ".pallas") == 0);

  char trace_basename[tracename_len];
  strncpy(trace_basename, global_archive->trace_name, extension_index);
  trace_basename[extension_index] = '\0';

  int len = strlen("archive") * 2 + 32 + 5;
  char* result = new char[len];
  snprintf(result, len, "archive_%d/archive.pallas", id);
  return result;
}

char* pallas_archive_fullpath(char* dir_name, char* trace_name) {
  int len = strlen(dir_name) + strlen(trace_name) + 2;
  char* fullpath = new char[len];
  snprintf(fullpath, len, "%s/%s", dir_name, trace_name);
  return fullpath;
}

void pallas::ParameterHandler::writeToFile(FILE* file) const {
  _pallas_fwrite(&compressionAlgorithm, sizeof(compressionAlgorithm), 1, file);
  _pallas_fwrite(&encodingAlgorithm, sizeof(encodingAlgorithm), 1, file);
  _pallas_fwrite(&zstdCompressionLevel, sizeof(zstdCompressionLevel), 1, file);
  _pallas_fwrite(&loopFindingAlgorithm, sizeof(loopFindingAlgorithm), 1, file);
  _pallas_fwrite(&maxLoopLength, sizeof(maxLoopLength), 1, file);
  _pallas_fwrite(&timestampStorage, sizeof(timestampStorage), 1, file);
}

void pallas::ParameterHandler::readFromFile(FILE* file) {
  _pallas_fread(&compressionAlgorithm, sizeof(compressionAlgorithm), 1, file);
  _pallas_fread(&encodingAlgorithm, sizeof(encodingAlgorithm), 1, file);
  _pallas_fread(&zstdCompressionLevel, sizeof(zstdCompressionLevel), 1, file);
  _pallas_fread(&loopFindingAlgorithm, sizeof(loopFindingAlgorithm), 1, file);
  _pallas_fread(&maxLoopLength, sizeof(maxLoopLength), 1, file);
  _pallas_fread(&timestampStorage, sizeof(timestampStorage), 1, file);
}

static void pallasReadArchive(pallas::Archive* global_archive,
                              pallas::Archive* archive,
                              char* dir_name,
                              char* trace_name) {
  archive->fullpath = pallas_archive_fullpath(dir_name, trace_name);
  archive->dir_name = dir_name;
  archive->trace_name = trace_name;
  archive->global_archive = global_archive;
  archive->nb_archives = 0;
  archive->nb_allocated_archives = 1;
  archive->archive_list = new pallas::Archive*();
  archive->definitions = pallas::Definition();
  if (archive->archive_list == nullptr) {
    pallas_error("Failed to allocate memory\n");
  }

  pallas_log(pallas::DebugLevel::Debug, "Reading archive {.dir_name='%s', .trace='%s'}\n", archive->dir_name,
             archive->trace_name);

  pallas::File file = pallas::File(archive->fullpath, "r");

  file.read(&archive->id, sizeof(pallas::LocationGroupId), 1);
  if (archive->id == PALLAS_MAIN_LOCATION_GROUP_ID) {
    // Version checking
    uint8_t abi_version;
    file.read(&abi_version, sizeof(abi_version), 1);
    pallas_assert_always(abi_version == PALLAS_ABI_VERSION);
    pallas::parameterHandler = new pallas::ParameterHandler();
    pallas::parameterHandler->readFromFile(file.file);
  }
  size_t size;

  file.read(&size, sizeof(size), 1);
  archive->location_groups.resize(size);
  file.read(&size, sizeof(size), 1);
  archive->locations.resize(size);

  file.read(&archive->nb_threads, sizeof(int), 1);
  if (archive->id == PALLAS_MAIN_LOCATION_GROUP_ID)
    archive->nb_threads = 0;

  archive->threads = (pallas::Thread**)calloc(sizeof(pallas::Thread*), archive->nb_threads);
  archive->nb_allocated_threads = archive->nb_threads;

  //  file.read(&COMPRESSION_OPTIONS, sizeof(COMPRESSION_OPTIONS), 1, f);
  file.read(&STORE_HASHING, sizeof(STORE_HASHING), 1);
  file.read(&STORE_TIMESTAMPS, sizeof(STORE_TIMESTAMPS), 1);

  char* store_timestamps_str = getenv("STORE_TIMESTAMPS");
  if (store_timestamps_str && strcmp(store_timestamps_str, "FALSE") == 0) {
    STORE_TIMESTAMPS = 0;
  }
  archive->store_timestamps = STORE_TIMESTAMPS;

  pallasReadString(archive);

  pallasReadRegions(archive);
  pallasReadAttributes(archive);

  if (!archive->location_groups.empty()) {
    pallasReadLocationGroups(archive);
  }

  if (!archive->locations.empty()) {
    pallasReadLocations(archive);
  }

  if (archive->id == PALLAS_MAIN_LOCATION_GROUP_ID) {
    global_archive = archive;
  }
  for (auto& location : archive->locations) {
    pallas_assert(location.id != PALLAS_THREAD_ID_INVALID);
    pallasLoadThread(global_archive, location.id);
  }

  file.close();
}

static pallas::Archive* pallasGetArchive(pallas::Archive* global_archive,
                                         pallas::LocationGroupId archive_id,
                                         bool print_warning) {
  /* check if archive_id is already known */
  for (int i = 0; i < global_archive->nb_archives; i++) {
    if (global_archive->archive_list[i]->id == archive_id) {
      return global_archive->archive_list[i];
    }
  }

  /* not found. we need to read the archive */
  auto* arch = new pallas::Archive();
  char* filename = _archive_filename(global_archive, archive_id);
  char* fullpath = pallas_archive_fullpath(global_archive->dir_name, filename);
  if (access(fullpath, R_OK) < 0) {
    if (print_warning)
      pallas_warn("I can't read %s: %s\n", fullpath, strerror(errno));
    free(fullpath);
    return nullptr;
  }

  pallas_log(pallas::DebugLevel::Verbose, "Reading archive %s\n", fullpath);
  delete[] fullpath;

  while (global_archive->nb_archives >= global_archive->nb_allocated_archives) {
    INCREMENT_MEMORY_SPACE(global_archive->archive_list, global_archive->nb_allocated_archives, pallas::Archive*);
  }

  pallasReadArchive(global_archive, arch, strdup(global_archive->dir_name), filename);

  int index = global_archive->nb_archives++;
  global_archive->archive_list[index] = arch;

  return arch;
}

void pallasLoadThread(pallas::Archive* archive, pallas::ThreadId thread_id) {
  for (int i = 0; i < archive->nb_threads; i++) {
    if (archive->threads[i] != NULL && archive->threads[i]->id == thread_id) {
      /* thread_id is already loaded */
      return;
    }
  }

  while (archive->nb_threads >= archive->nb_allocated_threads) {
    INCREMENT_MEMORY_SPACE(archive->threads, archive->nb_allocated_threads, pallas::Thread*);
  }

  int index = archive->nb_threads++;
  archive->threads[index] = new pallas::Thread();
  pallasReadThread(archive, archive->threads[index], thread_id);
  pallas_assert(archive->threads[index]->nb_events > 0);
}

void pallas_read_main_archive(pallas::Archive* archive, char* main_filename) {
  auto* temp_main_filename = strdup(main_filename);
  char* trace_name = strdup(basename(temp_main_filename));
  char* dir_name = strdup(dirname(temp_main_filename));
  free(temp_main_filename);

  pallasReadArchive(nullptr, archive, dir_name, trace_name);

  pallas::Archive* global_archive = archive->global_archive;

  if (archive->id == PALLAS_MAIN_LOCATION_GROUP_ID) {
    global_archive = archive;
  }

  for (auto& location : archive->locations) {
    pallasLoadThread(global_archive, location.id);
  }

  for (auto& location_group : archive->location_groups) {
    pallasGetArchive(global_archive, location_group.id, false);
  }
  for (auto& location : archive->locations) {
    pallasGetArchive(global_archive, location.id, false);
  }
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
