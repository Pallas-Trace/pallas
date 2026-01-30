# Pallas
[![BSD-3 License](https://img.shields.io/badge/License-BSD3-yellow.svg)](https://opensource.org/license/bsd-3-clause)
![Main Pipeline](https://gitlab.inria.fr/pallas/pallas/badges/main/pipeline.svg)
![Maintained Badge](https://img.shields.io/badge/Maintained%3F-Yes-<colour>.svg)
[![HAL Badge](https://img.shields.io/badge/HAL-04970114-white.svg)](https://inria.hal.science/hal-04970114/)
## Documentation
A detailed documentation is available [here](https://pallas.gitlabpages.inria.fr/pallas/#/).

## Building
Building is done with CMake and any compiler (GCC, Clang, Intel OneAPI).
ZSTD is required.
```bash
git clone https://gitlab.inria.fr/pallas/pallas.git
mkdir -p pallas/build && cd pallas/build
cmake ..
cmake --install . --config Release
```
### Optional libraries
If you want to use SZ and ZFP, you should install them,
then feed their install directory to CMake, using `{SZ/ZFP}_ROOT_DIR`

## Python Library
Pallas comes with a Python library to read your traces.
You can install it locally by running `pip install .`.
Otherwise it is available on the PyPi repository : `pip install pallas_trace`

Its requirements are the following:
- Python >=3.11
- Numpy
- pybind11

You can then read it like this:
```python
import pallas_trace as pallas
trace = pallas.open_trace("<my_trace_file>.pallas")
help( trace ) # Get some help reading you trace :)
```
A full documentation of the Pallas Python API is available [here](https://pallas.gitlabpages.inria.fr/pallas/#/06-api-reference/02-python).
## Usage
### In your application
To use Pallas to log your application, you need to understand the hierarchical structure:
- There's one *GlobalArchive* that stores global information.
- Each *Archive* corresponds to a process / a self-contained information group.
They're identified by LocationGroup
- Each *Thread* refers to a group of events that are to be logged.
They're identified by Location

There is a bit more nuance, but it's what you need for now.
The following code should give you an idea of what to do to start logging  your app in C++.
There is also a C API that functions the same way.
Their documentation is available [here](https://pallas.gitlabpages.inria.fr/pallas/#/06-api-reference/01-cxx).
```CPP
#include <pallas/pallas.h>
#include <pallas/pallas_write.h>
namespace pallas;
int main() {
    GlobalArchive globalArchive("<trace directory>", "<main trace file name>");
    globalArchive.addString(...);                   // Register a String

    // Add a process
    globalArchive.addLocationGroup(<processID>);    // Register a LocationGroup
    Archive archive(globalArchive, <processID>);
    // An Archive is the interface used to store local definitions
    // It's analogous to a process

    // Add a Thread
    archive.addLocation(<threadID>);         // Register a Location
    ThreadWriter threadWriter(archive, <threadID>);
    // A ThreadWriter is the interface used to log events
    // It's analogous to a Thread

    // Start logging
    pallas_record_generic(&threadWriter, <custom Attribute>, <timestamp>, <name>);

    // Write the trace to file
    threadWriter.close();
    globalArchive.close();
}
```


### Using EZTrace

After compiling Pallas and its OTF2 API, install [EZTrace](https://eztrace.gitlab.io/eztrace).
Make sure to build it from source, and to use the Pallas OTF2 library, not the normal OTF2 library.
You can check `which otf2-config` to see if you have the correct one. If not, check your PATH and LD_LIBRARY_PATH variables.

Make sure to enable the relevant ezTrace modules.
Then trace any program by running `mpirun -np N eztrace -t <your modules> <your programm>`.
The trace file will be generated in the `<your programm>_trace` folder.
You can then read it using `pallas_print <your programm>_trace/eztrace_log.pallas`

### Visualizing Pallas traces

[Blup](https://gitlab.inria.fr/blup/blup) is a web-based trace
visualizer able to display Pallas traces. It uses Pallas Python API.

![](https://gitlab.inria.fr/blup/blup/-/raw/main/doc/screenshot.png)

## About

Pallas implements a subset of the [OTF2](https://www.vi-hps.org/projects/score-p) API.
It also implements the [Murmur3 hashing function](https://github.com/PeterScott/murmur3).

## Configuration
You can visualise your current (static) configuration with `pallas_config`.
You can give your own dynamic configuration with the `PALLAS_CONFIG_PATH` environment variable.
Here are the configuration options with specific values:

- `compressionAlgorithm`: Specifies which compression algorithm is used for storing the timestamps. Its values are:
  - `None`
  - `ZSTD`
  - `SZ`
  - `ZFP`
  - `Histogram`
  - `ZSTD_Histogram`
- `encodingAlgorithm`: Specifies which encoding algorithm is used for storing the timestamps. If the specified
  compression algorithm is lossy, this is defaulted to None. Its values are:
  - `None`
  - `Masking`
  - `LeadingZeroes`
- `loopFindingAlgorithm`: Specifies what loop-finding algorithm is used. Its values are:
  - `None`
  - `Basic`
  - `BasicTruncated`
  - `Filter`

Here are the configuration options with number values:
- `zstdCompressionLevel`: Specifies the compression level used by ZSTD. Integer.
- `maxLoopLength`: Specifies the maximum loop length, if using a truncated loop finding algorithm. Integer.

## Contributing

Contribution to Pallas are welcome. Just send us a pull request.

## License
Pallas is distributed under the terms of both the BSD 3-Clause license.

See [LICENSE](LICENSE) for details

## Citing Pallas

Feel free to use the following publications to reference Pallas:

- ["Pallas: a generic trace format for large HPC trace analysis". IPDPS 2025](https://inria.hal.science/hal-04970114)

```
@inproceedings{guelque:hal-04970114,
  TITLE = {{Pallas: a generic trace format for large HPC trace analysis}},
  AUTHOR = {Guelque, Catherine and Honor{\'e}, Valentin and Swartvagher, Philippe and Thomas, Ga{\"e}l and Trahay, Fran{\c c}ois},
  URL = {https://inria.hal.science/hal-04970114},
  BOOKTITLE = {{IPDPS 2025: 39th IEEE International Parallel \& Distributed Processing Symposium}},
  ADDRESS = {Milan, Italy},
  YEAR = {2025},
  MONTH = Jun,
}
```
