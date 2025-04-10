# Pallas
[![BSD-3 License](https://img.shields.io/badge/License-BSD3-yellow.svg)](https://opensource.org/license/bsd-3-clause)
![Dev Pipeline](https://gitlab.inria.fr/pallas/pallas/badges/dev/pipeline.svg?=test)


Pallas provides an interface to write and read trace data.

## Building
You need to have ZSTD installed before you try to build this.
To build and install, simply run:
```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALLATION_DIR
make && make install
```

This will build and install `pallas_info` and `pallas_print`,
which are tools made to read Pallas traces, as well as the Pallas library, and a modified OTF2 library.

If you want to enable SZ and ZFP, you should install them, and then add `-DSZ_ROOT_DIR=<your SZ installation>`
and `-DZFP_ROOT_DIR=<your ZFP installation>` to the cmake command line. Documentation is built automatically if Doxygen is installed.

## Python Library
Pallas comes with a Python library to read your traces.
You need to enable building it with `-DENABLE_PYTHON=ON`

Its requirements are the following:
- Python >=3.11
- Numpy
- pybind11

You can then read it like this:
```
$ export PYTHONPATH=PATH_TO_PALLAS/lib
$ python
>>> import pallas
>>> trace=pallas.open_trace("eztrace_log.pallas")
...

```

Make sure these are installed (locally using a venv or globally) before building.
## Usage
### In your application
To use Pallas to log your application, you need to understand the hierarchical structure:
- There's one *GlobalArchive* that stores global information.
- Each *Archive* corresponds to a process / a self-contained information group.
They're identified by LocationGroup
- Each *Thread* refers to a group of events that are to be logged.
They're identified by Location

There is a bit more nuance, but it's what you need for now.
The following code should give you an idea of what to do to start logging your app in C++.
There's also a C API that functions pretty much the same way, as you'd expect it to.
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

A config file can be given to Pallas with the PALLAS_CONFIG_PATH environment variable.
If that variable is empty, Pallas will try to load a pallas.config in the current directory.
If that file does not exist, a default config will be loaded.
That file is a mirror image of the ParameterHandler class in `libraries/pallas/src/ParameterHandler.h`
An example config file is given here as pallas.config, each line has one `key=value` pair.

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

You can also override each of these configuration manually with an environment variable.
Here are the default values for each of them:

| Config Key Name      | Env Variable Name   | Default Value  |
|----------------------|---------------------|----------------|
| compressionAlgorithm | PALLAS_COMPRESSION  | None           |
| encodingAlgorithm    | PALLAS_ENCODING     | None           |
| loopFindingAlgorithm | PALLAS_LOOP_FINDING | BasicTruncated |
| zstdCompressionLevel | PALLAS_ZSTD_LVL     | 3              |
| maxLoopLength        | PALLAS_LOOP_LENGTH  | 100            |

## Contributing

Contribution to Pallas are welcome. Just send us a pull request.

## License
Pallas is distributed under the terms of both the BSD 3-Clause license.

See [LICENSE](LICENSE) for details
