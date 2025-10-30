You can build Pallas on your own using CMake.

## Requirements
The following software are **required**:
- git
- cmake
- C++ compiler >=17
- pkg-config
- ZSTD (compression library used by Pallas)

To install all the required packages on Debian, you can run:
```bash
sudo apt install build-essential cmake git libzstd-dev pkg-config
```

### Optional requirements
If you wish, you can also build the Python API of Pallas.
Doing so requires having [PyBind11](https://pybind11.readthedocs.io/en/stable/).
```bash
sudo apt install python3 python3-pip python3-pybind11
```
Pallas traces are compressed with ZSTD by default, but you can add [SZ](https://github.com/szcompressor/SZ3)
and [ZFP](https://zfp.readthedocs.io/en/release0.5.5/installation.html) compression as well.
Be aware that those are lossy compressions, and may well cause issues in the trace coherence.

Pallas can also be built with MPI, OpenMP and PThreads tests.
```bash
sudo apt install mpich libomp-dev
```

# Building Pallas
Just as any CMake library, you can start with
```bash
git clone https://gitlab.inria.fr/pallas/pallas
mkdir -p pallas/build && cd pallas/build
cmake ..
cmake --install . --config Release # You will probably need sudo to run it like that
```

Pallas has a few CMake variables you can edit:

| Name                      | Description                                                                   | Values   |
|---------------------------|-------------------------------------------------------------------------------|----------|
| `ENABLE_OTF2`             | Build the OTF2 compatibility library. Enable this if you want to use EZTrace. | ON / OFF |
| `BUILD_DOC`               | Build the doxygen documentation. Requires the Doxygen library.                | ON / OFF |
| `ENABLE_SZ`               | Build Pallas with SZ support.                                                 | ON / OFF |
| `ENABLE_ZFP`              | Build Pallas with ZFP support.                                                | ON / OFF |
| `ENABLE_PYTHON`           | Build the Pallas Python library.                                              | ON / OFF |
| `Python3_INSTALL_LOCALLY` | Install the Pallas Python library for the user instead of system-wide.        | ON / OFF |

Pallas also has a few compile-time macros which you can configure in `libraries/pallas/include/pallas/pallas_config.h.in`.
These are mostly related to the initial size of arrays used when writing Pallas traces.

Pallas also has a run-time configuration file. You can configure the default file in `libraries/pallas/pallas.config`.
Each entry in that configuration file can be overwritten by the environment variable of the same name.
You can also give your own custom configuration file with the `PALLAS_CONFIG_PATH` environment variable.

## Building EZTrace with Pallas support
Building EZTrace with Pallas should be no different from building EZTrace with OTF2.
There are two crucial steps:
- Make sure you built and installed Pallas with the `ENABLE_OTF2=ON` CMake build flag.
- Make sure Pallas is loaded in your path when configuring CMake for EZTrace. You can check that with `otf2-config | grep Pallas`.