## Building Pallas
You need to have ZSTD installed before you try to build this.
To build and install, simply run:
```bash
git clone https://gitlab.inria.fr/pallas/pallas.git
mkdir -p pallas/build && cd pallas/build
cmake ..
cmake --install . --config Release
```

## Building and using EZTrace with Pallas
If you want to trace calls to a library, we recommend using [EZTrace](https://eztrace.gitlab.io/eztrace/).
Make sure to build it using Pallas as a backend !
```bash
mkdir build && cd build
otf2-config --help  # Check that you have the correct otf2-config, i.e. the Pallas one
cmake .. -DEZTRACE_ENABLE_<your_module>=ON
make && make install
```

You can then trace your application as per the EZTrace documentation says.
```bash
eztrace -t "<your modules>" ./my-app
```

## In your application
You can also skip EZTrace and trace your application yourself. 
For this, you should get a full grasp of Pallas by reading the [full documentation](02-pallas/).

# Reading your trace
## Using Pallas
You can read a Pallas trace by using `pallas_info` or `pallas_print`.
The first one will print the content of a trace, while the second one will print all the events in an orderly fashion.

## Using Python
Pallas comes with a Python library to read your traces.
You need to enable building it with `-DENABLE_PYTHON=ON`

Its requirements are the following:
- Python >=3.11
- Numpy
- pybind11

You can then read your trace like this:
```python
import pallas_trace as pallas
trace = pallas.open_trace("<trace_name>.pallas")
```

You can run your own instance of a Jupyter-Notebook with Pallas already installed 
and some examples provided my doing the following steps (require Docker):
```bash
docker run -p 8888:8888 -w /jupyter-notebook -it registry.gitlab.inria.fr/pallas/pallas/eztrace_example:latest  jupyter-notebook --allow-root --ip=0.0.0.0 --no-browser
```

## Using C/C++
You can also use the Pallas library to read your traces. 
Once again, you should get a full grasp of Pallas by reading the [full documentation](02-pallas/index.md).

## Visualizing Pallas traces

[Blup](https://gitlab.inria.fr/blup/blup) is a web-based trace visualizer able to display Pallas traces. 
It uses the Pallas Python API.

![](https://gitlab.inria.fr/blup/blup/-/raw/main/doc/screenshot.png)