installdir := "/usr/local"

default:
    just --list

[private]
setup_builddir:
    mkdir -p build

[private]
[working-directory('build')]
make:
    make -j

[private]
[working-directory('build')]
make_install: make
    sudo make install

[working-directory('build')]
build: setup_builddir && make
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="{{ installdir }}"

install: build && make_install

[working-directory('build')]
dev: setup_builddir && make_install
    cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX="{{ installdir }}"

[working-directory('build')]
uninstall:
    xargs rm < install_manifest.txt

install_python:
    pip install .

# Install the python library by linking it dynamically to the cpp library
[working-directory('build')]
dev_python: setup_builddir && make_install
    cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX="{{ installdir }}" -DINSTALL_PYTHON=ON

clean:
    rm -rf build
