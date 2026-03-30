# dimlpfidex
Discretized Interpretable Multi Layer Perceptron (DIMLP) and related algorithms

### How to build

1. Pull submodules dependencies
```shell
$ git submodule init
$ git submodule update
```

2. Build with [docker CLI](https://docs.docker.com/reference/cli/docker/) (from project root):
```shell
docker build . -t compile:latest
docker run -u $(id -u):$(id -g) -v ./:/app compile:latest
```

### How to run

- From binaries (built inside the `bin` folder)
```shell
# example with fidexGloRules, from root project
./bin/fidexGloRules
```

- Using python bindings:
```shell
# install dependencies
python -m venv .venv
source .venv/bin/activate
pip install .
# install the file ending with .whl inside the `dist` directory
pip install dist/[FILENAME].whl
```

```py
# import example
from dimpfidex import dimlp
...
```

To download the required dependencies on your system, run:


### Install Python dependencies

#### Using pip

```shell
python -m venv .venv
source .venv/bin/activate
pip install .
```

#### Add dependencies

To add new dependencies to the project, add them to the `pyproject.toml` file.
To add them to the virtualenv, use:

```shell
pip install .
```

### Compile

1. Build binaries
```shell
mkdir build && cd build
cmake ..
cmake --build .
```

> [!NOTE]
> On Windows, you may have to use `cmake -DCMAKE_PREFIX_PATH="C:\<absolute\path\to>\.venv" ..` instead.

> [!TIP]
> To speed up the compilation process, you can also add `-j X` with `X` being your number of CPU cores.

> [!WARNING]
> If you need to rebuild the project, you must erase the content of the `build/` directory.

2. Build pip package

```shell
python -m build
```

## Documentation

Install [Doxygen](https://www.doxygen.nl/):

* **Linux, macOS, Windows/WSL**: Use your package manager to install `doxygen`
* **Windows**: `winget install DimitriVanHeesch.Doxygen`

Create the documentation:

```shell
mkdir build && cd build
cmake -DBUILD_DOCUMENTATION=ON ..
cmake --build .
```

The generated HTML documentation will be found in `build/docs/sphinx`.

## Credits
Our test suite is using [Obesity or CVD risk dataset](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster) from [AravindPCoder](https://www.kaggle.com/aravindpcoder) (under CC BY-SA 4.0 license)
