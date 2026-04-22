# dimlpfidex

Discretized Interpretable Multi Layer Perceptron (DIMLP) and related algorithms

Authors: Jean Marc Boutay, Guido Bologna & Damian Boquete

---

## How to build

First of all, pull submodules dependencies
```sh
git submodule init
git submodule update
```

### Option 1: via docker (easiest)

Build with [docker CLI](https://docs.docker.com/reference/cli/docker/) (from project root):
```sh
docker build . -t compile:latest
docker run -u $(id -u):$(id -g) -v ./:/app compile:latest
```

### Option 2 : from sources

1. Build binaries
```sh
mkdir build && cd build
cmake ..
cmake --build .
```

> [!NOTE]
> On Windows, you may have to use `cmake -DCMAKE_PREFIX_PATH="C:\<absolute\path\to>\.venv" ..` instead.

> [!WARNING]
> If you need to rebuild the project, you must erase the content of the `build/` directory.

2. Build pip package

```sh
python -m build
```

This will generate the `wheel` package into the `dist` folder. Later usable by installing it via `pip`.

## How to run

### From binaries (built inside the `bin` folder)
```sh
# example with fidexGloRules, from root project
./bin/fidexGloRules
```

### Using python bindings:

1. install Python dependencies
```sh
# install dependencies
python -m venv .venv
source .venv/bin/activate
pip install .
# install the file ending with .whl inside the `dist` directory
pip install dist/[FILENAME].whl
```

2. Import a dimlpfidex library to your code like so:
```py
# import example
from dimlpfidex import dimlp
...
```

## How to generate documentation

1. Install [Doxygen](https://www.doxygen.nl/).

2. You'll probably have to install [Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html) too. 

3. Generate the documentation following these commands:

```sh
mkdir build && cd build
cmake -DBUILD_DOCUMENTATION=ON ..
cmake --build .
```

4. If the build process is successful, You should find the `index.html` page inside `build/docs/sphinx`. 


## Credits
Our test suite is using [Obesity or CVD risk dataset](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster) from [AravindPCoder](https://www.kaggle.com/aravindpcoder) (under CC BY-SA 4.0 license)
