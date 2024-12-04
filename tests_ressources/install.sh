#! /bin/bash

git clone https://github.com/Jean-Marc-B/dimlpfidex_Hepia.git && \
cd dimlpfidex_Hepia && \
python -m venv .venv && \
source .venv/bin/activate && \
pip install . && \
rm -rf bin build && \
mkdir build && \
cd build && \
cmake .. && \
cmake --build . -j 8 && \
cd .. && \
python3 -m build . && \
pip uninstall dimlpfidex && \
pip install dist/*.whl
# add openpyxl and pandas