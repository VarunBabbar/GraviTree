# PyTorch-free extension for falling_trees. Needs pybind11 + Python development headers.

CXX ?= g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC -DFALLING_TREES_PYBIND

# Interpreter used for include paths and extension suffix (must have pybind11).
# Example: make PYTHON="$HOME/miniforge3/envs/foo/bin/python"
PYTHON ?= python

PYBIND_INCLUDES := $(shell $(PYTHON) -c "import sysconfig, pybind11; print('-I' + sysconfig.get_path('include') + ' -I' + pybind11.get_include())" 2>/dev/null)

EXTENSION_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; s=sysconfig.get_config_var('EXT_SUFFIX'); print(s if s else '.so')" 2>/dev/null)

ifeq ($(strip $(PYBIND_INCLUDES)),)
$(error Missing pybind11 for "$(PYTHON)". Run: $(PYTHON) -m pip install pybind11   Then retry, or: make PYTHON=/path/to/python-with-pybind11)
endif

ifeq ($(strip $(EXTENSION_SUFFIX)),)
EXTENSION_SUFFIX := .so
endif

SRC = gravitree.cpp
TARGET = falling_rashomon_cpp$(EXTENSION_SUFFIX)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND_INCLUDES) $< -o $@

clean:
	rm -f *.so

.PHONY: all clean
