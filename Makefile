# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
SHELL := /bin/bash
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
PY3_VER ?= $(shell python3 -c "import sys;print('%d%d' % sys.version_info[:2])")
# Allowed CPU values: k8, armv7a, aarch64
CPU ?= k8
# Allowed COMPILATION_MODE values: opt, dbg
COMPILATION_MODE ?= opt

BAZEL_OUT_DIR :=  $(MAKEFILE_DIR)/bazel-out/$(CPU)-$(COMPILATION_MODE)/bin
BAZEL_BUILD_FLAGS := --crosstool_top=@crosstool//:toolchains \
                     --compilation_mode=$(COMPILATION_MODE) \
                     --copt=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
                     --verbose_failures \
                     --sandbox_debug \
                     --subcommands \
                     --define PY3_VER=$(PY3_VER) \
                     --linkopt=-L$(MAKEFILE_DIR)/libedgetpu/direct/$(CPU) \
                     --linkopt=-l:libedgetpu.so.1.0 \
                     --compiler=gcc \
                     --cpu=$(CPU)

ifeq ("$(COMPILATION_MODE)", "opt")
BAZEL_BUILD_FLAGS += --linkopt=-Wl,--strip-all
endif

ifeq ("$(CPU)", "k8")
ifeq ("$(wildcard /usr/include/glibc_compat.h)","")
GLIBC_COMPAT_PATH := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))docker/glibc_compat.h
$(shell sudo cp -f ${GLIBC_COMPAT_PATH} /usr/include)
endif
BAZEL_BUILD_FLAGS += --copt=-includeglibc_compat.h
else ifeq ("$(CPU)", "aarch64")
BAZEL_BUILD_FLAGS += --copt=-ffp-contract=off
else ifeq ("$(CPU)", "armv7a")
BAZEL_BUILD_FLAGS += --copt=-ffp-contract=off
else
$(error Unknown value $(CPU) of CPU variable)
endif

# $(1): pattern, $(2) destination directory
define copy_out_files =
	for f in `find $(BAZEL_OUT_DIR) -name $(1) -type f -printf '%P\n'`; do \
		mkdir -p $(2)/`dirname $$f`; \
		cp -f $(BAZEL_OUT_DIR)/$$f $(2)/$$f; \
	done
endef

EXAMPLES_OUT_DIR    := $(MAKEFILE_DIR)/out/$(CPU)/examples

.PHONY: all \
        examples \
        help

all: examples 

examples:
	bazel build $(BAZEL_BUILD_FLAGS) //src/cpp/examples:minimal

	mkdir -p $(EXAMPLES_OUT_DIR)
	cp -f $(BAZEL_OUT_DIR)/src/cpp/examples/minimal \
	      $(EXAMPLES_OUT_DIR)
clean:
	rm -rf $(MAKEFILE_DIR)/bazel-* \
	       $(MAKEFILE_DIR)/out

DOCKER_WORKSPACE=$(MAKEFILE_DIR)
DOCKER_CPUS=k8 armv7a aarch64
DOCKER_TAG_BASE=coral-edgetpu
include $(MAKEFILE_DIR)/docker/docker.mk

deb:
	dpkg-buildpackage -rfakeroot -us -uc -tc -b

deb-armhf:
	dpkg-buildpackage -rfakeroot -us -uc -tc -b -a armhf -d

deb-arm64:
	dpkg-buildpackage -rfakeroot -us -uc -tc -b -a arm64 -d

wheel:
	python3 $(MAKEFILE_DIR)/setup.py bdist_wheel -d $(MAKEFILE_DIR)/dist

help:
	@echo "make all               - Build everything"
	@echo "make examples          - Build all examples"
	@echo "make help              - Print help message"

# Debugging util, print variable names. For example, `make print-ROOT_DIR`.
print-%:
	@echo $* = $($*)
