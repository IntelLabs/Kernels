# Copyright 2015 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
include ../../common/LEGION.defs
ifneq ($(LEGIONTOP),)
LG_RT_DIR=$(LEGIONTOP)/runtime
endif

#CXX=$(LEGIONCC)
CXX=g++

#Flags for directing the runtime makefile what to include
DEBUG           ?= 0		# Include debugging symbols
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time print level
SHARED_LOWLEVEL ?= 0		# Use shared-memory runtime (not recommended)
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

# Put the binary file name here
OUTFILE		:= transpose 
# List all the application source files here
GEN_SRC		:= transpose.cc # .cc files
GEN_GPU_SRC	:=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=
CC_FLAGS	?=
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?=

###########################################################################
#
#   Don't change anything below here
#
###########################################################################

usage:
	@echo "Usage: type \"make $(OUTFILE)\" to build executable"
	@echo "            \"make clean\" to remove objects and executables"

ifdef LG_RT_DIR
include $(LG_RT_DIR)/runtime.mk
$(GEN_OBJS): clean
else
clean:
endif

Transpose.o:
	$(CXX) -o transpose.o -c transpose.cc $(CC_FLAGS) $(INC_FLAGS) -O3
