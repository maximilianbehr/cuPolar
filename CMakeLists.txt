# MIT License
#
# Copyright (c) 2024 Maximilian Behr
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.23)
project(cupolar LANGUAGES C CUDA VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# check cuda version
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER_TOOLKIT_VERSION VERSION_GREATER_EQUAL "11.4.2")
    message(STATUS "CUDA version ${CMAKE_CUDA_COMPILER_TOOLKIT_VERSION} is supported.")
else()
    message(FATAL_ERROR "CUDA version ${CMAKE_CUDA_COMPILER_TOOLKIT_VERSION} is not supported. Please install CUDA version 11.4.2 or higher.")
endif()

# cupolar library
add_library(cupolar SHARED cupolar_newton.cu cupolar_halley.cu)
set_property(TARGET cupolar PROPERTY CUDA_ARCHITECTURES all)
set_target_properties(cupolar PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cupolar PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(cupolar PUBLIC cublas cusolver)
target_compile_options(cupolar PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
install(TARGETS cupolar DESTINATION lib)
install(FILES cupolar.h DESTINATION include)

# examples
foreach(x s d c z)
  # Newton
  add_executable(example_cupolar_${x}Newton example_cupolar_${x}Newton.cu)
  target_link_libraries(example_cupolar_${x}Newton PUBLIC cupolar)
  set_property(TARGET example_cupolar_${x}Newton PROPERTY CUDA_ARCHITECTURES all)

  # Halley
  add_executable(example_cupolar_${x}Halley example_cupolar_${x}Halley.cu)
  target_link_libraries(example_cupolar_${x}Halley PUBLIC cupolar)
  set_property(TARGET example_cupolar_${x}Halley PROPERTY CUDA_ARCHITECTURES all)
endforeach()
