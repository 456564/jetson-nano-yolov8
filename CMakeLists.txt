cmake_minimum_required(VERSION 3.10)
project(nano_yolo_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 53)
if(NOT DEFINED CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "--generate-code arch=compute_53,code=sm_53")
endif()

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} src)

file(GLOB_RECURSE CUDA_SOURCES src/*.cu)
file(GLOB_RECURSE CPP_SOURCES src/*.cpp)

list(FILTER CPP_SOURCES EXCLUDE REGEX "src/ncnn/*")
list(FILTER CPP_SOURCES EXCLUDE REGEX "src/test_infer.cpp")
list(FILTER CPP_SOURCES EXCLUDE REGEX "src/detect.cpp")

add_executable(detect ${CPP_SOURCES} ${CUDA_SOURCES})

target_link_libraries(detect ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvinfer OpenMP::OpenMP_CXX)
