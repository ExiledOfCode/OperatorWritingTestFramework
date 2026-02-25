cmake_minimum_required(VERSION 3.22)
project(cuda_op_library LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80) # A100 = sm80

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()
find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED)
# ===== CUTLASS 本地路径 =====
set(CUTLASS_PATH ${PROJECT_SOURCE_DIR}/cutlass)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${CUTLASS_PATH}/include
  ${CUTLASS_PATH}/tools/util/include
)

file(GLOB_RECURSE SRC_FILES
  src/*.cu
  src/*.cpp
  src/hand_gemm/*.cu
)

file(GLOB_RECURSE TEST_FILES
  tests/*.cu
  tests/*.cpp
)

add_executable(test_runner
  ${SRC_FILES}
  ${TEST_FILES}
)

target_link_libraries(test_runner
    PRIVATE
    GTest::gtest
    GTest::gtest_main
    pthread
    CUDA::cublas
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(test_runner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

set_target_properties(test_runner PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME AllTests COMMAND test_runner)

# ===== GPU 属性打印工具 =====
add_executable(gpu_props
  tool/gpu_props.cu
)

set_target_properties(gpu_props PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
