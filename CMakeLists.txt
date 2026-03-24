cmake_minimum_required(VERSION 3.22)
project(cuda_op_library LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80-real 80-virtual)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

set(CUTLASS_PATH ${PROJECT_SOURCE_DIR}/third_party/cutlass)
set(OP_RUNNER_SOURCE ${PROJECT_SOURCE_DIR}/tests/op_runner.cpp)
set(OP_RUNTIME_SOURCE ${PROJECT_SOURCE_DIR}/src/framework/op_runtime.cpp)
set(OP_BINARY_DIR ${PROJECT_BINARY_DIR}/bin)

function(configure_cuda_common_target target_name)
  target_include_directories(${target_name}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CUTLASS_PATH}/include
    ${CUTLASS_PATH}/tools/util/include
  )

  target_link_libraries(${target_name}
    PRIVATE
    CUDA::cudart
    CUDA::cublas
  )

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Ncu")
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endif()

  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

function(configure_cuda_op_target target_name)
  configure_cuda_common_target(${target_name})
  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OP_BINARY_DIR})
endfunction()

add_library(opfw_common STATIC
  ${OP_RUNNER_SOURCE}
  ${OP_RUNTIME_SOURCE}
)
configure_cuda_common_target(opfw_common)

file(GLOB_RECURSE OP_CANDIDATES CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/src/*.cu
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

set(OP_TARGETS)
foreach(op_source IN LISTS OP_CANDIDATES)
  file(READ ${op_source} OP_SOURCE_CONTENT)
  if(NOT OP_SOURCE_CONTENT MATCHES "(REGISTER_OP(_FUNCS)?|LC_REGISTER_[A-Z0-9_]+)\\(")
    continue()
  endif()

  file(RELATIVE_PATH rel_path ${PROJECT_SOURCE_DIR}/src ${op_source})
  get_filename_component(op_basename ${op_source} NAME_WE)

  string(REPLACE "/" "__" target_suffix ${rel_path})
  string(REPLACE "." "_" target_suffix ${target_suffix})
  string(REPLACE "-" "_" target_suffix ${target_suffix})
  set(target_name op__${target_suffix})

  add_executable(${target_name}
    ${op_source}
  )
  configure_cuda_op_target(${target_name})
  target_link_libraries(${target_name} PRIVATE opfw_common)
  set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${op_basename})
  list(APPEND OP_TARGETS ${target_name})
endforeach()

if(NOT OP_TARGETS)
  message(FATAL_ERROR "No registered operator sources were found under src/.")
endif()

add_custom_target(build_all_ops DEPENDS ${OP_TARGETS})

add_executable(gpu_props
  tool/gpu_props.cu
)

target_include_directories(gpu_props
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(gpu_props
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY ${OP_BINARY_DIR}
)
