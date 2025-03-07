cmake_minimum_required(VERSION 3.17)

# Project name and version
project(GPGPU-Sim 
        VERSION 4.2.0 
        DESCRIPTION "cycle-level simulator modeling contemporary graphics processing units (GPUs)" 
        HOMEPAGE_URL https://github.com/accel-sim/gpgpu-sim_distribution
        LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# GPGPU-Sim build option
option(GPGPUSIM_ENABLE_TRACE "Whether to enable GPGPU-Sim debug tracing" ON)

# GPGPU-Sim conditional build variable
set(GPGPUSIM_USE_POWER_MODEL OFF)
set(GPGPUSIM_USE_OPENCL OFF)

# Check for dependencies
include(gpgpusim_check.cmake)

# Create version file
add_custom_target(gen_build_string ALL 
                    COMMAND ${CMAKE_COMMAND} -D INPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR} -D OUTPUT_DIR=${CMAKE_BINARY_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/gpgpusim_gen_build_string.cmake
                    COMMENT "Generating build string file to ${CMAKE_CURRENT_BINARY_DIR}")

# CMake target
# GPGPU-Sim CUDA Runtime lib
# Use the entrypoint object files sources else CMake will complain
add_library(cudart SHARED $<TARGET_OBJECTS:gpgpusim_entrypoint>)
add_library(entrypoint STATIC $<TARGET_OBJECTS:gpgpusim_entrypoint>)

# Add global C/CXX compilation flags and definitions
# TODO Specify more build modes like gem5 with fast opt?
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_compile_definitions(DEBUG=1)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wno-unused-function;-Wno-sign-compare;-g;-fPIC>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Wall;-Wno-unused-function;-Wno-sign-compare;-ggdb;-fPIC>")
else()
    add_compile_definitions(DEBUG=0)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-O3;-g;-Wall;-Wno-unused-function;-Wno-sign-compare;-fPIC>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Wall;-Wno-unused-function;-Wno-sign-compare;-fPIC>")
endif()

# Add CUDA version
add_compile_definitions(CUDART_VERSION=${CUDA_VERSION_NUMBER})

# OpenCL support
if(GPGPUSIM_USE_OPENCL)
    add_compile_definitions(OPENGL_SUPPORT)
endif()

# Tracing support
if(GPGPUSIM_ENABLE_TRACE)
    add_compile_definitions(TRACING_ON=1)
endif()

# Add subdirectory
add_subdirectory(src)
add_subdirectory(libcuda)
add_subdirectory(libopencl)

# Set linker option for libcudart.so
if(APPLE)
    target_link_options(cudart PUBLIC "-Wl,-headerpad_max_install_names,-undefined,dynamic_lookup,-compatibility_version,1.1,-current_version,1.1;-lm;-lz;-pthread")
else()
    target_link_options(cudart PUBLIC
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/linux-so-version.txt;-lm;-lz;-lGL;-pthread")
        target_link_options(entrypoint PUBLIC
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/linux-so-version.txt;-lm;-lz;-lGL;-pthread")
endif()
# cuda: CUDA API lib
# ptxsim: cuda-sim, functional simulator
# gpgpusim: gpu simulator (gpgpu-sim)
# intersim: interconnect simulator
# accelwattch: power simulator
# Rest of source files in src/ will be created with gpgpusim_entrypoint target
target_link_libraries(cudart PUBLIC cuda ptxsim gpgpusim intersim)
target_link_libraries(entrypoint PUBLIC cuda ptxsim gpgpusim intersim)
if(GPGPUSIM_USE_POWER_MODEL)
target_link_libraries(cudart PUBLIC cuda ptxsim gpgpusim intersim accelwattch)
target_link_libraries(entrypoint PUBLIC cuda ptxsim gpgpusim intersim accelwattch)
endif()

# TODO Conditionally build for Opencl?
# if(GPGPUSIM_USE_OPENCL)
# add_library(OpenCL)
# endif()

# Install and post-install
# Get configure
set(GPGPUSIM_CONFIG "gcc-${CMAKE_CXX_COMPILER_VERSION}/cuda-${CUDA_VERSION_NUMBER}/${GPGPUSIM_BUILD_MODE}")

# Env var setup script
include(gpgpusim_gen_setup_environment.cmake)

# Installation
set(GPGPUSIM_INSTALL_PATH ${PROJECT_SOURCE_DIR}/lib/${GPGPUSIM_CONFIG})
install(TARGETS cudart DESTINATION ${GPGPUSIM_INSTALL_PATH})

# Installing symlinks
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.2\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.3\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.4\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.5.0\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.5.5\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.6.0\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.6.5\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.7.0\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.7.5\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.8.0\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.9.0\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.9.1\)")
    install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.9.2\)")
    install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.10.0\)")
    install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.10.1\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.11.0\)")
install(CODE "execute_process\(\
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart> \
    ${GPGPUSIM_INSTALL_PATH}/$<TARGET_FILE_NAME:cudart>.12\)")
