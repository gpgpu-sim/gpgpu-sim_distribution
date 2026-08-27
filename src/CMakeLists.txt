# gpgpusim_entrypoint objects
add_library(gpgpusim_entrypoint OBJECT
    abstract_hardware_model.cc
    debug.cc
    gpgpusim_entrypoint.cc
    option_parser.cc
    statwrapper.cc
    stream_manager.cc
    trace.cc
    utils.cc)

# Add current folder and CUDA include to include path
target_include_directories(gpgpusim_entrypoint PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpgpusim_entrypoint PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

# Add subdir
add_subdirectory(accelwattch)
add_subdirectory(cuda-sim)
add_subdirectory(gpgpu-sim)
add_subdirectory(intersim2)