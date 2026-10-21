set(GPGPUSIM_ACCELWATTCH_NTHREADS "4" CACHE STRING "Accelwattch MCPAT thread count")
add_library(accelwattch STATIC 
            cacti/Ucache.cc
            XML_Parse.cc
            cacti/arbiter.cc
            cacti/area.cc
            array.cc
            cacti/bank.cc
            cacti/basic_circuit.cc
            basic_components.cc
            cacti/cacti_interface.cc
            cacti/component.cc
            core.cc
            cacti/crossbar.cc
            cacti/decoder.cc
            cacti/htree2.cc
            interconnect.cc
            cacti/io.cc
            iocontrollers.cc
            logic.cc
            main.cc
            cacti/mat.cc
            memoryctrl.cc
            noc.cc
            cacti/nuca.cc
            cacti/parameter.cc
            processor.cc
            cacti/router.cc
            sharedcache.cc
            cacti/subarray.cc
            cacti/technology.cc
            cacti/uca.cc
            cacti/wire.cc
            xmlParser.cc
            gpgpu_sim_wrapper.cc)
target_include_directories(accelwattch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(accelwattch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cacti)
# Compile options
target_compile_options(accelwattch PRIVATE "-Wno-unknown-pragmas")
if($<CONFIG:Debug>)
    target_compile_definitions(NTHREADS=1)
else()
    target_compile_options(accelwattch PRIVATE "-msse2;-mfpmath=sse")
    target_compile_definitions(accelwattch PRIVATE -DNTHREADS=${GPGPUSIM_ACCELWATTCH_NTHREADS})
endif()
target_link_options(accelwattch PRIVATE "-lm;-lpthread;-lz")