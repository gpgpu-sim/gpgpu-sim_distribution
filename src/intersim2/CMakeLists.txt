option(GPGPUSIM_INTERSIM_STANDALONE "Whether to also build intersim in standalone mode" OFF)

# Specify Flex and Bison target
BISON_TARGET(intersim_config_parser config.y ${CMAKE_CURRENT_BINARY_DIR}/y.tab.c
            COMPILE_FLAGS "-y -d --file-prefix=${CMAKE_CURRENT_BINARY_DIR}/y")
FLEX_TARGET(intersim_config_lexer config.l ${CMAKE_CURRENT_BINARY_DIR}/lex.yy.c)
ADD_FLEX_BISON_DEPENDENCY(intersim_config_lexer intersim_config_parser)

# Set generated source files to CXX
set_source_files_properties(${BISON_intersim_config_parser_OUTPUT_SOURCE} 
                            ${FLEX_intersim_config_lexer_OUTPUTS}
                            PROPERTIES LANGUAGE CXX)

# Create booksim or libintersim.a
# Shared include path
list(APPEND intersim_INC ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/allocators
    ${CMAKE_CURRENT_SOURCE_DIR}/arbiters
    ${CMAKE_CURRENT_SOURCE_DIR}/networks
    ${CMAKE_CURRENT_SOURCE_DIR}/power
    ${CMAKE_CURRENT_SOURCE_DIR}/routers
    ${PROJECT_SOURCE_DIR}/src)

# Shared source files
list(APPEND intersim_SRC
        ${BISON_intersim_config_parser_OUTPUT_SOURCE}
        ${FLEX_intersim_config_lexer_OUTPUTS}
        allocators/allocator.cpp
        allocators/islip.cpp
        allocators/loa.cpp
        allocators/maxsize.cpp
        allocators/pim.cpp
        allocators/selalloc.cpp
        allocators/separable.cpp
        allocators/separable_input_first.cpp
        allocators/separable_output_first.cpp
        allocators/wavefront.cpp
        arbiters/arbiter.cpp
        arbiters/matrix_arb.cpp
        arbiters/prio_arb.cpp
        arbiters/roundrobin_arb.cpp
        arbiters/tree_arb.cpp
        batchtrafficmanager.cpp
        booksim_config.cpp
        buffer.cpp
        buffer_state.cpp
        config_utils.cpp
        credit.cpp
        flitchannel.cpp
        flit.cpp
        gputrafficmanager.cpp
        injection.cpp
        interconnect_interface.cpp
        intersim_config.cpp
        main.cpp
        misc_utils.cpp
        module.cpp
        networks/anynet.cpp
        networks/cmesh.cpp
        networks/dragonfly.cpp
        networks/fattree.cpp
        networks/flatfly_onchip.cpp
        networks/fly.cpp
        networks/kncube.cpp
        networks/network.cpp
        networks/qtree.cpp
        networks/tree4.cpp
        outputset.cpp
        packet_reply_info.cpp
        power/buffer_monitor.cpp
        power/power_module.cpp
        power/switch_monitor.cpp
        rng_double_wrapper.cpp
        rng_wrapper.cpp
        routefunc.cpp
        routers/chaos_router.cpp
        routers/event_router.cpp
        routers/iq_router.cpp
        routers/router.cpp
        stats.cpp
        traffic.cpp
        trafficmanager.cpp
        vc.cpp)

# If standalone, also build for it
if(GPGPUSIM_INTERSIM_STANDALONE)
    list(REMOVE_ITEM ${intersim_SRC} interconnect_interface.cpp)
    add_executable(booksim ${intersim_SRC})
    target_include_directories(booksim PUBLIC 
        ${intersim_INC})
    target_include_directories(booksim PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_include_directories(booksim PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    # Remove globally set TRACING_ON flag
    target_compile_options(booksim PRIVATE -UTRACING_ON)
endif()

# Specify sources for libintersim.a
add_library(intersim STATIC ${intersim_SRC})
target_include_directories(intersim PUBLIC 
    ${intersim_INC}
    ${PROJECT_SOURCE_DIR}/src/gpgpu-sim)
target_include_directories(intersim PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(intersim PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(intersim PRIVATE CREATE_LIBRARY)
# Remove globally set TRACING_ON flag
target_compile_options(intersim PRIVATE -UTRACING_ON)
