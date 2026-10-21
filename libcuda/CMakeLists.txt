# Specify Flex and Bison target
BISON_TARGET(cuobjdump_parser cuobjdump.y ${CMAKE_CURRENT_BINARY_DIR}/cuobjdump_parser.c
            COMPILE_FLAGS "-t -d -v --report=all -p cuobjdump_ --file-prefix=${CMAKE_CURRENT_BINARY_DIR}/cuobjdump")
FLEX_TARGET(cuobjdump_lexer cuobjdump.l ${CMAKE_CURRENT_BINARY_DIR}/cuobjdump_lexer.c
            COMPILE_FLAGS "-B -P cuobjdump_")
ADD_FLEX_BISON_DEPENDENCY(cuobjdump_lexer cuobjdump_parser)

# Set generated source files to CXX
set_source_files_properties(${BISON_cuobjdump_parser_OUTPUT_SOURCE} 
                            ${FLEX_cuobjdump_lexer_OUTPUTS}
                            PROPERTIES LANGUAGE CXX)

# Create libcuda.a with all source files
add_library(cuda STATIC 
    cuda_runtime_api.cc
    ${BISON_cuobjdump_parser_OUTPUT_SOURCE} ${FLEX_cuobjdump_lexer_OUTPUTS})

    # Add current dir to include path
# Also add flex/bison generated header files
target_include_directories(cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# Add cuda include path for own reference
target_include_directories(cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Add project build dir to include path
target_include_directories(cuda PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
