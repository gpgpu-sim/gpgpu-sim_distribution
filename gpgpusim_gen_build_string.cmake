# Get hash
execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${INPUT_DIR}
    OUTPUT_VARIABLE GPGPUSIM_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get diff
execute_process(
    COMMAND git diff --numstat
    COMMAND wc
    COMMAND sed -re "s/^\\s+([0-9]+).*/\\1./"
    WORKING_DIRECTORY ${INPUT_DIR}
    OUTPUT_VARIABLE GPGPUSIM_GIT_DIFF
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND git diff --numstat --staged
    COMMAND wc
    COMMAND sed -re "s/^\\s+([0-9]+).*/\\1./"
    WORKING_DIRECTORY ${INPUT_DIR}
    OUTPUT_VARIABLE GPGPUSIM_GIT_DIFF_STAGED
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(GPGPUSIM_BUILD_STRING "gpgpu-sim_git-commit-${GPGPUSIM_GIT_HASH}_modified_${GPGPUSIM_GIT_DIFF}${GPGPUSIM_GIT_DIFF_STAGED}")
configure_file(${INPUT_DIR}/version.in ${OUTPUT_DIR}/detailed_version)
