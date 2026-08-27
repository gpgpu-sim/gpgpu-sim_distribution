# Need to create a setup script to set some variables for others to interact with
set(SETUP_SCRIPT_FILENAME "setup")
message(STATUS "Writing setup commands to '${SETUP_SCRIPT_FILENAME}'")
file(WRITE ${SETUP_SCRIPT_FILENAME} "export GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN=1\n")
file(APPEND ${SETUP_SCRIPT_FILENAME} "export GPGPUSIM_ROOT=${PROJECT_SOURCE_DIR}\n")
file(APPEND ${SETUP_SCRIPT_FILENAME} "export GPGPUSIM_CONFIG=${GPGPUSIM_CONFIG}\n")
file(APPEND ${SETUP_SCRIPT_FILENAME} "export CUDA_INSTALL_PATH=${CUDAToolkit_TARGET_DIR}\n")
file(APPEND ${SETUP_SCRIPT_FILENAME} "export PATH=`echo $PATH | sed 's#$GPGPUSIM_ROOT/bin:$CUDA_INSTALL_PATH/bin:##'`\n")
file(APPEND ${SETUP_SCRIPT_FILENAME} "export PATH=$GPGPUSIM_ROOT/bin:$CUDA_INSTALL_PATH/bin:$PATH\n")
file(APPEND ${SETUP_SCRIPT_FILENAME} "export CUDA_VERSION_NUMBER=${CUDA_VERSION_NUMBER}\n")
if(CUDA_VERSION_NUMBER GREATER_EQUAL 6000)
    file(APPEND ${SETUP_SCRIPT_FILENAME} "export PTX_SIM_USE_PTX_FILE=1.ptx\n")
    file(APPEND ${SETUP_SCRIPT_FILENAME} "export PTX_SIM_KERNELFILE=_1.ptx\n")
    file(APPEND ${SETUP_SCRIPT_FILENAME} "export CUOBJDUMP_SIM_FILE=jj\n")
endif()
# TODO What about OpenCL support?

# setting LD_LIBRARY_PATH as follows enables GPGPU-Sim to be invoked by 
# native CUDA and OpenCL applications. GPGPU-Sim is dynamically linked
# against instead of the CUDA toolkit.  This replaces this cumbersome
# static link setup in prior GPGPU-Sim releases.
# Create a softlink for backward support
if(APPLE)
file(APPEND ${SETUP_SCRIPT_FILENAME} "export DYLD_LIBRARY_PATH=`echo $DYLD_LIBRARY_PATH | sed -Ee 's#'$GPGPUSIM_ROOT'\/lib\/[0-9]+\/(debug|release):##'`\n")
file(APPEND ${SETUP_SCRIPT_FILENAME} "export DYLD_LIBRARY_PATH=$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG:$DYLD_LIBRARY_PATH\n")
else()
file(APPEND ${SETUP_SCRIPT_FILENAME} "export LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | sed -re 's#'$GPGPUSIM_ROOT'\/lib\/[0-9]+\/(debug|release):##'`\n")
file(APPEND ${SETUP_SCRIPT_FILENAME} "export LD_LIBRARY_PATH=$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG:$LD_LIBRARY_PATH\n")
endif()

# TODO ignore the OPENCL_REMOTE_GPU_HOST part?