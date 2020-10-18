Welcome to GPGPU-Sim, a cycle-level simulator modeling contemporary graphics
processing units (GPUs) running GPU computing workloads written in CUDA or
OpenCL. Also included in GPGPU-Sim is a performance visualization tool called
AerialVision and a configurable and extensible energy model called GPUWattch.
GPGPU-Sim and GPUWattch have been rigorously validated with performance and
power measurements of real hardware GPUs.

This version of GPGPU-Sim has been tested with a subset of CUDA version 4.2,
5.0, 5.5, 6.0, 7.5, 8.0, 9.0, 9.1, 10, and 11

Please see the copyright notice in the file COPYRIGHT distributed with this
release in the same directory as this file.

If you use GPGPU-Sim 4.0 in your research, please cite:

Mahmoud Khairy, Zhesheng Shen, Tor M. Aamodt, Timothy G Rogers.
Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling.
In proceedings of the 47th IEEE/ACM International Symposium on Computer Architecture (ISCA),
May 29 - June 3, 2020.

If you use CuDNN or PyTorch support, checkpointing or our new debugging tool for functional 
simulation errors in GPGPU-Sim for your research, please cite:

Jonathan Lew, Deval Shah, Suchita Pati, Shaylin Cattell, Mengchi Zhang, Amruth Sandhupatla, 
Christopher Ng, Negar Goli, Matthew D. Sinclair, Timothy G. Rogers, Tor M. Aamodt
Analyzing Machine Learning Workloads Using a Detailed GPU Simulator, arXiv:1811.08933,
https://arxiv.org/abs/1811.08933


If you use the Tensor Core model in GPGPU-Sim or GPGPU-Sim's CUTLASS Library 
for your research please cite:

Md Aamir Raihan, Negar Goli, Tor Aamodt,
Modeling Deep Learning Accelerator Enabled GPUs, arXiv:1811.08309, 
https://arxiv.org/abs/1811.08309

If you use the GPUWattch energy model in your research, please cite:

Jingwen Leng, Tayler Hetherington, Ahmed ElTantawy, Syed Gilani, Nam Sung Kim,
Tor M. Aamodt, Vijay Janapa Reddi, GPUWattch: Enabling Energy Optimizations in
GPGPUs, In proceedings of the ACM/IEEE International Symposium on Computer
Architecture (ISCA 2013), Tel-Aviv, Israel, June 23-27, 2013.

If you use the support for CUDA dynamic parallelism in your research, please cite:

Jin Wang and Sudhakar Yalamanchili, Characterization and Analysis of Dynamic 
Parallelism in Unstructured GPU Applications, 2014 IEEE International Symposium 
on Workload Characterization (IISWC), November 2014.

If you use figures plotted using AerialVision in your publications, please cite:

Aaron Ariel, Wilson W. L. Fung, Andrew Turner, Tor M. Aamodt, Visualizing
Complex Dynamics in Many-Core Accelerator Architectures, In Proceedings of the
IEEE International Symposium on Performance Analysis of Systems and Software
(ISPASS), pp. 164-174, White Plains, NY, March 28-30, 2010.

This file contains instructions on installing, building and running GPGPU-Sim.
Detailed documentation on what GPGPU-Sim models, how to configure it, and a
guide to the source code can be found here: <http://gpgpu-sim.org/manual/>.
Instructions for building doxygen source code documentation are included below.
Detailed documentation on GPUWattch including how to configure it and a guide
to the source code can be found here: <http://gpgpu-sim.org/gpuwattch/>.

If you have questions, please sign up for the google groups page (see
gpgpu-sim.org), but note that use of this simulator does not imply any level of
support. Questions answered on a best effort basis.

To submit a bug report, go here: http://www.gpgpu-sim.org/bugs/

See Section 2 "INSTALLING, BUILDING and RUNNING GPGPU-Sim" below to get started.

See file CHANGES for updates in this and earlier versions.

# CONTRIBUTIONS and HISTORY

## GPGPU-Sim

GPGPU-Sim was created by Tor Aamodt's research group at the University of
British Columbia. Many have directly contributed to development of GPGPU-Sim
including: Tor Aamodt, Wilson W.L. Fung, Ali Bakhoda, George Yuan, Ivan Sham,
Henry Wong, Henry Tran, Andrew Turner, Aaron Ariel, Inderpret Singh, Tim
Rogers, Jimmy Kwa, Andrew Boktor, Ayub Gubran Tayler Hetherington and others.

GPGPU-Sim models the features of a modern graphics processor that are relevant
to non-graphics applications. The first version of GPGPU-Sim was used in a
MICRO'07 paper and follow-on ACM TACO paper on dynamic warp formation. That
version of GPGPU-Sim used the SimpleScalar PISA instruction set for functional
simulation, and various configuration files indicating which loops should be
spawned as kernels on the GPU, along with reconvergence points required for
SIMT execution to provide a programming model simlar to CUDA/OpenCL. Creating
benchmarks for the original GPGPU-Sim simulator was a very time consuming
process and the validity of code generation for CPU run on a GPU was questioned
by some. These issues motivated the development an interface for directly
running CUDA applications to leverage the growing number of applications being
developed to use CUDA. We subsequently added support for OpenCL and removed
all SimpleScalar code.

The interconnection network is simulated using the booksim simulator developed
by Bill Dally's research group at Stanford.

To produce output that matches the output from running the same CUDA program on
the GPU, we have implemented several PTX instructions using the CUDA Math
library (part of the CUDA toolkit). Code to interface with the CUDA Math
library is contained in cuda-math.h, which also includes several structures
derived from vector_types.h (one of the CUDA header files).

## GPUWattch Energy Model

GPUWattch (introduced in GPGPU-Sim 3.2.0) was developed by researchers at the
University of British Columbia, the University of Texas at Austin, and the
University of Wisconsin-Madison. Contributors to GPUWattch include Tor
Aamodt's research group at the University of British Columbia: Tayler
Hetherington and Ahmed ElTantawy; Vijay Reddi's research group at the
University of Texas at Austin: Jingwen Leng; and Nam Sung Kim's research group
at the University of Wisconsin-Madison: Syed Gilani.

GPUWattch leverages McPAT, which was developed by Sheng Li et al. at the
University of Notre Dame, Hewlett-Packard Labs, Seoul National University, and
the University of California, San Diego. The paper can be found at
http://www.hpl.hp.com/research/mcpat/micro09.pdf.

# INSTALLING, BUILDING and RUNNING GPGPU-Sim

Assuming all dependencies required by GPGPU-Sim are installed on your system,
to build GPGPU-Sim all you need to do is add the following line to your
~/.bashrc file (assuming the CUDA Toolkit was installed in /usr/local/cuda):

```
  export CUDA_INSTALL_PATH=/usr/local/cuda
```

then type

```
  bash
  source setup_environment
  make
```

If the above fails, see "Step 1" and "Step 2" below.

If the above worked, see "Step 3" below, which explains how to run a CUDA
benchmark on GPGPU-Sim.

## Step 1: Dependencies

GPGPU-Sim was developed on SUSE Linux (this release was tested with SUSE
version 11.3) and has been used on several other Linux platforms (both 32-bit
and 64-bit systems). In principle, GPGPU-Sim should work with any linux
distribution as long as the following software dependencies are satisfied.

Download and install the CUDA Toolkit. It is recommended to use version 3.1 for
normal PTX simulation and version 4.0 for cuobjdump support and/or to use
PTXPlus (Harware instruction set support). Note that it is possible to have
multiple versions of the CUDA toolkit installed on a single system -- just
install them in different directories and set your CUDA_INSTALL_PATH
environment variable to point to the version you want to use.

[Optional] If you want to run OpenCL on the simulator, download and install
NVIDIA's OpenCL driver from <http://developer.nvidia.com/opencl>. Update your
PATH and LD_LIBRARY_PATH as indicated by the NVIDIA install scripts. Note that
you will need to use the lib64 directory if you are using a 64-bit machine. We
have tested OpenCL on GPGPU-Sim using NVIDIA driver version 256.40
<http://developer.download.nvidia.com/compute/cuda/3_1/drivers/devdriver_3.1_linux_64_256.40.run>
This version of GPGPU-Sim has been updated to support more recent versions of
the NVIDIA drivers (tested on version 295.20).

GPGPU-Sim dependencies:
- gcc
- g++
- make
- makedepend
- xutils
- bison
- flex
- zlib
- CUDA Toolkit

GPGPU-Sim documentation dependencies:
- doxygen
- graphvi

AerialVision dependencies:
- python-pmw
- python-ply
- python-numpy
- libpng12-dev
- python-matplotlib

We used gcc/g++ version 4.5.1, bison version 2.4.1, and flex version 2.5.35.

If you are using Ubuntu, the following commands will install all required
dependencies besides the CUDA Toolkit.

GPGPU-Sim dependencies:

	sudo apt-get install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev

GPGPU-Sim documentation dependencies:

	sudo apt-get install doxygen graphviz

AerialVision dependencies:

	sudo apt-get install python-pmw python-ply python-numpy libpng12-dev python-matplotlib

CUDA SDK dependencies:

	sudo apt-get install libxi-dev libxmu-dev libglut3-dev

If you are running applications which use NVIDIA libraries such as cuDNN and 
cuBLAS, install them too.

Finally, ensure CUDA_INSTALL_PATH is set to the location where you installed
the CUDA Toolkit (e.g., /usr/local/cuda) and that \$CUDA_INSTALL_PATH/bin is in
your PATH. You probably want to modify your .bashrc file to incude the
following (this assumes the CUDA Toolkit was installed in /usr/local/cuda):

	export CUDA_INSTALL_PATH=/usr/local/cuda
	export PATH=$CUDA_INSTALL_PATH/bin

If running applications which use cuDNN or cuBLAS:

	export CUDNN_PATH=<Path To cuDNN Directory>
	export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$CUDA_INSTALL_PATH/lib:$CUDNN_PATH/lib64

	

## Step 2: Build

To build the simulator, you first need to configure how you want it to be
built. From the root directory of the simulator, type the following commands in
a bash shell (you can check you are using a bash shell by running the command
"echo \$SHELL", which should print "/bin/bash"):

source setup_environment <build_type>

replace <build_type> with debug or release. Use release if you need faster
simulation and debug if you need to run the simulator in gdb. If nothing is
specified, release will be used by default.

Now you are ready to build the simulator, just run

	make


After make is done, the simulator would be ready to use. To clean the build,
run

	make clean

To build the doxygen generated documentations, run

	make docs

To clean the docs run

	make cleandocs


The documentation resides at doc/doxygen/html.

To run Pytorch applications with the simulator, install the modified Pytorch library as well by following instructions [here](https://github.com/gpgpu-sim/pytorch-gpgpu-sim).
## Step 3: Run

Before we run, we need to make sure the application's executable file is dynamically linked to CUDA runtime library. This can be done during compilation of your program by introducing the nvcc flag "--cudart shared" in makefile (quotes should be excluded).

To confirm the same, type the follwoing command:

`ldd <your_application_name>`

You should see that your application is using libcudart.so file in GPGPUSim directory. If the application is a Pytorch application, `<your_application_name>` should be `$PYTORCH_BIN`, which should be set during the Pytorch installation.

If running applications which use cuDNN or cuBLAS:

* Modify the Makefile or the compilation command of the application to change 
   all the dynamic links to static ones, for example:
	* `-L$(CUDA_PATH)/lib64 -lcublas` to
	  `-L$(CUDA_PATH)/lib64 -lcublas_static`

	* `-L$(CUDNN_PATH)/lib64 -lcudnn` to
	  `-L$(CUDNN_PATH)/lib64 -lcudnn_static`

* Modify the Makefile or the compilation command such that the following 
   flags are used by the nvcc compiler:
	`-gencode arch=compute_61,code=compute_61`

   (the number 61 refers to the SM version. You would need to set it based 
   on the GPGPU-Sim config `-gpgpu-ptx-force-max-capability` you use)

Copy the contents of configs/QuadroFX5800/ or configs/GTX480/ to your
application's working directory. These files configure the microarchitecture
models to resemble the respective GPGPU architectures.

To use ptxplus (native ISA) change the following options in the configuration
file to "1" (Note: you need CUDA version 4.0) as follows:

	-gpgpu_ptx_use_cuobjdump 1
	-gpgpu_ptx_convert_to_ptxplus 1

Now To run a CUDA application on the simulator, simply execute

	source setup_environment <build_type>

Use the same <build_type> you used while building the simulator. Then just
launch the executable as you would if it was to run on the hardware. By
running `source setup_environment <build_type>` you change your LD_LIBRARY_PATH
to point to GPGPU-Sim's instead of CUDA or OpenCL runtime so that you do NOT
need to re-compile your application simply to run it on GPGPU-Sim.

To revert back to running on the hardware, remove GPGPU-Sim from your
LD_LIBRARY_PATH environment variable.

The following GPGPU-Sim configuration options are used to enable GPUWattch

	-power_simulation_enabled 1 (1=Enabled, 0=Not enabled)
	-gpuwattch_xml_file <filename>.xml


The GPUWattch XML configuration file name is set to gpuwattch.xml by default and
currently only supplied for GTX480 (default=gpuwattch_gtx480.xml). Please refer to
<http://gpgpu-sim.org/gpuwattch/> for more information.

Running OpenCL applications is identical to running CUDA applications. However,
OpenCL applications need to communicate with the NVIDIA driver in order to
build OpenCL at runtime. GPGPU-Sim supports offloading this compilation to a
remote machine. The hostname of this machine can be specified using the
environment variable OPENCL_REMOTE_GPU_HOST. This variable should also be set
through the setup_environment script. If you are offloading to a remote machine,
you might want to setup passwordless ssh login to that machine in order to
avoid having too retype your password for every execution of an OpenCL
application.

If you need to run the set of applications in the NVIDIA CUDA SDK code
samples then you will need to download, install and build the SDK.

The CUDA applications from the ISPASS 2009 paper mentioned above are
distributed separately on github under the repo ispass2009-benchmarks.
The README.ISPASS-2009 file distributed with the benchmarks now contains
updated instructions for running the benchmarks on GPGPU-Sim v3.x.

# (OPTIONAL) Contributing to GPGPU-Sim (ADVANCED USERS ONLY)

If you have made modifications to the simulator and wish to incorporate new
features/bugfixes from subsequent releases the following instructions may help.
They are meant only as a starting point and only recommended for users
comfortable with using source control who have experience modifying and
debugging GPGPU-Sim.

WARNING: Before following the procedure below, back up your modifications to
GPGPU-Sim. The following procedure may cause you to lose all your changes. In
general, merging code changes can require manual intervention and even in the
case where a merge proceeds automatically it may introduce errors. If many
edits have been made the merge process can be a painful manual process. Hence,
you will almost certainly want to have a copy of your code as it existed before
you followed the procedure below in case you need to start over again. You
will need to consult the documentation for git in addition to these
instructions in the case of any complications.

STOP. BACK UP YOUR CHANGES BEFORE PROCEEDING. YOU HAVE BEEN WARNED. TWICE.

To update GPGPU-Sim you need git to be installed on your system. Below we
assume that you ran the following command to get the source code of GPGPU-Sim:

```
  git clone git://dev.ece.ubc.ca/gpgpu-sim
```

Since running the above command you have made local changes and we have
published changes to GPGPU-Sim on the above git server. You have looked at the
changes we made, looking at both the new CHANGES file and probably even the
source code differences. You decide you want to incorporate our changes into
your modified version of GPGPU-Sim.

Before updating your source code, we recommend you remove any object files:

```
  make clean
```

Then, run the following command in the root directory of GPGPU-Sim:

```
  git pull
```

While git is pulling the latest changes, conflicts might arise due to changes
that you made that conflict with the latest updates. In this case, you need to
resolved those conflicts manually. You can either edit the conflicting files
directly using your favorite text editor, or you can use the following command
to open a graphical merge tool to do the merge:

```
  git mergetool
```

## Testing updated version of GPGPU-Sim

Now you should test that the merged version "works". This means following the
steps for building GPGPU-Sim in the _new_ README file (not this version) since
they may have changed. Assuming the code compiles without errors/warnings the
next step is to do some regression testing. At UBC we have an extensive set of
regression tests we run against our internal development branch when we make
changes. In the future we may make this set of regression tests publically
available. For now, you will want to compile the merged code and re-run all of
the applications you care about (implying these applications worked for you
before you did the merge). You want to do this before making further changes to
identify any compile time or runtime errors that occur due to the code merging
process.


# MISCELLANEOUS

## Speeding up the execution

Some applications take several hours to execute on GPGPUSim. This is because the simulator has to dump the PTX, analyze them and get resource usage statistics. This can be avoided everytime we execute the program in the following way:

1. Execute the program by enabling “-save_embedded_ptx 1” in config file, execute the code and let cuobjdump command dump all necessary files. After this process, you will get 2 new files namely:  _cuobjdump_complete_output_<some_random_name> and _1.ptx

2. Create new environment variables or include the below in your .bashrc file:
	1. export PTX_SIM_USE_PTX_FILE=_1.ptx
	2. export PTX_SIM_KERNELFILE=_1.ptx
	3. export CUOBJDUMP_SIM_FILE=_cuobjdump_complete_output_<some_random_name>

3. Disable -save_embedded_ptx flag, execute the code again. This will skip the dumping by cuobjdump and directly goes to executing the program thus saving time.


## Debugging failing GPGPU-Sim Regressions
 
Credits: Tor M Aamodt

To debug failing GPGPU-Sim regression tests you need to run them locally.  The fastest way to do this, assuming you are working with GPGPU-Sim versions more recent than the GPGPU-Sim dev branch circa March 28, 2018 (commit hash 2221d208a745a098a60b0d24c05007e92aaba092), is to install Docker.  The instructions below were tested with Docker CE version 18.03 on Ubuntu and Mac OS.  Docker will enable you to run the same set of regressions used by GPGPU-Sim when submitting a pull request to https://github.com/gpgpu-sim/gpgpu-sim_distribution and also allow you to log in and launch GPGPU-Sim in gdb so you can inspect failures.  

1. Install Docker.  On Ubuntu 14.04 and 16.04 the following instructions work:  https://docs.docker.com/install/linux/docker-ce/ubuntu/#uninstall-old-versions 

2. Clone GPGPU-Sim from your fork of GPGPU-Sim. For example:

	git clone https://github.com/<YOUR GITHUB USERNAME>/gpgpu-sim_distribution.git


3. Run the following command (this is all one line) to run the regressions in docker:
	```
	docker run --privileged -v `pwd`:/home/runner/gpgpu-sim_distribution:rw aamodt/gpgpu-sim_regress:latest /bin/bash -c "./start_torque.sh; chown -R runner /home/runner/gpgpu-sim_distribution; su - runner -c 'source /home/runner/gpgpu-sim_distribution/setup_environment && make -j -C /home/runner/gpgpu-sim_distribution && cd /home/runner/gpgpu-sim_simulations/ && git pull && /home/runner/gpgpu-sim_simulations/util/job_launching/run_simulations.py -c /home/runner/gpgpu-sim_simulations/util/job_launching/regression_recipies/rodinia_2.0-ft/configs.gtx1080ti.yml -N regress && /home/runner/gpgpu-sim_simulations/util/job_launching/monitor_func_test.py -v -N regress'; tail -f /dev/null"
	```
	Explanation: The last part of this command, "tail -f /dev/null" will keep the docker container running after the regressions finish.  This enables you to log into the container to run the same tests inside gdb so you can debug.   The "--privileged" part enables you to use breakpoints inside gdb in a container.  The "-v" part maps the current directory (with the GPGPU-Sim source code you want to test) into the container. The string "aamodt/gpgpu-sim_regress:latest" is a tag for a container setup to run regressions which will be downloaded from docker hub.  The portion starting with /bin/bash is a set of commands run inside a bash shell inside the container.  E.g., the command start_torque.sh starts up a queue manager inside the container.  

	If the above command stops with the message "fatal: unable to access 'https://github.com/tgrogers/gpgpu-sim_simulations.git/': Could not resolve host: github.com" this likely means your computer sits behind a firewall which is blocking access to Google's name servers (e.g., 8.8.8.8).  To get around this you will need to modify th above command to point to your local DNS server.  Lookup your DNS server IP address which we will call <DNS_IP_ADDRESS> below.  On Ubuntu run "ifconfig" to lookup the network interface connecting your computer to the network.  Then run "nmcli device show <interface name>" to find the IP address of your DNS server.  Modify the above command to include "--dns <DNS_IP_ADDRESS>" after "run", E.g.,
	```
	docker run --dns <DNS_IP_ADDRESS> --privileged -v `pwd`:/home/runner/gpgpu-sim_distribution:rw aamodt/gpgpu-sim_regress:latest /bin/bash -c "./start_torque.sh; chown -R runner /home/runner/gpgpu-sim_distribution; su - runner -c 'source /home/runner/gpgpu-sim_distribution/setup_environment && make -j -C /home/runner/gpgpu-sim_distribution && cd /home/runner/gpgpu-sim_simulations/ && git pull && /home/runner/gpgpu-sim_simulations/util/job_launching/run_simulations.py -c /home/runner/gpgpu-sim_simulations/util/job_launching/regression_recipies/rodinia_2.0-ft/configs.gtx1080ti.yml -N regress && /home/runner/gpgpu-sim_simulations/util/job_launching/monitor_func_test.py -v -N regress'; tail -f /dev/null"
	```

4. Find the CONTAINER ID associated with your docker container by running "docker ps". 

5. Log into the container by running the command:
	```
	docker exec -it <CONTAINER_ID> /bin/bash -c "su -l runner"`
	```
	The container is running Ubuntu 16.04 and has screen, cscope and vim installed (if you find a favorite Linux tool missing, it is fairly easy to create derived containers that have additional tools).

6. Lookup the directory of the regression test you want to debug by going to the regression log file directory:
	```
	cd /home/runner/gpgpu-sim_simulations/util/job_launching/logfiles
	```

7.  The file "failed_job_log_sim_log.regress.<DATE>.txt" includes information about the failed test including its simulation directory.  For the following example, I'll assume the first failing test was "hotspot-rodinia-2.0-ft-30_6_40___data_result_30_6_40_txt--GTX1080Ti" for which the simulation directory is /home/runner/gpgpu-sim_simulations/util/job_launching/../../sim_run_4.2/hotspot-rodinia-2.0-ft/30_6_40___data_result_30_6_40_txt/GTX1080Ti/

8.  Change to the simulation directory using:
	```
	cd <simulation_directory>
	```
	E.g., `cd /home/runner/gpgpu-sim_simulations/util/job_launching/../../sim_run_4.2/hotspot-rodinia-2.0-ft/30_6_40___data_result_30_6_40_txt/GTX1080Ti/`

	This directory should contain a file called "torque.sim" that contains commands used to launch the simulation during regression tests.  We will modify this file to enable us to re-run the regression test in gdb.   This directory should also contain a file containing the standard output during the regression test.  This file will end in .o<number> where <number> is the torque queue manager job number.  For the running example for me this file is called "hotspot-rodinia-2.0-ft-30_6_40___data_result_30_6_40_txt.o2".  Open this file to determine the LD_LIBRARY_PATH settings used when launching the simulation.  Look for a line that starts "doing: export LD_LIBRARY_PATH" and copy the entire line starting with "export LD_LIBRARY_PATH ..."

9. Paste the "export LD_LIBRARY_PATH ..." line into the bash shell to set LD_LIBRARY_PATH.  E.g.,
	```
	export LD_LIBRARY_PATH=/home/runner/gpgpu-sim_simulations/util/job_launching/../../sim_run_4.2/gpgpu-sim-builds/libcudart_gpgpu-sim_git-commit-177d02254ae38b6331b17dd6cd139b570a03c589_modified_0.so:/gpgpu-sim/usr/local/gcc-4.5.4/lib64:/gpgpu-sim/usr/local/gcc-4.5.4/lib:/gpgpu-sim/usr/local/gcc-4.5.4/lib/gcc/x86_64-unknown-linux-gnu/lib64/:/gpgpu-sim/usr/local/gcc-4.5.4/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/:/usr/lib/x86_64-linux-gnu:/home/runner/gpgpu-sim_distribution/lib/gcc-4.5.4/cuda-4020/release:/gpgpu-sim/usr/local/gcc-4.5.4/lib64:/gpgpu-sim/usr/local/gcc-4.5.4/lib:/gpgpu-sim/usr/local/gcc-4.5.4/lib/gcc/x86_64-unknown-linux-gnu/lib64/:/gpgpu-sim/usr/local/gcc-4.5.4/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/:/usr/lib/x86_64-linux-gnu:
	```

10. In the same shell, build the debug version of GPGPU-Sim then return to the directory above:
	```
	pushd ~/gpgpu-sim_distribution/
	source setup_environment debug
	make
	popd
	```

11. Open and edit torque.sim and preface the very last line with "gdb --args ".  After editing the last line in torque.sim should look something like:
	```
	gdb --args /home/runner/gpgpu-sim_simulations/util/job_launching/../../benchmarks/bin/4.2/release/hotspot-rodinia-2.0-ft 30 6 40 ./data/result_30_6_40.txt
	```

12. Re-run the regression test in gdb by sourcing the torque.sim file:
	```
	. torque.sim
	```
	This will put you in at the (gdb) prompt.  Setup any breakpoints needed and run.  

