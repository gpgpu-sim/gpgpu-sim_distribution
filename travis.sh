if [ ! -n "$CUDA_INSTALL_PATH" ]; then
	echo "ERROR ** Install CUDA Toolkit and set CUDA_INSTALL_PATH.";
	exit;
fi

if [ ! -n "$CONFIG" ]; then
	echo "ERROR ** set the CONFIG env variable to one of those found in ./accel-sim-framework/util/job_launching/configs/define-standard-cfgs.yml";
	exit;
fi

if [ ! -n "$GPUAPPS_ROOT" ]; then
	echo "ERROR ** GPUAPPS_ROOT to a location where the apps have been compiled";
	exit;
fi

export PATH=$CUDA_INSTALL_PATH/bin:$PATH
source ./setup_environment
make -j

pip install psutil
rm -rf accel-sim-framework
git clone https://github.com/accel-sim/accel-sim-framework.git
./accel-sim-framework/util/job_launching/run_simulations.py -C $CONFIG -B rodinia_2.0-ft -N regress -l local
./accel-sim-framework/util/job_launching/monitor_func_test.py -v -N regress -j procman
