if [ ! -n "$CUDA_INSTALL_PATH" ]; then
	echo "ERROR ** Install CUDA Toolkit and set CUDA_INSTALL_PATH.";
	exit 1;
fi

if [ ! -n "$CONFIG" ]; then
	echo "ERROR ** set the CONFIG env variable to one of those found in ./accel-sim-framework/util/job_launching/configs/define-standard-cfgs.yml";
	exit 1;
fi

if [ ! -n "$GPUAPPS_ROOT" ]; then
	echo "ERROR ** GPUAPPS_ROOT to a location where the apps have been compiled";
	exit 1;
fi

# Set default value for APP
if [ ! -n "$APP" ]; then
	APP=rodinia_2.0-ft
fi
echo "Running simulation for $APP with config $CONFIG"

git config --system --add safe.directory '*'

export PATH=$CUDA_INSTALL_PATH/bin:$PATH

cmake -B build
cmake --build build -j
cmake --install build
source setup

git clone https://github.com/accel-sim/accel-sim-framework.git
./accel-sim-framework/util/job_launching/run_simulations.py -C $CONFIG -B $APP -N regress -l local
./accel-sim-framework/util/job_launching/monitor_func_test.py -v -N regress -j procman
