all: docs

docs:
	DOXYGEN_STRIP_PATH=`readlink -f ../../..` doxygen gpgpu-sim.doxygen

clean:
	rm -rf latex html
