#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets add_vectors.cu -lcudart -o add_vectors

prefetch:
	nvcc -g -DPREF -Wno-deprecated-gpu-targets add_vectors.cu -lcudart -o add_vectors

clean:
	rm -rf add_vectors gpgpusim_power_report* _cuobjdump_* *.txt
