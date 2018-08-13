all: tensorcore_type32_32.cu
	nvcc --gpu-architecture=compute_70 --gpu-code=compute_70 -lcudart -g -o tensor_core tensorcore_type32_32.cu
#	nvcc -arch=sm_70 -lcudart -g -o tensor_core tensor_core.cu

.PHONY:
clean:
	rm _cuob*
	rm gpgpusim_power* 
	rm gpgpu_inst_stats.txt
#	nvcc -arch=sm_70 --gpu-architecture=compute_50 --gpu-code=compute_50 -lcudart -g -o tensor_core tensor_core.cu
