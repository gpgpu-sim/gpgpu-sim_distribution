all: tensor_core.cu
	nvcc -arch=sm_70 -lcudart -g -o tensor_core tensor_core.cu

.PHONY:
clean:
	rm tensorcore 
#	nvcc -arch=sm_70 --gpu-architecture=compute_50 --gpu-code=compute_50 -lcudart -g -o tensor_core tensor_core.cu
