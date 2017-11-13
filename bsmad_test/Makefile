all: bsmad_test.cu
	nvcc --gpu-architecture=compute_50 --gpu-code=compute_50 -lcudart -g -o bsmad bsmad_test.cu

.PHONY:
clean:
	rm bsmad
