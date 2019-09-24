NVCC=nvcc -ccbin=`which gcc` -D_FORCE_INLINES
NVBIT_PATH=../../core
INCLUDES=-I$(NVBIT_PATH)
LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))
SOURCES=$(wildcard *.cu)
OBJECTS=$(SOURCES:.cu=.o)
ARCH=35

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

all: $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=sm_$(ARCH) -O3 *.o $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o ${current_dir}.so

%.o: %.cu
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -maxrregcount=16 -Xcompiler -Wall -arch=sm_$(ARCH) -O3 -Xcompiler -fPIC $< -o $@

$(NVBIT_PATH)/libnvbit.a:
	make -C $(NVBIT_PATH)

clean:
	rm -f *.so *.o
