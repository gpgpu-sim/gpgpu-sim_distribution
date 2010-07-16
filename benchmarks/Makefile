default:
	make -C CUDA/AES
	make -C CUDA/BFS
	make -C CUDA/BlackScholes
	cd ./CUDA/CP/; export PARBOIL_ROOT=`pwd`; cd common/src; make; cd -; ./parboil compile cp cuda_short
	make -C CUDA/DG/3rdParty/ParMetis-3.1
	make -C CUDA/DG
	make -C CUDA/LIB
	make -C CUDA/LPS
	make -C CUDA/MUM
	make -C CUDA/NN
	make -C CUDA/NQU
	make -C CUDA/RAY
	make -C CUDA/STO
	make -C CUDA/template
	make -C CUDA/WP

clean:
	make clean -C CUDA/AES
	make clean -C CUDA/BFS
	make clean -C CUDA/BlackScholes
	cd ./CUDA/CP/; ./parboil clean cp cuda_short	 
	make clean -C ./CUDA/DG/3rdParty/ParMetis-3.1
	make clean -C ./CUDA/DG
	make clean -C ./CUDA/LIB
	make clean -C ./CUDA/LPS
	make clean -C ./CUDA/MUM
	make clean -C ./CUDA/NN
	make clean -C ./CUDA/NQU
	make clean -C ./CUDA/RAY
	make clean -C ./CUDA/STO
	make clean -C ./CUDA/template
	make clean -C ./CUDA/WP

clean-visualizer:
	rm `find -name "gpgpusim_visualizer__*.log.gz"`
