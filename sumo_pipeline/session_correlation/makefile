INC  := -I /usr/local/cuda/include
LIBF := -L /usr/lib/nvidia-current
LIBS := -lOpenCL -lgomp
FLAGS := -fopenmp 
LINKER_FLAGS := -fPIC --shared 
OPTS := -O3 -std=c++14
CXX  := g++

# forces recompilation in case of changes in the makefile
subsetSum_opencl_2d.cpp: makefile
	touch -a -m $@
subsetSum_opencl.cpp: makefile
	touch -a -m $@
	

torpedosubsetsumopencl_TEST: subsetSum_opencl.cpp
	$(CXX) $(INC) $(LIBF) $(FLAGS) $(OPTS) $^ -o $@ $(LIBS) 

torpedosubsetsumopencl2d_TEST: subsetSum_opencl_2d.cpp
	$(CXX) $(INC) $(LIBF) $(FLAGS) $(OPTS) $^ -o $@ $(LIBS) 

torpedosubsetsumopencl.so: subsetSum_opencl.cpp
	$(CXX) $(INC) $(LIBF) $(FLAG) $(LINKER_FLAGS) $(OPTS) $^ -o $@ $(LIBS) 

torpedosubsetsumopencl2d.so: subsetSum_opencl_2d.cpp
	$(CXX) $(INC) $(LIBF) $(FLAG) $(LINKER_FLAGS) $(OPTS) $^ -o $@ $(LIBS) 

clean:
	rm -fr torpedosubsetsumopencl_TEST torpedosubsetsumopencl.so