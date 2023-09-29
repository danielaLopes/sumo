#!/bin/bash

FOLDER_EXP5=experiment5
mkdir experiment5

# execute SUMo
if [ ! -d ./$FOLDER_EXP5/samples_subsetsum2d ]
then
	cd sumo_pipeline/session_correlation
	bash ./testPerformanceOnGPU.sh
	mv ./samples_subsetsum2d ../../$FOLDER_EXP5
	cd ../../
fi

# execute DeepCoFFEA
if [ ! -d ./$FOLDER_EXP5/samples_deepcoffea ]
then
	cd dl_comparisons
	mkdir samples_deepcoffea
	for i in $(seq 10)
	do
		touch samples_deepcoffea/sample_deepcoffea_s$i
		echo " === DeepCoFFEA RUN=$i === "
		python3 performance.py deepcoffea $pairs >> samples_deepcoffea/sample_deepcoffea_s$i
	done
	mv ./samples_deepcoffea ../$FOLDER_EXP5
	cd ../
fi

