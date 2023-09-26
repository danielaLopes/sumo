#!/bin/bash

cp ../throughput_data/plot.py .
cd ../sumo_pipeline/session_correlation
bash ./testPerformanceOnGPU.sh
mv ./samples_subsetsum2d ../../experiments
cd ../../experiments

cd samples_subsetsum2d
for f in sample_subsetsum2d_s*
do
	cat $f | sed -n 's/=* REPEAT \([0-9]*\) =*/\1/p' > col1_$f
	cat $f | sed -n 's/Nb pairs: \(.*\)/\1/p' > col2_$f
	cat $f | sed -n 's/Exec time (s): \(.*\)/\1/p' > col3_$f
	echo -e "idx\tNbPairs\tTime" > parsed_$f
	paste col1_$f col2_$f col3_$f >> parsed_$f
	rm col1_$f col2_$f col3_$f
done
cd ..

python3 plot.py "subsetsum2d" "." 10

