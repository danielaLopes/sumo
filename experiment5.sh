#!/bin/bash

FOLDER_EXP5=experiment5
mkdir experiment5

# execute SUMo
cp throughput_data/plot.py ./$FOLDER_EXP5
cd sumo_pipeline/session_correlation
# bash ./testPerformanceOnGPU.sh
mv ./samples_subsetsum2d ../../$FOLDER_EXP5
cd ../../

# execute DeepCoFFEA
cd dl_comparisons
mkdir samples_deepcoffea
for i in $(seq 10)
do
  touch samples_deepcoffea/sample_deepcoffea_s$i
  echo " === DeepCoFFEA RUN=$i PAIRS=$pairs === "
  python3 performance.py deepcoffea $pairs >> samples_deepcoffea/sample_deepcoffea_s$i
done
mv ./samples_deepcoffea ../$FOLDER_EXP5
cd ..

cd $FOLDER_EXP5

cd samples_subsetsum2d
for f in sample_subsetsum2d_s*
do
	cat $f | sed -n 's/Nb pairs: \(.*\)/\1/p' > col2_$f
	cat $f | sed -n 's/Exec time (s): \(.*\)/\1/p' > col3_$f
	WC_COUNT=$(wc -l col3_$f| cut -f 1 -d " ")
  seq $WC_COUNT > col1_$f
	echo -e "idx\tNbPairs\tTime" > parsed_$f
	paste col1_$f col2_$f col3_$f >> parsed_$f
	rm col1_$f col2_$f col3_$f
done
cd ..

cd samples_deepcoffea
for f in sample_deepcoffea_s*
do
	cat $f | sed -n 's/pairs: \([0-9]*\),.*/\1/p' > col2_$f
	cat $f | sed -n 's/time: \(.*\)s/\1/p' > col3_$f
  WC_COUNT=$(wc -l col3_$f | cut -f 1 -d " ")
  seq $WC_COUNT > col1_$f
	echo -e "idx\tNbPairs\tTime" > parsed_$f
	paste col1_$f col2_$f col3_$f >> parsed_$f
	rm col1_$f col2_$f col3_$f
done
cd ..

python3 plot.py
