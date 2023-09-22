#!/bin/bash

for f in sample_subsetsum_s*
do
	cat $f | sed -n 's/=* REPEAT \(.*\) =*/\1/p' > col1_$f
	cat $f | sed -n 's/Nb pairs: \(.*\)/\1/p' > col2_$f
	cat $f | sed -n 's/total_subsetsum_time \(.*\)/\1/p' > col3_$f
	echo -e "idx\tNbPairs\tTime" > parsed_$f
	paste col1_$f col2_$f col3_$f >> parsed_$f
	rm col1_$f col2_$f col3_$f
done

