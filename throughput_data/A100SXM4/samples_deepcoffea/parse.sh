#!/bin/bash

for f in sample_deepcoffea_s*
do
	cat $f | sed -n 's/pairs: \(.*\), .*/\1/p' > col1_$f
	cat $f | sed -n 's/time: \(.*\)s/\1/p' > col2_$f
	echo "$(seq $(wc -l < col2_$f))" > idx_$f
	echo -e "idx\tNbPairs\tTime" > parsed_$f
	paste idx_$f col1_$f col2_$f >> parsed_$f
	rm idx_$f col1_$f col2_$f
done

