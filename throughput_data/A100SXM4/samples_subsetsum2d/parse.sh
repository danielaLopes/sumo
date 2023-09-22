#!/bin/bash

for f in sample_subsetsum2d_s*
do
	# cat $f | sed -n 's/=* REPEAT \(.*\) =*/\1/p' > col1_$f
	cat $f | sed -n 's/^.*nPairs=\([0-9]*\).*$/\1/p' > col2_$f
	seq 0 $(( $(wc -l "col2_$f" | cut -f 1 -d " ") - 1)) > col1_$f

	cat $f | sed -n 's/Exec time subsetsum2d:\(.*\)s/\1/p' > col3_$f

	echo -e "idx\tNbPairs\tTime" > parsed_$f
	paste col1_$f col2_$f col3_$f >> parsed_$f
	rm col1_$f col2_$f col3_$f
done

