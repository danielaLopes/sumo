#!/bin/bash

for f in OSreqData.txt OriginChecker.txt
do
	cat $f | sed -n 's/.*pairs \(.*\)/\1/p' > col1_$f
	cat $f | sed -n 's/.*TIMES \(.*\)/\1/p' > col2_$f
	echo "$(seq $(wc -l < col2_$f))" > idx_$f
	echo -e "idx\tNbPairs\tTime" > parsed_$f
	paste idx_$f col1_$f col2_$f >> parsed_$f
	rm idx_$f col1_$f col2_$f
done

