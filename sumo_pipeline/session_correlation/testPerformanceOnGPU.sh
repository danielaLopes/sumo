
make torpedosubsetsumopencl2d_TEST
mkdir -p samples_subsetsum2d

cat <<EOF > EXP_NB_PAIRS
8821
37073
76001
155803
319397
654763
1342265
2751645
5640872
11563788
23705766
48596822
99623485
EOF

for i in $(seq 5)
do
  j=0
  touch samples_subsetsum2d/sample_subsetsum2d_s$i
  for pairs in $(cat ./EXP_NB_PAIRS)
  do
    j=$(($j + 1))
    echo " === SUMo RUN=$i PAIRS=$pairs === "
    echo " === REPEAT $j === " >> samples_subsetsum2d/sample_subsetsum2d_s$i
    ./torpedosubsetsumopencl2d_TEST $pairs >> samples_subsetsum2d/sample_subsetsum2d_s$i
  done
done