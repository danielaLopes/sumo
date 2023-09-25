# SUMo

This repository presents the code for the artefact evaluation for The Network and Distributed System Security Symposium (NDSS) 2024.

To run the experiements we need the extracted features from the dataset. Download them from here:
https://zenodo.org/record/8369700/files/extracted_features.tar.gz

Decompress with:
```gzip -d extracted_features.tar.gz && tar -xf extracted_features.tar```

The datasets are also available from:
 * [OSTrain.tar.gz](https://zenodo.org/record/8362616/files/OSTrain.tar.gz)
 * [OSValidate.tar.gz](https://zenodo.org/record/8360991/files/OSValidate.tar.gz)
 * [OSTest.tar.gz](https://zenodo.org/record/8359342/files/OSTest.tar.gz)

## Source separation



## Target separation



## Correlation



## Throughput/latency

To collect the latency/throughput metrics of SUMo follow the instructions in (throughput_sumo.sh)[./experiments/throughput_sumo.sh]. It should output a plot in `sumo/experiments/plot_subsetsum2d.pdf` with the latency/throughput curve of our solution.


### If you make use of our please cite our NDSS'24 paper:

"Flow Correlation Attacks on Tor Onion Service Sessions with Sliding Subset Sum". Daniela Lopes, Jin-Dong Dong, Daniel Castro, Pedro Medeiros, Diogo Barradas, Bernardo Portela, João Vinagre, Bernardo Ferreira, Nicolas Christin, and Nuno Santos. The Network and Distributed System Security Symposium (NDSS) 2024.
