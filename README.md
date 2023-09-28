# SUMo: Flow Correlation Attacks on Tor Onion Sessions using Sliding Subset Sum

This repository presents the code for the artefact evaluation for The Network and Distributed System Security Symposium (NDSS) 2024.

### If you make use of our work please cite our NDSS'24 paper:

"Flow Correlation Attacks on Tor Onion Service Sessions with Sliding Subset Sum". Daniela Lopes, Jin-Dong Dong, Daniel Castro, Pedro Medeiros, Diogo Barradas, Bernardo Portela, João Vinagre, Bernardo Ferreira, Nicolas Christin, and Nuno Santos. The Network and Distributed System Security Symposium (NDSS) 2024.

## Artifact Evaluation

We make available a set of scripts to run all the experiments that reproduce the main results of the paper. The whole set of experiments can be executed by running:
```
./experiment_all.sh
```
Alternatively, you can follow the following steps to individually execute each experiment:

#### Setup
Run the following script to install dependencies and compile the C code necessary for the following experiments:
```
./setup.sh
```

#### Experiment (E1) Session Matching with Perfect Filtering Phase
We expect this to take a maximum of 40 minutes. 
```
./experiment1.sh
./experiment1_results.sh
```

#### Experiment (E2) Session Matching with Partial Coverage
We expect this to take a maximum of 2.5 hours. 
```
./experiment2.sh
```

#### Experiment (E3) Session Matching with Imperfect Filtering Phase
We expect this to take a maximum of 1 hour. 
```
./experiment3_setup.sh
./experiment3.sh
./experiment3_results.sh
```

#### Experiment (E4) Comparison with the State-of-the-Art on Flow Correlation
We expect this to take a maximum of xxx. 
```
./experiment4_setup.sh
./experiment4.sh
./experiment4_results.sh
```

#### Experiment (E5) Throughput Evaluation
We expect this to take a maximum of 1 hour. 
```
./experiment5.sh
```





## Installation

To run the experiments we need the extracted features from the dataset. Download them from here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8369700.svg)](https://zenodo.org/record/8369700/files/extracted_features.tar.gz)

Decompress with:
```gzip -d extracted_features.tar.gz && tar -xf extracted_features.tar```

The datasets are also available from:
 * OSTrain.tar.gz: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8362616.svg)](https://zenodo.org/record/8362616/files/OSTrain.tar.gz)
 * OSValidate.tar.gz: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8360991.svg)](https://zenodo.org/record/8360991/files/OSValidate.tar.gz)
 * OSTest.tar.gz: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8359342.svg)](https://zenodo.org/record/8359342/files/OSTest.tar.gz)



## Run SUMo

### Filtering phase
The pre-trained models used to take the paper results are available in [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8366378.svg)](https://zenodo.org/record/8366378/files/extracted_features.tar.gz)

#### Source separation
```
cd sumo_pipeline/source_separation
```

Train the model:
```

```



#### Target separation


### Matching phase
#### Correlation



### Comparison with DeepCorr / DeepCoFFEA


### Throughput/latency

To collect the latency/throughput metrics of SUMo follow the instructions in (throughput_sumo.sh)[./experiments/throughput_sumo.sh]. It should output a plot in `sumo/experiments/plot_subsetsum2d.pdf` with the latency/throughput curve of our solution.

