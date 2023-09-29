# SUMo: Flow Correlation Attacks on Tor Onion Sessions using Sliding Subset Sum

This repository presents the code for the artefact evaluation for The Network and Distributed System Security Symposium (NDSS) 2024.

## Artifact Evaluation
Repository: [![DOI](https://zenodo.org/badge/693254187.svg)](https://zenodo.org/badge/latestdoi/693254187)

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
We expect this to take a maximum of 40 minutes. This will generate plots in ./experiment1 
```
./experiment1.sh
./experiment1_results.sh
```

#### Experiment (E2) Session Matching with Partial Coverage
We expect this to take a maximum of 2.5 hours. This will generate plots in ./experiment2
```
./experiment2.sh
```

#### Experiment (E3) Session Matching with Imperfect Filtering Phase
We expect this to take a maximum of 1 hour. This will generate plots in ./experiment3
```
./experiment3_setup.sh
./experiment3.sh
./experiment3_results.sh
```

#### Experiment (E4) Comparison with the State-of-the-Art on Flow Correlation
We expect this to take a maximum of xxx. This will generate plots in ./experiment4
```
./experiment4_setup.sh
./experiment4.sh
./experiment4_results.sh
```

#### Experiment (E5) Throughput Evaluation
We expect this to take a maximum of 1 hour. This will generate plots in ./experiment5
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
python app.py --help
python app.py [COMMAND] --help
```


#### Target separation
```
cd sumo_pipeline/target_separation
python app.py --help
python app.py [COMMAND] --help
```


### Matching phase
#### Correlation
```
cd sumo_pipeline/session_correlation
python app.py --help
python app.py [COMMAND] --help
```



### Comparison with DeepCorr / DeepCoFFEA
The SUMo features converted to the DeepCoFFEA format are available at:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8386335.svg)](https://zenodo.org/record/8386335/files/sumo_features_for_deepcoffea.tar.gz)

The DeepCoFFEA models trained with SUMos data are available at:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8388196.svg)](https://zenodo.org/record/8386335/files/deepcoffea_models.tar.gz)


### Throughput/latency

To collect the latency/throughput metrics of SUMo follow the instructions in (experience5.sh)[./experiment4.sh]. It should output a plot in `experiment4/plot_subsetsum2d.pdf` with the latency/throughput curve of our solution. The script also prints the point with maximum throughput.


### Guard coverage
#### Study Tor relays in client circuit
```
cd guard_coverage
```

* Execute tor_client_stem.ipynb to obtain ./results/data/client_guard_nodes.joblib, client_middle_nodes.joblib, ./results/data/client_exit_nodes.joblib

* Plot client-side guard coverage per ISP, per AS and per country by executing plot_guard_probabilities_client_only.ipynb


# If you make use of our work please cite our NDSS'24 paper:

"Flow Correlation Attacks on Tor Onion Service Sessions with Sliding Subset Sum". Daniela Lopes, Jin-Dong Dong, Daniel Castro, Pedro Medeiros, Diogo Barradas, Bernardo Portela, João Vinagre, Bernardo Ferreira, Nicolas Christin, and Nuno Santos. The Network and Distributed System Security Symposium (NDSS) 2024.
