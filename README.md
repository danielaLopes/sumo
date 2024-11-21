# SUMo: Flow Correlation Attacks on Tor Onion Sessions using Sliding Subset Sum

This repository presents the code for the artefact evaluation for The Network and Distributed System Security Symposium (NDSS) 2024.

* **Paper available at:** [https://www.ndss-symposium.org/wp-content/uploads/2024-337-paper.pdf](https://www.ndss-symposium.org/wp-content/uploads/2024-337-paper.pdf)

* **Presentation at NDSS'24:** [Presentation pdf](./f0337-lopes.pdf)


# If you make use of our work please cite our NDSS'24 paper:

"Flow Correlation Attacks on Tor Onion Service Sessions with Sliding Subset Sum". Daniela Lopes, Jin-Dong Dong, Daniel Castro, Pedro Medeiros, Diogo Barradas, Bernardo Portela, João Vinagre, Bernardo Ferreira, Nicolas Christin, and Nuno Santos. The Network and Distributed System Security Symposium (NDSS) 2024.



## Artifact Evaluation
Repository: [![DOI](https://zenodo.org/badge/693254187.svg)](https://zenodo.org/doi/10.5281/zenodo.8393157) 


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
The pre-trained models used to take the paper results are available in [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8366377.svg)](https://zenodo.org/records/8366378/files/models.tar.gz)


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
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8388196.svg)](https://zenodo.org/records/8388196/files/deepcoffea_models.tar.gz)


### Throughput/latency

To collect the latency/throughput metrics of SUMo follow the instructions in (experience5.sh)[./experiment4.sh]. It should output a plot in `experiment4/plot_subsetsum2d.pdf` with the latency/throughput curve of our solution. The script also prints the point with maximum throughput.


## Additional functionality
Besides the previous sections for the Arfifact Evaluation, we also made available other resources used for the paper.


### Generating datasets composed of actual Tor traffic
We made a functional prototype of a framework for the automated generation of real Tor traffic datasets to be used to generate datasets for testing attacks like Website Fingerprinting and Traffic Correlation.
* **Github repository:** [https://github.com/danielaLopes/tiger_tor_traffic_generator](https://github.com/danielaLopes/tiger_tor_traffic_generator)
* **WPES '23 Paper:** [TIGER: Tor Traffic Generator for Realistic Experiments](https://dl.acm.org/doi/pdf/10.1145/3603216.3624960)




### Extracting features from the raw .pcaps
We made the features for the filtering phase available online to allow not having the raw .pcaps datasets that are over 50 GB to run the pipeline. However, this step is required to run the SUMo pipeline, and in can be done in the following way:
```
cd sumo_pipeline/extract_raw_pcap_features/
python3 app.py [DATA_FOLDER] [DATASET_NAME]
```

We used the scapy Python library to extract packet data from the raw .pcap files.


### Hyperparameter tuning
#### Source separation
```
cd sumo_pipeline/source_separation
python app.py hyperparameter-tuning [STATS_FILE_TRAIN] [STATS_FILE_VALIDATE] [STATS_FILE_TEST]
```


#### Target separation
```
cd sumo_pipeline/target_separation
python app.py hyperparameter-tuning [STATS_FILE_TRAIN] [STATS_FILE_VALIDATE] [STATS_FILE_TEST]
```


#### Correlation
```
cd sumo_pipeline/session_correlation
python app.py hyperparameter-tuning [DATASET_FOLDER_VALIDATE] [DATASET_FOLDER_TEST]
```



### Guard coverage
To study the feasibility of correlation attacks on the Tor network for circuits with onion services, we conducted the two following studies:
1. **Client-side guard probability:** Establish 3-hop circuits and gather the guard probabilities by country, by AS, and by ISP.
    * Generate circuits script: tor_client_stem.ipynb
        * This will generate ./results/data/client_guard_nodes.joblib, client_middle_nodes.joblib, and ./results/data/client_exit_nodes.joblib
    * Script to plot client-side guard coverage per ISP, per AS and per country: tor_guard_probabilities_client_only.ipynb
2. **Probability of onion service circuit being deanonymized:** We consider the deanonymization probability as the probability of both guard nodes of a circuit between a client and an onion service of being on the same country, AS, ISP, or group of colluding countries, ASes, or ISPs. For this, we establish circuits between our and our onion services and gather the probability that both guard nodes are within the same AS or the same country. 
    * Generate circuits script: client_onion_service_sessions.ipynb
    * Plot script: tor_guard_probabilities_client_onion_session.ipynb

We used stem Python library and Tor version 0.4.7.14 for both experiments.



# If you make use of our work please cite our NDSS'24 paper:

"Flow Correlation Attacks on Tor Onion Service Sessions with Sliding Subset Sum". Daniela Lopes, Jin-Dong Dong, Daniel Castro, Pedro Medeiros, Diogo Barradas, Bernardo Portela, João Vinagre, Bernardo Ferreira, Nicolas Christin, and Nuno Santos. The Network and Distributed System Security Symposium (NDSS) 2024.
