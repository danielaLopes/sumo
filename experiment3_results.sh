#!/bin/bash

cd sumo_pipeline/source_separation
python3 app.py plot-validate ../../extracted_features/extracted_features_OSTest/stats_source_separation.csv ../../models/source_separation/source_separation_model_bayesian_optimization.joblib precision_recall_curve_source_separation_hyperparameter_tuning_zoomin
cp results/*.{pdf,png} ../../experiment3

cd ../target_separation
python3 app.py plot-validate ../../extracted_features/extracted_features_OSTest/stats_target_separation.csv ../../models/target_separation/target_separation_model_bayesian_optimization.joblib precision_recall_curve_target_separation_hyperparameter_tuning_zoomin
cp results/*.{pdf,png} ../../experiment3

cd ../session_correlation
python3 app.py plot-full-pipeline OSTest
cp results/figures_full_pipeline/*.{pdf,png} ../../experiment3