cd sumo_pipeline/source_separation
python3 app.py plot-validate /mnt/nas-shared/torpedo/extracted_features/extracted_features_OSTest/stats_source_separation.csv source_separation_model_bayesian_optimization.joblib precision_recall_curve_source_separation_hyperparameter_tuning_zoomin
cd ../target_separation
python3 app.py plot-validate /mnt/nas-shared/torpedo/extracted_features/extracted_features_OSTest/stats_target_separation.csv target_separation_model_bayesian_optimization.joblib precision_recall_curve_target_separation_hyperparameter_tuning_zoomin
cd ../session_correlation
python3 app.py plot-full-pipeline /mnt/nas-shared/torpedo/extracted_features_OSTest/ OSTest