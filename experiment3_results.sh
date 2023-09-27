cd sumo_pipeline/source_separation
python3 app.py plot-validate ../../extracted_features/extracted_features_OSTest/stats_source_separation.csv ../../models/source_separation/source_separation_model_bayesian_optimization.joblib precision_recall_curve_source_separation_hyperparameter_tuning_zoomin
cd ../target_separation
python3 app.py plot-validate ../../extracted_features/extracted_features_OSTest/stats_target_separation.csv ../../models/target_separation/target_separation_model_bayesian_optimization.joblib precision_recall_curve_target_separation_hyperparameter_tuning_zoomin
cd ../session_correlation
python3 app.py plot-full-pipeline ../../extracted_features/extracted_features_OSTest/ OSTest
