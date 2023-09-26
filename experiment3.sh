cd sumo_pipeline/source_separation
python3 app.py test-full-pipeline OSTest /mnt/nas-shared/torpedo/extracted_features/extracted_features_OSTest/stats_source_separation.csv source_separation_model_bayesian_optimization.joblib
cd ../target_separation
python3 app.py test-full-pipeline OSTest target_separation_model_bayesian_optimization.joblib
cd ../session_correlation
python3 app.py correlate-sessions-full-pipeline OSTest
