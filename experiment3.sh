mkdir experiment3

echo "Testing source separation model ..."
cd sumo_pipeline/source_separation
LOCATION_EXTRACTED_FEATURES=../../extracted_features
python3 app.py test-full-pipeline OSTest $LOCATION_EXTRACTED_FEATURES/extracted_features_OSTest/stats_source_separation.csv ../../models/source_separation/source_separation_model_bayesian_optimization.joblib

echo "Testing target separation model ..."
cd ../target_separation
python3 app.py test-full-pipeline OSTest ../../models/target_separation/target_separation_model_bayesian_optimization.joblib

echo "Testing session correlation with full pipeline ..."
cd ../session_correlation
python3 app.py correlate-sessions-full-pipeline OSTest

cp results/figures_full_pipeline/*.{pdf,png} ../../experiment3
