import typer

import classifier


app = typer.Typer()


#captures_folder_train = '../OSTrain/TrafficCapturesClient/'
#captures_folder_test = '../OSTest/TrafficCapturesClient/'

#featureFolderTrain = '../session_correlation/extracted_features_small_OSTrain/'
featureFolderTrain = '/mnt/nas-shared/torpedo/extracted_features_small_OSTrain/'
featureFolderTest = '/mnt/nas-shared/torpedo/extracted_features_small_OSTest/'
featureFolderValidate = '/mnt/nas-shared/torpedo/extracted_features_small_OSValidate/'

dataset_name = 'small_OSTest'
#dataset_name = 'small_OSValidate'

plFileTrain = featureFolderTrain + 'pl_target_separation.csv'
statsFileTrain = featureFolderTrain + 'stats_target_separation.csv'

plFileTest = featureFolderTest + 'pl_target_separation.csv'
statsFileTest = featureFolderTest + 'stats_target_separation.csv'

plFileValidate = featureFolderValidate + 'pl_target_separation.csv'
statsFileValidate = featureFolderValidate + 'stats_target_separation.csv'

model_save_file = 'target_separation_model.joblib'
model_save_file_validate = 'target_separation_model_bayesian_optimization.joblib'

results_file = 'precision_recall_curve_target_separation_zoomin'
results_file_validate = 'precision_recall_curve_target_separation_validation_zoomin'


@app.command()
def train():
    print("Training model ...")
    classifier.train(plFileTrain, statsFileTrain, model_save_file)


@app.command()
def validate():
    print("Hyperparameter tuning ...")
    #classifier.hyperparameter_tuning(plFileValidate, statsFileValidate, algorithm='GridSearch')
    classifier.hyperparameter_tuning(plFileTrain, statsFileTrain, plFileValidate, statsFileValidate, plFileTest, statsFileTest, algorithm='BayesianOptimization')
    

@app.command()
def test_standalone():
    print("Testing model ...")
    classifier.test(plFileTest, statsFileTest, model_save_file_validate)


@app.command()
def plot():
    print("Plotting precision-recall curve with validation results ...")
    classifier.plot_precision_recall_curve_zoomin(plFileTest, statsFileTest, model_save_file, results_file)


@app.command()
def plot_validate():
    print("Plotting precision-recall curve without validation results ...")
    classifier.plot_precision_recall_curve_zoomin(plFileTest, statsFileTest, model_save_file_validate, results_file_validate)


@app.command()
def test_full_pipeline():
    print("Testing model with full pipeline data ...")
    classifier.test_full_pipeline(dataset_name, model_save_file_validate)


if __name__ == "__main__":
    app()