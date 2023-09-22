import typer

import classifier
import extract_features


app = typer.Typer()


featureFolderTrain = '/mnt/nas-shared/torpedo/extracted_features_small_OSTrain/'
featureFolderTest = '/mnt/nas-shared/torpedo/extracted_features_small_OSTest/'
featureFolderValidate = '/mnt/nas-shared/torpedo/extracted_features_small_OSValidate/'

plFileTrain = featureFolderTrain + 'pl_source_separation.csv'
statsFileTrain = featureFolderTrain + 'stats_source_separation.csv'

plFileTest = featureFolderTest + 'pl_source_separation.csv'
statsFileTest = featureFolderTest + 'stats_source_separation.csv'

plFileValidate = featureFolderValidate + 'pl_source_separation.csv'
statsFileValidate = featureFolderValidate + 'stats_source_separation.csv'

#dataset_name = 'OSTrain_filtered'
dataset_name = 'small_OSTest'
#dataset_name = 'small_OSValidate'

model_save_file = 'source_separation_model.joblib'
model_save_file_validate = 'source_separation_model_bayesian_optimization.joblib'

results_file = 'precision_recall_curve_source_separation_zoomin'
results_file_validate = 'precision_recall_curve_source_separation_validation_zoomin'


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
    print("Plotting precision-recall curve without validation results ...")
    classifier.plot_precision_recall_curve_zoomin(plFileTest, statsFileTest, model_save_file, results_file)


@app.command()
def plot_validate():
    print("Plotting precision-recall curve with validation results ...")
    classifier.plot_precision_recall_curve_zoomin(plFileTest, statsFileTest, model_save_file_validate, results_file_validate)


@app.command()
def test_full_pipeline():
    print("Testing model with full pipeline data ...")
    #classifier.test_full_pipeline(plFileTest, statsFileTest, model_save_file, optimalThr=False)
    classifier.test_full_pipeline(dataset_name, plFileTest, statsFileTest, model_save_file_validate, optimalThr=True)
    #classifier.test_full_pipeline(dataset_name, plFileValidate, statsFileValidate, model_save_file_validate, optimalThr=True)


@app.command()
def full_execution():
    validate()
    test_standalone()
    

if __name__ == "__main__":
    app()