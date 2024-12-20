import typer

import classifier

app = typer.Typer()


@app.command()
def train(stats_file_train: str, model_save_file: str) -> None:
    """
    Train the source separation model with the extracted features.

    Args:
        stats_file_train (str): The path to the training features.
        model_save_file (str): The path to the trained model's file
                                        without hyperparameter tuning.

    Example:
        $ python3 app.py train /mnt/nas-shared/torpedo/extracted_features/extracted_features_OSTrain/stats_source_separation.csv models/source_separation_model.joblib
    """
    typer.echo("Training model ...")
    classifier.train(stats_file_train, model_save_file)

@app.command()
def hyperparameter_tuning(stats_file_train: str, 
                        stats_file_validate: str, 
                        stats_file_test: str) -> None:
    """
    Tune source separation hyperparameters using bayesian optimization.

    Args:
        stats_file_train (str): The path to the training features.
        stats_file_validate (str): The path to the validation features.
        stats_file_test (str): The path to the testing features.

    Example:
        $ python3 app.py hyperparameter-tuning /mnt/nas-shared/torpedo/extracted_features/extracted_features_OSTrain/stats_source_separation.csv /mnt/nas-shared/torpedo/extracted_features/extracted_features_OSValidate/stats_source_separation.csv /mnt/nas-shared/torpedo/extracted_features/extracted_features_OSTest/stats_source_separation.csv
    """
    typer.echo("Hyperparameter tuning ...")
    classifier.hyperparameter_tuning(stats_file_train, 
                                     stats_file_validate, 
                                     stats_file_test, 
                                     algorithm='BayesianOptimization')

@app.command()
def test_standalone(stats_file_test: str, 
                    model_save_file_validate: str) -> None:
    """
    Test the source separation model and plot precision-recall curve.

    Args:
        stats_file_test (str): The path to testing features.
        model_save_file_validate (str): The path to the trained model's file
                                        with hyperparameter tuning.

    Example:
        $ python3 app.py test-standalone /mnt/nas-shared/torpedo/extracted_features/extracted_features_OSTest/stats_source_separation.csv models/source_separation_model_bayesian_optimization.joblib
    """
    typer.echo("Testing model ...")
    classifier.test(stats_file_test, model_save_file_validate)

@app.command()
def plot(stats_file_test: str, 
         model_save_file: str, 
         results_file: str) -> None:
    """
    Test the source separation model and plot precision-recall curve
    with zoomed-in curve, without hyperparameter tuning.

    Args:
        stats_file_test (str): The path to the testing features.
        model_save_file (str): The path to the trained model's file
                                without hyperparameter tuning.
        results_file (str): The path save the precision-recall curve plot.

    Example:
        $ python3 app.py plot /mnt/nas-shared/torpedo/extracted_features/extracted_features_OSTest/stats_source_separation.csv models/source_separation_model.joblib precision_recall_curve_source_separation_zoomin
    """
    typer.echo("Plotting precision-recall curve without validation results ...")
    classifier.plot_precision_recall_curve_zoomin(stats_file_test, model_save_file, results_file)

@app.command()
def plot_validate(stats_file_test: str, 
                  model_save_file_validate: str, 
                  results_file_validate: str) -> None:
    """
    Test the source separation model and plot precision-recall curve
    with zoomed-in curve, with hyperparameter tuning.

    Args:
        stats_file_test (str): The path to the testing features.
        model_save_file_validate (str): The path to the trained model's file
                                with hyperparameter tuning.
        results_file_validate (str): The path save the precision-recall curve plot.

    Example:
        $ python3 app.py plot-validate /mnt/nas-shared/torpedo/extracted_features/extracted_features_OSTest/stats_source_separation.csv models/source_separation_model_bayesian_optimization.joblib precision_recall_curve_source_separation_hyperparameter_tuning_zoomin
    """
    typer.echo("Plotting precision-recall curve with validation results ...")
    classifier.plot_precision_recall_curve_zoomin(stats_file_test, model_save_file_validate, results_file_validate)

@app.command()
def test_full_pipeline(dataset_name: str, 
                       stats_file_test: str, 
                       model_save_file_validate: str) -> None:
    """
    Test the source separation model and plot precision-recall curve
    with zoomed-in curve, with hyperparameter tuning and produce data
    for next pipeline stages.

    Args:
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.
        stats_file_test (str): The path to the testing features.
        model_save_file_validate (str): The path to the trained model's file
                                        with hyperparameter tuning.

    Example:
        $ python3 app.py test-full-pipeline OSTest /mnt/nas-shared/torpedo/extracted_features/extracted_features_OSTest/stats_source_separation.csv models/source_separation_model_bayesian_optimization.joblib
    """
    typer.echo("Testing model with full pipeline data ...")
    #classifier.test_full_pipeline(dataset_name, stats_file_test, model_save_file_validate, optimal_thr=True)
    classifier.test_full_pipeline(dataset_name, stats_file_test, model_save_file_validate, optimal_thr=False)


if __name__ == "__main__":
    app()