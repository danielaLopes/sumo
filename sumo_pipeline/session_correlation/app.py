import typer
import multiprocessing
import os
# Set max threads to avoid safe limit from being established
os.environ['NUMEXPR_MAX_THREADS'] = str(multiprocessing.cpu_count())

import sliding_subset_sum
import results_plot_maker
import extract_pair_features
from constants import *


class SlidingSubsetSumConfig:
    epoch_size = 5
    epoch_tolerance = 1
    time_sampling_interval = 100
    window_size = 4
    overlap = 2
    delta = 100


class SlidingSubsetSumFullPipelineConfig:
    epoch_size = 5
    epoch_tolerance = 1
    time_sampling_interval = 200
    window_size = 6
    overlap = 3
    delta = 60


def __get_instance(dataset_name: str) -> sliding_subset_sum.SlidingSubsetSum:
    """
    Wrapper to initialize or retrieve SlidingSubsetSum instance

    Args:
        dataset_folder (str): The path to the testing features.

    Returns:
        An instance of sliding_subset_sum.SlidingSubsetSum
    """
    return sliding_subset_sum.SlidingSubsetSum.get_instance(dataset_name,
                                                            SlidingSubsetSumConfig.epoch_size,
                                                            SlidingSubsetSumConfig.epoch_tolerance,
                                                            SlidingSubsetSumConfig.time_sampling_interval,
                                                            SlidingSubsetSumConfig.window_size,
                                                            SlidingSubsetSumConfig.overlap,
                                                            SlidingSubsetSumConfig.delta,
                                                            SlidingSubsetSumFullPipelineConfig.epoch_size,
                                                            SlidingSubsetSumFullPipelineConfig.epoch_tolerance,
                                                            SlidingSubsetSumFullPipelineConfig.time_sampling_interval,
                                                            SlidingSubsetSumFullPipelineConfig.window_size,
                                                            SlidingSubsetSumFullPipelineConfig.overlap,
                                                            SlidingSubsetSumFullPipelineConfig.delta)


app = typer.Typer()


@app.command()
def extract_pairs_features(dataset_folder: str, dataset_name: str, timeSamplingInterval: int):
    """
    Goes through every client-side flow features (packets and timings) and groups it with all
    all onion-side flow features overlapping in time.

    Args:
        dataset_folder (str): The path to the testing features.
        dataset_name (str): The name that uniquely identifies this dataset so that we can store results.
        timeSamplingInterval (int): The bucket size in miliseconds. If timeSamplingInterval==500, then we will 
                                    create an histogram where each bucket gets the count of packets seen
                                    every 500 ms.

    Example:
        $ python3 app.py extract-pairs-features /mnt/nas-shared/torpedo/extracted_features_OSTest/ OSTest 500
    """
    typer.echo("Extracting pairs features ...")
    extract_pair_features.extract_pairs_features(dataset_folder, dataset_name, timeSamplingInterval)

@app.command()
def correlate_sessions(dataset_name: str):
    """
    Run the session correlation stage assuming a perfect filtering phase.

    Args:
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.

    Example:
        $ python3 app.py correlate-sessions OSTest
    """
    typer.echo("Correlating sessions ...")
    instance = __get_instance(dataset_name)
    instance.correlate_sessions(dataset_name, is_full_pipeline=False)

# TODO: there's a bug, giving better results than the ones in the paper, missing fps
@app.command()
def correlate_sessions_full_pipeline(dataset_name: str):
    """
    Run the session correlation stage using filtering phase.

    Args:
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.

    Example:
        $ python3 app.py correlate-sessions-full-pipeline OSTest
    """
    typer.echo("Correlating sessions full pipeline ...")
    instance = __get_instance(dataset_name)
    instance.correlate_sessions(dataset_name, is_full_pipeline=True)

# TODO
@app.command()
def plot_dataset_stats(dataset_folder: str, dataset_name: str):
    """
    Plot dataset statistics pertaining session duration, 
    number of requests per session, and number of sessions
    per onion service.

    Args:
        dataset_folder (str): The path to the testing features.
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.

    Example:
        $ python3 app.py plot-dataset-stats /mnt/nas-shared/torpedo/extracted_features_OSTest/ OSTest
    """
    typer.echo("Plotting dataset statistics ...")
    results_plot_maker.session_dataset_statistics(FIGURES_RESULTS_FOLDER, dataset_folder, dataset_name)

@app.command()
def plot(pcaps_folder: str, dataset_name: str):
    """
    Plot correlation results assuming a perfect filtering phase.

    Args:
        pcaps_folder (str): The path to the raw pcaps.
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.

    Example:
        $ python3 app.py plot /mnt/nas-shared/torpedo/datasets_20230521/OSTest/experiment_results/ OSTest
    """
    typer.echo("Plotting correlation results ...")
    instance = __get_instance(dataset_name)
    instance.plot(pcaps_folder, dataset_name)

@app.command()
def plot_paper_results(dataset_name: str):
    """
    Plot correlation results assuming a perfect filtering phase
    for the main plots presented in the paper.

    Args:
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.

    Example:
        $ python3 app.py plot-paper-results OSTest
    """
    typer.echo("Plotting correlation results presented in paper ...")
    instance = __get_instance(dataset_name)
    instance.plot_paper_results(dataset_name)

@app.command()
def plot_full_pipeline(dataset_name: str):
    """
    Plot correlation results using filtering phase.

    Args:
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.

    Example:
        $ python3 app.py plot-full-pipeline OSTest
    """
    typer.echo("Plotting correlation results full pipeline ...")
    instance = __get_instance(dataset_name)
    instance.plot_full_pipeline()

@app.command()
def partial_coverage(dataset_name: str):
    """
    Plot correlation results assuming partial coverage setting
    where we exclude a continent from the adversary's coverage
    each time.

    Args:
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.

    Example:
        $ python3 app.py partial-coverage OSTest
    """
    typer.echo("Evaluating partial coverage ...")
    instance = __get_instance(dataset_name)
    instance.evaluate_coverage_by_continent(dataset_name)

@app.command()
def partial_coverage_by_eu_country(dataset_name: str):
    """
    Plot correlation results assuming partial coverage setting by
    guard probability percentage of the EU countries with higher
    guard coverage.

    Args:
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.

    Example:
        $ python3 app.py partial-coverage-by-eu-country OSTest
    """
    typer.echo("Evaluating partial coverage by EU country ...")
    instance = __get_instance(dataset_name)
    instance.evaluate_coverage_by_eu_country(dataset_name)

# TODO
@app.command()
def hyperparameter_tuning(dataset_folder_validate: str, dataset_folder_test: str):
    """
    Tune session correlation hyperparameters without filtering phase
    using bayesian optimization.

    Args:
        dataset_folder_validate (str): The path to the validation features.
        dataset_folder_test (str): The path to the testing features.

    Example:
        $ python3 app.py hyperparameter-tuning /mnt/nas-shared/torpedo/extracted_features_small_OSValidate/ /mnt/nas-shared/torpedo/extracted_features_OSTest/
    """
    typer.echo("Tuning parameters with validation dataset ...")
    sliding_subset_sum.hyperparameter_tuning_bayesian_optimization(dataset_folder_validate, dataset_folder_test, is_full_pipeline=False)

# TODO
@app.command()
def hyperparameter_tuning_full_pipeline(dataset_name_validate: str, dataset_name_test: str):
    """
    Tune session correlation hyperparameters with filtering phase.

    Args:
        dataset_folder_validate (str): The path to the validation features.
        dataset_folder_test (str): The path to the testing features.

    Example:
        $ python3 app.py hyperparameter-tuning-full-pipeline /mnt/nas-shared/torpedo/extracted_features_small_OSValidate/ /mnt/nas-shared/torpedo/extracted_features_OSTest/
    """
    typer.echo("Tuning parameters with validation dataset for full pipeline...")
    sliding_subset_sum.hyperparameter_tuning_bayesian_optimization(dataset_name_validate, dataset_name_test,  is_full_pipeline=True)

# TODO
@app.command()
def test_hyperparameters(dataset_name: str):
    """
    Test session correlation hyperparameters assuming perfect filtering phase.

    Args:
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.

    Example:
        $ python3 app.py hyperparameter-tuning-full-pipeline /mnt/nas-shared/torpedo/extracted_features_OSTest/
    """
    typer.echo("Selecting best tuned hyperparameters with test dataset ...")
    sliding_subset_sum.test_hyperparameter_tuning_parameters(dataset_name,  is_full_pipeline=False)

# TODO
@app.command()
def test_hyperparameters_full_pipeline(dataset_name: str):
    """
    Test session correlation hyperparameters with filtering phase.

    Args:
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.

    Example:
        $ python3 app.py hyperparameter-tuning-full-pipeline /mnt/nas-shared/torpedo/extracted_features_OSTest/
    """
    typer.echo("Selecting best tuned hyperparameters with test dataset for full pipeline ...")
    sliding_subset_sum.test_hyperparameter_tuning_parameters(dataset_name,  is_full_pipeline=True)

# TODO
@app.command()
def check_scores(dataset_folder: str, dataset_name: str):
    """
    Plots a CDF for the scores given by our sliding subset sum algorithm, 
    to correlated and non-correlated sessions.

    Args:
        dataset_folder (str): The path to the testing features.
        dataset_name (str): The name that uniquely identifies this dataset 
                            so that we can store results.

    Example:
        $ python3 app.py check-scores /mnt/nas-shared/torpedo/extracted_features_OSTest/ OSTest
    """
    typer.echo("Producing scores CDFs so that we can analyze the distribution of produced scores ...")
    sliding_subset_sum.check_scores(dataset_folder, dataset_name, is_full_pipeline=False)


if __name__ == "__main__":
    app()
