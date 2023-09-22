from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from hyperopt.early_stop import no_progress_loss

from constants import *
import sliding_subset_sum


def cost_function(metricsMapFinalScores):

    precisions = []
    for threshold in metricsMapFinalScores.keys():
        precisions.append(metricsMapFinalScores[threshold]['precision'])
    average_precision = sum(precisions) / len(precisions)
    min_precision = min(precisions)
    max_precision = max(precisions)
    
    recalls = []
    for threshold in metricsMapFinalScores.keys():
        recalls.append(metricsMapFinalScores[threshold]['recall'])
    average_recall = sum(recalls) / len(recalls)
    min_recall = min(recalls)
    max_recall = max(recalls)


    if average_precision + average_recall == 0:
        f1_score =  0
    else:
        f1_score =  (2 * average_precision * average_recall) / (average_precision + average_recall)

    return average_precision, average_recall, f1_score, min_precision, max_precision, min_recall, max_recall


def test_hyperparameter_tuning_parameters(dataset_name_test, is_full_pipeline=False):
    pattern = r"Hyperparameters: ({.*})"  # Regular expression pattern to match the desired structure

    hyperparameters = []  # List to store the extracted dictionaries

    if is_full_pipeline:
        parameters_file = 'log_parameters_full_pipeline.txt'
        tested_parameters_file = 'log_parameters_tested_full_pipeline.txt'
        precision = 0.95
        recall = 0.8
    else:
        parameters_file = 'log_parameters.txt'
        tested_parameters_file = 'log_parameters_tested.txt'
        precision = 0.98
        recall = 0.8


    with open(TUNING_FOLDER+parameters_file, "r") as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                hyperparameters.append(eval(match.group(1)))


    # Print the collected dictionaries
    for i, params in enumerate(hyperparameters):
        buckets_per_window = int(params['window_size'])
        buckets_overlap = int(params['overlap'])
        # Test dataset
        possible_request_combinations_test, clients_rtts_test, oses_rtts_test, missed_client_flows_full_pipeline_test, missed_os_flows_full_pipeline_test, _, _ = pre_process(dataset_name_test, is_full_pipeline, time_sampling_interval=params['time_sampling_interval'], epoch_size=params['epoch_size'], epoch_tolerance=params['epoch_tolerance'], load=False)
        session_buckets_test, session_windows_test = get_buckets_and_windows(possible_request_combinations_test, buckets_per_window=buckets_per_window, buckets_overlap=buckets_overlap)
        predictions_test = predict(possible_request_combinations_test, clients_rtts_test, oses_rtts_test, session_buckets_test, session_windows_test, delta=params['delta'])
        metricsMap_test, metricsMapFinalScores_test, scoresPerSessionPerClient_test, metricsMapFinalScoresPerSession_test = evaluate_confusion_matrix(possible_request_combinations_test, predictions_test, missed_client_flows_full_pipeline_test, missed_os_flows_full_pipeline_test, delta=delta)

        average_precision_test, average_recall_test, average_f1_score_test, min_precision_test, max_precision_test, min_recall_test, max_recall_test = cost_function(metricsMapFinalScores_test)

        # Minimum requirements for these parameters to be considered
        if min_precision_test >= precision and max_recall_test >= recall:
            print("\n--- BEST params to test", params)
            if os.path.isfile(TUNING_FOLDER+tested_parameters_file):
                write_mode = 'a'
            else:
                write_mode = 'w'
            # Log the hyperparameters and metrics to a file
            with open(TUNING_FOLDER+tested_parameters_file, write_mode) as f:
                f.write(f"Hyperparameters: {params}\n")
                f.write(f"--- TEST Average precision: {average_precision_test}, Average recall: {average_recall_test}, F1 score: {average_f1_score_test}\n\n")

        del possible_request_combinations_test
        del clients_rtts_test
        del oses_rtts_test
        del session_buckets_test
        del session_windows_test
        del predictions_test
        del metricsMap_test
        del metricsMapFinalScores_test
        del scoresPerSessionPerClient_test
        del metricsMapFinalScoresPerSession_test


# Function to be optimized
def objective_function(params, extra_args):

    dataset_name_validate = extra_args[0]
    dataset_name_test = extra_args[1]
    is_full_pipeline = extra_args[2]

    if is_full_pipeline:
        parameters_file = 'log_parameters_full_pipeline.txt'
        dataset_name_validate += '_full_pipeline'
        dataset_name_test += '_full_pipeline'
    else:
        parameters_file = 'log_parameters.txt'

    #buckets_per_window = int(params['window_size'] / (time_sampling_interval / 1000))
    #buckets_overlap = int(params['overlap'] / (time_sampling_interval / 1000))
    buckets_per_window = int(params['window_size'])
    buckets_overlap = int(params['overlap'])

    # These combinations of parameters are not possible
    if buckets_overlap >= buckets_per_window:
        print("--- Skipped overlap {}".format(params['overlap']))
        # Return -1e-6 as very small negative number so that it never gets chosen as good parameter combination
        print("WORSE PARAMETERS", params)
        return {'loss': -1e6, 'params': params, 'status': STATUS_OK}

    print("params", params)
    print("buckets_per_window", buckets_per_window)
    print("buckets_overlap", buckets_overlap)

    # Validation dataset
    possible_request_combinations_validation, clients_rtts_validation, oses_rtts_validation, missed_client_flows_full_pipeline_validation, missed_os_flows_full_pipeline_validation, _, _ = pre_process(dataset_name_validate, is_full_pipeline, time_sampling_interval=params['time_sampling_interval'], epoch_size=params['epoch_size'], epoch_tolerance=params['epoch_tolerance'], load=False)
    session_buckets_validation, session_windows_validation = get_buckets_and_windows(possible_request_combinations_validation, buckets_per_window=buckets_per_window, buckets_overlap=buckets_overlap)
    predictions_validation = predict(possible_request_combinations_validation, clients_rtts_validation, oses_rtts_validation, session_buckets_validation, session_windows_validation, delta=params['delta'])
    metricsMap_validation, metricsMapFinalScores_validation, scoresPerSessionPerClient_validation, metricsMapFinalScoresPerSession_validation = evaluate_confusion_matrix(possible_request_combinations_validation, predictions_validation, missed_client_flows_full_pipeline_validation, missed_os_flows_full_pipeline_validation, delta=delta)
    average_precision_validation, average_recall_validation, average_f1_score_validation, min_precision_validation, max_precision_validation, min_recall_validation, max_recall_validation = cost_function(metricsMapFinalScores_validation)

    if os.path.isfile(TUNING_FOLDER+parameters_file):
        write_mode = 'a'
    else:
        write_mode = 'w'
    # Log the hyperparameters and metrics to a file
    with open(TUNING_FOLDER+parameters_file, write_mode) as f:
        f.write(f"Hyperparameters: {params}\n")
        f.write(f"--- VALIDATION Average precision: {average_precision_validation}, Average recall: {average_recall_validation}, F1 score: {average_f1_score_validation}\n")

    del possible_request_combinations_validation
    del clients_rtts_validation
    del oses_rtts_validation
    del session_buckets_validation
    del session_windows_validation
    del predictions_validation
    del metricsMap_validation
    del metricsMapFinalScores_validation
    del scoresPerSessionPerClient_validation
    del metricsMapFinalScoresPerSession_validation

    #return {'loss': -average_precision, 'params': params, 'status': STATUS_OK}
    return {'loss': -average_f1_score_validation, 'params': params, 'status': STATUS_OK}


def hyperparameter_tuning_bayesian_optimization(dataset_name_validate, dataset_name_test, is_full_pipeline=False):
    np.random.seed(123)
    
    start_time = time.time()
    
    best_parameters = []
    best_parameters_final_score = []

    # Search space definition
    space = {
        #'time_sampling_interval': hp.choice('time_sampling_interval', [20, 100, 200, 500, 1000]), # in seconds
        'time_sampling_interval': hp.choice('time_sampling_interval', [20, 100, 200, 500]), # in seconds
        'epoch_size': hp.choice('epoch_size', [5, 10, 15, 20]), # seconds
        'epoch_tolerance': hp.choice('epoch_tolerance', [1, 2, 5, 10]),
        'window_size': hp.choice('window_size', [2, 4, 6, 8]),
        'overlap': hp.choice('overlap', [0, 1, 2, 3, 4]),
        #'overlap': hp.choice('overlap', [1, 2, 3, 4]),
        'delta': hp.choice('delta', [10, 20, 60, 100])
    }

    print("\n=== Bayesian Optimization...")
    trials = Trials()

    extra_args = (dataset_name_validate, dataset_name_test, is_full_pipeline)
    fmin_objective = partial(objective_function, extra_args=extra_args)

    best = fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials, early_stop_fn=no_progress_loss(20))

    best_hyperparameters = space_eval(space, best)
    
    end_time = time.time()
    print("\n=== Hyperparameter tuning time: {}".format(end_time - start_time))
    with open(TUNING_FOLDER+'log_parameters.txt', 'a') as f:
        f.write(f"Hyperparameter tuning time: {end_time - start_time}\n")

    # Print the index of the best parameters
    print("\n--- best:", best)
    # Print the values of the best parameters
    print("\n--- best_hyperparameters", best_hyperparameters)

    save_file_name = 'best_hyperparameters_session_correlation.joblib'
    if not os.path.exists(TUNING_FOLDER):
        os.makedirs(TUNING_FOLDER)
    joblib.dump(best_hyperparameters, TUNING_FOLDER+save_file_name)


def check_scores(captures_folder_test, dataset_name, is_full_pipeline=False):
    #pdb.set_trace()

    if is_full_pipeline == True:
        dataset_name += '_full_pipeline'
    
    if not os.path.isdir(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)
    if not os.path.isdir(DATA_RESULTS_FOLDER):
        os.mkdir(DATA_RESULTS_FOLDER)

    possible_request_combinations_file = 'possible_request_combinations_{}.pickle'.format(dataset_name)
    clients_rtts_file = 'clients_rtts_{}.pickle'.format(dataset_name)
    oses_rtts_file = 'oses_rtts_{}.pickle'.format(dataset_name)

    missed_client_flows_full_pipeline_file = "missed_client_flows_full_pipeline_{}.pickle".format(dataset_name)
    missed_os_flows_full_pipeline_file = "missed_os_flows_full_pipeline_{}.pickle".format(dataset_name)
    missed_client_flows_per_duration_full_pipeline_file = "missed_client_flows_per_duration_full_pipeline_{}.pickle".format(dataset_name)
    missed_os_flows_per_duration_full_pipeline_file = "missed_os_flows_per_duration_full_pipeline_{}.pickle".format(dataset_name)

    possible_request_combinations = pickle.load(open(DATA_RESULTS_FOLDER+possible_request_combinations_file, "rb"))
    clients_rtts = pickle.load(open(DATA_RESULTS_FOLDER+clients_rtts_file, "rb"))
    oses_rtts = pickle.load(open(DATA_RESULTS_FOLDER+oses_rtts_file, "rb"))
    missed_client_flows_full_pipeline = pickle.load(open(DATA_RESULTS_FOLDER+"missed_client_flows_full_pipeline_{}.pickle".format(dataset_name), "rb"))
    missed_os_flows_full_pipeline = pickle.load(open(DATA_RESULTS_FOLDER+"missed_os_flows_full_pipeline_{}.pickle".format(dataset_name), "rb"))
    metricsMapFinalScores = pickle.load(open(DATA_RESULTS_FOLDER+"metricsMapFinalScores_{}.pickle".format(dataset_name), "rb"))

    session_buckets, session_windows = get_buckets_and_windows(possible_request_combinations)
    print("\n--- After get_buckets_and_windows")
    predictions = predict(possible_request_combinations, clients_rtts, oses_rtts, session_buckets, session_windows)
    print("\n--- After predict")

    metricsMap, metricsMapFinalScores, scoresPerSessionPerClient, metricsMapFinalScoresPerSession = evaluate_confusion_matrix(possible_request_combinations, predictions, missed_client_flows_full_pipeline, missed_os_flows_full_pipeline)
    
    scores_correlated = []
    scores_non_correlated = []

    for pair_preds in predictions.values():

        for (client_session_id, onion_session_id), label_score in pair_preds.items():               
            
            if client_session_id == onion_session_id:
                scores_correlated.append(label_score['score'])
            else:
                scores_non_correlated.append(label_score['score'])

    print("\n ===== MAX scores_correlated", max(scores_correlated))
    print("MIN scores_correlated", min(scores_correlated))
    print("AVG scores_correlated", sum(scores_correlated) / len(scores_correlated))
    print("\n ===== MAX scores_non_correlated", max(scores_non_correlated))
    print("MIN scores_non_correlated", min(scores_non_correlated))
    print("AVG scores_non_correlated", sum(scores_non_correlated) / len(scores_non_correlated))

    results_plot_maker.cdf_pair_scores(FIGURES_RESULTS_FOLDER, predictions, dataset_name)

    