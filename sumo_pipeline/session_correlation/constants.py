RESULTS_FOLDER = 'results/'
DATA_RESULTS_FOLDER = RESULTS_FOLDER+'data/'
FIGURES_RESULTS_FOLDER = RESULTS_FOLDER+'figures/'
FIGURES_PAPER_RESULTS_FOLDER = RESULTS_FOLDER+'figures_paper/'

TUNING_FOLDER = "hyperparameter_tuning/"

def get_session_concurrency_at_onion_file_name(dataset_name: str) -> str:
    return f'{DATA_RESULTS_FOLDER}os_session_concurrency_{dataset_name}.pickle'