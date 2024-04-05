import sumo_pipeline.extract_raw_pcap_features.query_sumo_dataset as query_sumo_dataset
import results_plot_maker


RESULTS_FOLDER = 'results/'
DATA_RESULTS_FOLDER = RESULTS_FOLDER+'data/'
FIGURES_RESULTS_FOLDER = RESULTS_FOLDER+'figures/'


#data_folder = '/mnt/nas-shared/torpedo/datasets_20230521/small-ostest/experiment_results/'
#data_folder = '/mnt/nas-shared/torpedo/datasets_20230521/small-ostrain-fixed/experiment_results/'
data_folder = '/mnt/nas-shared/torpedo/datasets_20230521/small-osvalidate/experiment_results/'

#dataset_name = 'small_OSTest'
#dataset_name = 'small_OSTrain'
dataset_name = 'small_OSValidate'


def print_dataset_characteristics(data_folder):
    dataset = query_sumo_dataset.SumoDataset(data_folder)
    alexa_paths = dataset.alexa_sessions_paths_all()
    client_paths = dataset.client_sessions_paths_to_oses_all()

    print("\n# Client sessions to alexa before excluding failed requests", len(alexa_paths))
    print("# Client sessions to OSes before excluding failed requests", len(client_paths))

    print("list_alexas", dataset.list_alexas())
    print("list_clients", dataset.list_clients())
    print("list_onions", dataset.list_onions())

    print("len(list_alexas)", len(dataset.list_alexas()))
    print("len(list_clients)", len(dataset.list_clients()))
    print("len(list_onions)", len(dataset.list_onions()))

    try:
        session_paths_to_remove = dataset.filter_sessions_with_failed_requests()
    except query_sumo_dataset.MissingFailedRequestsLogException:
        print("No failed_requests.log files in the dataset")
        exit()

    # remove failed sessions
    alexa_paths = [x for x in alexa_paths if x not in session_paths_to_remove]
    client_paths = [x for x in client_paths if x not in session_paths_to_remove]

    print("\n# Client sessions to alexa", len(alexa_paths))
    print("# Client sessions to OSes", len(client_paths))

    alexa_requests = 0
    for session_path in alexa_paths:
        session_id = query_sumo_dataset.get_session_id_from_path(session_path)
        alexa = query_sumo_dataset.get_alexa_name(session_path)
        client = query_sumo_dataset.get_client_name(session_id)
        alexa = client + "_" + alexa
        alexa_requests += dataset.get_alexa_session_nb_requests(alexa, session_id)

    client_requests = 0
    for session_path in client_paths:
        session_id = query_sumo_dataset.get_session_id_from_path(session_path)
        client = query_sumo_dataset.get_client_name(session_id)
        client_requests += dataset.get_client_session_nb_requests(client, session_id)

    print("\n# Client requests to alexa", alexa_requests)
    print("# Client requests to OSes", client_requests)

    results_plot_maker.session_dataset_statistics(FIGURES_RESULTS_FOLDER, data_folder, dataset_name)




def main():
    print_dataset_characteristics(data_folder)


if __name__ == "__main__":
    main()

