from format_converter import deepcoffea_map

train_dir_path = "../sumo_features_for_deepcoffea/OSTrain/experiment_results_for_deepcoffea_req/"
val_dir_path = "../sumo_features_for_deepcoffea/OSValidate/experiment_results_for_deepcoffea_req/"
test_dir_path = "../sumo_features_for_deepcoffea/OSTest/experiment_results_for_deepcoffea_req/"
output_train_dir_path = "datasets/datasets_20230521_train_deepcoffea/"
output_test_dir_path= "datasets/datasets_20230521_test_deepcoffea/"

deepcoffea_map(train_dir_path, val_dir_path, output_train_dir_path)
deepcoffea_map(test_dir_path, None, output_test_dir_path)