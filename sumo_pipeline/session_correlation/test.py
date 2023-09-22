import pandas as pd

pd.set_option('display.max_colwidth', None)

#arff_path_stats_target = 'extracted_features_small_OSValidate/stats_target_separation.csv'
#arff_path_pl_target = 'extracted_features_small_OSValidate/pl_target_separation.csv'
#arff_path_stats_target = 'extracted_features_small_OSValidate/stats_source_separation.csv'
#arff_path_pl_target = 'extracted_features_small_OSValidate/pl_source_separation.csv'

arff_path_stats_target = 'extracted_features_small_OSTest/stats_source_separation.csv'
arff_path_pl_target = 'extracted_features_small_OSTest/pl_source_separation.csv'

#arff_path_stats_target = 'extracted_features_small_OSTest/stats_target_separation.csv'
#arff_path_pl_target = 'extracted_features_small_OSTest/pl_target_separation.csv'


stats_data = pd.read_csv(arff_path_stats_target)
pl_data = pd.read_csv(arff_path_pl_target)
print(stats_data)
print(pl_data)
#print(stats_data['Class'])
#print(stats_data['Capture'])
#print(pl_data['Capture'])

#print("alexa", stats_data['Class'][:100])
#print("alexa", stats_data['Capture'][:100])

#print("client", stats_data['Class'][8000:8100])
#print("client", stats_data['Capture'][8000:8100])

#print("onion", stats_data['Class'][14000:14100])
#print("onion", stats_data['Capture'][14000:14100])

selected_rows = stats_data[stats_data['Class'] == 0]
print(selected_rows)

selected_rows = pl_data[pl_data['Class'] == 0]
print(selected_rows)


print("========================================")

# This capture is missing from here: client-small-ostest-5-client4_os4-os-small-ostest-4_roccojvl27posqmmoawcmquox2ld4baf62zh36tou6yvehsma7mpe7yd_session_991_client.pcap
# They should have length: 14967
selected_rows = stats_data[stats_data['Class'] == 1]
print(selected_rows)

selected_rows = pl_data[pl_data['Class'] == 1]
print(selected_rows)

# target stats data is missing last line for some reason. all others seems fine