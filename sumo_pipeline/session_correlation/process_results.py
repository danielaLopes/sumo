def get_clients_windows_with_packets(possible_request_combinations, buckets_per_window, buckets_overlap):
    windows_with_packets = {}
    for clientSessionId, osSessionId in possible_request_combinations.keys():
        packetListClient = possible_request_combinations[(clientSessionId, osSessionId)]['yPacketCountIn']

        first_bucket = 0
        for i in range(0, len(packetListClient) - buckets_per_window + 1, buckets_overlap):
        #for i in range(client_windows[clientSessionId]):
            windowed_packets_client = packetListClient[first_bucket : first_bucket + buckets_per_window]

            client_sum = sum(windowed_packets_client)

            if clientSessionId not in windows_with_packets:
                windows_with_packets[clientSessionId] = 0
            if client_sum > 0:
                windows_with_packets[clientSessionId] += 1
            
            first_bucket += buckets_overlap
    
    return windows_with_packets


# TODO: CHANGE FOR EPOCHS REALTIME
"""
def calculate_overall_score(possible_request_combinations, database, buckets_per_window, buckets_overlap, epoch, delta):
    windows_with_packets = get_clients_windows_with_packets(possible_request_combinations[epoch], buckets_per_window, buckets_overlap)
    scores_per_session = {} # This variable is not needed, just easier to debug scores
    score_counter = {}
    for (clientSessionId, osSessionId), value in database[epoch][delta].items():
        if (clientSessionId, osSessionId) not in score_counter:
            score_counter[(clientSessionId, osSessionId)] = 0
            scores_per_session[(clientSessionId, osSessionId)] = []
        for window, score in value:
            score_counter[(clientSessionId, osSessionId)] += score
            scores_per_session[(clientSessionId, osSessionId)].append((window, score))
    
    return windows_with_packets, score_counter, scores_per_session
"""


def calculate_overall_score(possible_request_combinations, database, buckets_per_window, buckets_overlap):
    windows_with_packets = get_clients_windows_with_packets(possible_request_combinations, buckets_per_window, buckets_overlap)
    scores_per_session = {} # This variable is not needed, just easier to debug scores
    score_counter = {}
    for key, value in database.items():
        if key not in score_counter:
            score_counter[key] = 0
            scores_per_session[key] = []
        for window, score in value:
            score_counter[key] += score
            scores_per_session[key].append((window, score))
    
    return windows_with_packets, score_counter, scores_per_session


def count_client_correlated_sessions_highest_score(scores, threshold):
    client_sessions_with_highest_scores = {}

    # scores is a dictionary like: {clientSessionId: {osSessionId1: score1, osSessionId2: score2, ...}}
    for clientSessionId in scores.keys():
        client_sessions_with_highest_scores[clientSessionId] = {'correlatedHighestScore': False, 'falseHighestScore': False, 'falseSession': '', 'falseSessions': []}
        sessionScores = []
        for osSessionId, score in scores[clientSessionId].items():
            sessionScores.append(score)
            if osSessionId != clientSessionId:
                client_sessions_with_highest_scores[clientSessionId]['falseSessions'].append(osSessionId)
        max_score_per_session = max(list(sessionScores))
        if clientSessionId in scores[clientSessionId]:
            correlated_score = scores[clientSessionId][clientSessionId]
            if correlated_score == max_score_per_session:
                if correlated_score > threshold:
                    client_sessions_with_highest_scores[clientSessionId]['correlatedHighestScore'] = True
            elif max_score_per_session > threshold:
                for osSessionId, score in scores[clientSessionId].items():
                    if score == max_score_per_session:
                        client_sessions_with_highest_scores[clientSessionId]['falseHighestScore'] = True
                        client_sessions_with_highest_scores[clientSessionId]['falseSession'] = osSessionId
        else:
            for osSessionId, score in scores[clientSessionId].items():
                if score == max_score_per_session:
                    client_sessions_with_highest_scores[clientSessionId]['falseHighestScore'] = True
                    client_sessions_with_highest_scores[clientSessionId]['falseSession'] = osSessionId
    
    return client_sessions_with_highest_scores