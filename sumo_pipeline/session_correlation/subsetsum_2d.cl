#define MAX_LOCAL_SIZE 20000

// #define assert(_cond) if (!(_cond)) printf("[%i] invalid condition: %s\n", __LINE__, #_cond);
#define assert(_cond) /* put empty on production build */

int get_index(const int i, const int j, const int n_columns)
{
    return i * n_columns + j;
}

int get_bucket(const int window, const int buckets_per_window, const int buckets_overlap, const int thread_pair_id) {
    return window * (buckets_per_window - buckets_overlap) + (thread_pair_id * buckets_overlap);
}

#define LOOKUP_ARRAY_SIZE 12288

// #define GET_IN_BITARRAY(_bitArray, _idx) \
//     (((1 << ((_idx) & 0x1f)) & ((_bitArray)[(_idx) >> 5])) >> ((_idx) & 0x1f))
// #define SET_IN_BITARRAY(_bitArray, _idx, _bit) \
//         (_bitArray)[(_idx) >> 5] = (_bit) ?  \
//         (((_bitArray)[(_idx) >> 5]) | (1 << (((_idx)&0x1f)))) : \
//         (((_bitArray)[(_idx) >> 5]) & ~(1 << (((_idx)&0x1f)))) 


void kernel outer_loop_subset_sum(
    global const  int* client_nums_array,
    global const int* os_nums_array,
    local int* windowed_client_nums,
    local int* windowed_os_nums,
    const int delta,
    const int buckets_per_window,
    const int buckets_overlap,
    global const int* n_windows,
    global int* score,
    global const int* acc_windows)
{
    int thread_pair_id = get_global_id(0); // first dimension
    int thread_window_id = get_global_id(1); // second dimension


    local bool lookup[LOOKUP_ARRAY_SIZE];
    // local unsigned lookup[LOOKUP_ARRAY_SIZE]; // Bit array
    int n_cols = LOOKUP_ARRAY_SIZE / buckets_per_window;

    // Since I have a fixed number of threads per pair of N_WINDOWS, I don't actually want to use the extra ones
    if (thread_window_id > n_windows[thread_pair_id] - 1) {
        return;
    }
    int start_bucket_index = acc_windows[thread_pair_id] + get_bucket(thread_window_id, buckets_per_window, buckets_overlap, thread_pair_id);
    int start_window_index = acc_windows[thread_pair_id] + thread_window_id;
    for (int k = 0; k < buckets_per_window; k++) {
        windowed_client_nums[k] = client_nums_array[start_bucket_index + k];
        windowed_os_nums[k] = os_nums_array[start_bucket_index + k];
    }
    int clientSum = 0;
    int osSum = 0;
    // TODO: check max OS sum in the cpp code
    for (int k = 0; k < buckets_per_window; k++)
    {
        clientSum += windowed_client_nums[k];
        osSum += windowed_os_nums[k];
    }
    if (clientSum == 0 && osSum == 0) {
        score[start_window_index] = 0;
        return;  
    }
    else if (clientSum == 0 && osSum > 0) {
        score[start_window_index] = -1;
        return;
    }
    else if (clientSum > 0 && osSum == 0) {
        score[start_window_index] = -1;
        return;
    }
    else {
        
    // TODO: sometimes get_index(0, osSum-1, n_cols) is larger than LOOKUP_ARRAY_SIZE
    //     printf("osSum=%i idx=%i\n", osSum, get_index(0, osSum-1, n_cols));
    //     score[start_window_index] = lookup[get_index(0, 0, n_cols)];
    // return;
        for (int j = 0; j <= osSum; j++)
        {
            int lookup_index = get_index(0, j, n_cols);
            if (!(lookup_index < LOOKUP_ARRAY_SIZE))
                continue;
            assert(lookup_index < LOOKUP_ARRAY_SIZE);
            if(lookup_index >= LOOKUP_ARRAY_SIZE) printf("lookup_index=%i, j=%i, osSum=%i, n_cols=%i\n", lookup_index, j, osSum, n_cols);
            lookup[lookup_index] = (windowed_os_nums[0] == j);
            /* *** Using bit array *** */
            // SET_IN_BITARRAY(lookup, lookup_index, (windowed_os_nums[0] == j));
            /* *********************** */
        }
        // from here on, 
        // printf();
        for (int i = 1; i < buckets_per_window; i++) {
            for (int j = 0; j <= osSum; j++) { 
                int lookup_index = get_index(i, j, n_cols);
                int lookup_index_up = get_index(i - 1, j, n_cols);
                int lookup_index_left = get_index(i - 1, j - windowed_os_nums[i], n_cols);
                if (!((lookup_index < LOOKUP_ARRAY_SIZE) && (lookup_index_up < LOOKUP_ARRAY_SIZE)))
                    continue;
                if (0 <= j - windowed_os_nums[i] && j - windowed_os_nums[i] <= osSum) {
                    if (!(lookup_index_left < LOOKUP_ARRAY_SIZE))
                        continue;
                    assert(lookup_index < LOOKUP_ARRAY_SIZE);
                    assert(lookup_index_up < LOOKUP_ARRAY_SIZE);
                    assert(lookup_index_left < LOOKUP_ARRAY_SIZE);
                    lookup[lookup_index] = 
                        windowed_os_nums[i] == j ||
                        lookup[lookup_index_up] ||
                        lookup[lookup_index_left];
                    /* *** Using bit array *** */
                    // SET_IN_BITARRAY(lookup, lookup_index, windowed_os_nums[i] == j ||
                    //     GET_IN_BITARRAY(lookup, lookup_index_up) ||
                    //     GET_IN_BITARRAY(lookup, lookup_index_left)
                    // );
                    /* *********************** */
                }
                else {
                    assert(lookup_index < LOOKUP_ARRAY_SIZE);
                    assert(lookup_index_up < LOOKUP_ARRAY_SIZE);
                    lookup[lookup_index] = 
                        windowed_os_nums[i] == j ||
                        lookup[lookup_index_up];
                    /* *** Using bit array *** */
                    // SET_IN_BITARRAY(lookup, lookup_index, windowed_os_nums[i] == j ||
                    //     GET_IN_BITARRAY(lookup, lookup_index_up)
                    // );
                    /* *********************** */
                }
            }
        }

        int start = clientSum - delta;
        int end = clientSum + delta;

        /* If our range is out of bounds of the possible values for the given
        set, then it has no solution */
        if ((start < 0 && end < 0) || (start > osSum && end > osSum)) {
            score[start_window_index] = -1;
            return;
        }

        /* Ajust our range to the subset's to only search within 
        possible values */
        if (start < 0 && end <= osSum && end >= 0) {
            start = 0;
        }
        if (start >= 0 && start <= osSum && end > osSum) {
            end = osSum;
        }
        if (start < 0 && end > osSum) {
            start = 0;
            end = osSum;
        }

        /* Search if there is a solution in the last line of the lookup table,
        only for the range of values we want */
        for (int j = end; j >= start; j--) {
            int lookup_index = get_index(buckets_per_window - 1, j, n_cols);
            assert(lookup_index < LOOKUP_ARRAY_SIZE);
            if (lookup[lookup_index] == 1) {
            // if (GET_IN_BITARRAY(lookup, lookup_index) == 1) {
                score[start_window_index] = 1;
                return;
            }
        }
        // If it reached here, then subset sum has no solution
        score[start_window_index] = -1;
        return;
    }
}
