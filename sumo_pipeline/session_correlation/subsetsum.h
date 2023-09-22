#include <iostream>
#include <vector>


class SubsetSum 
{
    public:
        //SubsetSum(int* nums, int n, int osSum);
        SubsetSum(std::vector<int> nums, int n, int osSum);
        ~SubsetSum();
        void solve();
        int has_solution_in_range(int start, int end);

    private:
        std::vector<int> _nums;
        int _N;
        int _min;
        int _max;
        bool** _lookup;
};


struct SessionPair {
    int* client_nums_array;
    int* os_nums_array;
    int n_buckets;
    int n_windows;
};


struct Scores {
    int* scores;
};


const int PENALTY = -1;
const int BASE = 0;
const int GAIN = 1;


// 21march_2022 window len = 1000 ms
//Average #windows per pair:  111.67436887368227
//Maximum #windows per pair:  1205
//Minimum #windows per pair:  7

// 06june_2022 window len = 1000 ms
//Average #windows per pair:  121.90372755390234
// Maximum #windows per pair:  1521
// Minimum #windows per pair:  51

// azureDataset window len = 1000 ms
// Average #windows per pair:  129.01267336202773
// Maximum #windows per pair:  196
// Minimum #windows per pair:  67

// 29 november window len = 1000 ms
// Maximum #windows per pair: 4593 
const int N_WINDOWS = 5000;
