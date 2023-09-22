import ctypes
from ctypes import alignment, cdll


lib = ctypes.CDLL('./subsetsumopencl2d.so')

lib.wholeLoopSubsetSum.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
lib.wholeLoopSubsetSum.restype = None


def whole_loop_subset_sum(packetListClient, packetListOS, nPairs, nBuckets, delta, buckets_per_window, buckets_overlap, nWindows, acc_windows):
    count_buckets = sum(nBuckets)
    count_windows = sum(nWindows)
    #print("nPairs", nPairs)
    #print("acc_windows", acc_windows)
    client_nums_array = (ctypes.c_int * count_buckets)(*packetListClient)
    os_nums_array = (ctypes.c_int * count_buckets)(*packetListOS)
    n_buckets_array = (ctypes.c_int * nPairs)(*nBuckets)
    n_windows_array = (ctypes.c_int * nPairs)(*nWindows)
    scores = (ctypes.c_int * count_windows)()
    acc_windows_array = (ctypes.c_int * nPairs)(*acc_windows)

    lib.wholeLoopSubsetSum(client_nums_array, os_nums_array, n_buckets_array, n_windows_array, nPairs, delta, buckets_per_window, buckets_overlap, scores, acc_windows_array)

    return list(scores)


def sample_test():
    print("Main")


if __name__ == "__main__":
    sample_test()
