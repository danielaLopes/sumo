# Deep Coffea implementations

## Directory/file structure

### author-implementation/
- https://github.com/traffic-analysis/deepcoffea
- Snapshot taken on 2022-12-8.
- Uses an old version of Tensorflow.

### data/
- CrawlE_Proc/
  - Deep Coffea's **raw** packet captures.
  - Each file contains the timestamp and the packet size.
  - Structure:
    ```
    - ./CrawlE_Proc
      - inflow/
        - 0_1
        - 0_10
        - ...
        - 104_101640
        - ...
      - outflow/
        - 0_1
        - 0_10
        - ...
        - 104_101640
        - ...
    ```

- DCF_data/
  - Deep Coffea's **processed** data (ready for their model, `author-implementation/new_model.py`).
  - Ready: after filtering and window partitioning.

- DCF_timegap/
  - Not used.
  - From their github README.

- Deepcorr_data/
  - Deep Corr data in Deep Coffea format (ready for `author-implementation/new_model.py`).

### data_utils.py
- All things related to processing Deep Coffea data.

### model.py
- The actual ML model and training stage for Deep Coffea.

## Notes
### Vector representation
- `v_i = [I_i || S_i]`
- I: IPD, S: size
- `I_i` consists of upstream (`I^u`) and downstream (`-I^d`) IPDs (local interleaving)

### Deep Corr data to Deep Coffea format?
In Deep Coffea's original data format, the feature vector looks like this:

- (tor) `v_i = [(T)I_i || (T)S_i]`
- (exit) `v_i = [(E)I_i || (E)S_i]`

where `I_i` consists of (interleaved) upstream IPDs (`I^u`) and downstream IPDs (`I^d`) and `S_i` consists of (interleaved) upstream sizes (`S^u`) and downstream sizes (`S^d`). However, this representation can only be constructed when we have the timestamps. In Deep Corr's data, the upstream/downstream packets are already separated and we don't know the actual timestamps. Therefore, we can only construct the reprentation like this:

- (tor) `v_i = [(T)I_i^u || (T)I_i^d || (T)S_i^u || (T)S_i^d]`
- (exit) `v_i = [(E)I_i^u || (E)I_i^d || (E)S_i^u || (E)S_i^d]`

(ones that were mentioned in the DeepCoffea paper, but they yield poor results).

Since the original IDPs are interleaved, the length can be fixed simply by setting the tor_len (checkout preprocess_dcf_data). However, here we need to deal with upstream/downstream separately. The way I do it is just make sure both the upstream/downstream don't exceed half of what we set as the fixed length, e.g., 500 -> 250.

### Deep Coffea data to Deep Corr format?
Since we have the raw packet captures, it is possible to do it! But the directions may be a little mixed up for now. Although we are not sure if `inflow` is client-side captures or server-side captures.