
# %%
import joblib

from guard_coverage.session_guard_nodes import Session

RESULTS_FILE = "./results/data/guard_nodes.joblib"

if __name__ == "__main__":
    guard_nodes = joblib.load(RESULTS_FILE)
    print(len(guard_nodes))
    print(guard_nodes[: -10])


