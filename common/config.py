ID = "sample_id"
LABEL = "label"
FAMILY = "family"

SEEDS = [0, 1, 2, 3, 4]
TEST_FRAC = 0.25          # Holds out 3 of the 11 ransomware families per seed for testing
N_INNER = 5               # 5-fold family-disjoint CV inside dev for OOF calibration
N_BENIGN_GROUPS = 20      # Similarity clusters for benign samples