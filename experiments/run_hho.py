"""
run_hho_30.py
Menjalankan HHO 30 kali dengan seed 101 s.d. 130 untuk skripsi.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import pandas as pd
import joblib
import time
import os
from hho import hho

# ============================================================================
# ATO System Parameters
# ============================================================================
ARRIVAL_RATES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
PRODUCTION_MEANS = [0.11, 0.09, 0.18, 0.12, 0.15, 0.13,
                    0.11, 0.22, 0.17, 0.13, 0.21, 0.15]
PRODUCTION_STDS = [0.03, 0.02, 0.02, 0.03, 0.02, 0.01,
                   0.03, 0.02, 0.01, 0.02, 0.01, 0.02]
PROFIT_PER_ITEM = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
HOLDING_COST_PER_ITEM = [0.7] * 12
CAPACITY_PER_ITEM = [30] * 12
SIMULATION_HORIZON = 70
WARM_UP_PERIOD = 20

# ============================================================================
# HHO Parameters
# ============================================================================
HHO_POP_SIZE = 30
MAX_FE = 500                     # Bisa disesuaikan, 5000 cukup untuk skripsi
N_RUNS = 30
START_SEED = 101

# ============================================================================
# Wrapper untuk Random Forest
# ============================================================================
class ObjectiveWrapper:
    def __init__(self, rf_model):
        self.rf_model = rf_model
        self.n_FE = 0
    def reset(self):
        self.n_FE = 0
    def __call__(self, x):
        self.n_FE += 1
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return float(self.rf_model.predict(x)[0])

# ============================================================================
# Load model
# ============================================================================
def load_rf_model(path="best_random_forest_surrogate.joblib"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    return joblib.load(path)

# ============================================================================
# Main
# ============================================================================
def main():
    print("="*60)
    print(f"HHO 30 runs (MAX_FE={MAX_FE}, seeds {START_SEED}..{START_SEED+N_RUNS-1})")
    print("="*60)
    
    # Muat model Random Forest
    try:
        rf = load_rf_model()
        print("Model Random Forest berhasil dimuat.\n")
    except FileNotFoundError as e:
        print(e)
        return
    
    lb = [0]*12
    ub = [30]*12
    results = []
    
    for i in range(N_RUNS):
        seed = START_SEED + i
        print(f"HHO run {i+1}/{N_RUNS} (seed={seed})...", end=" ", flush=True)
        
        obj = ObjectiveWrapper(rf)
        t_start = time.perf_counter()
        x_best, f_best, n_fe, _ = hho(
            obj,
            max_FE=MAX_FE,
            n_pop=HHO_POP_SIZE,
            lb=lb,
            ub=ub,
            seed=seed
        )
        t_elapsed = time.perf_counter() - t_start
        
        print(f"profit={f_best:.4f}, FE={n_fe}, time={t_elapsed:.2f}s")
        
        results.append({
            'run': i+1,
            'seed': seed,
            'x_best': ' '.join(map(str, x_best)),
            'profit_predicted': f_best,
            'n_FE_used': n_fe,
            'time_seconds': t_elapsed
        })
        
        # Simpan setiap run secara bertahap
        pd.DataFrame(results).to_csv('hho_30_runs.csv', index=False)
    
    print("HHO 30 runs selesai. Hasil disimpan ke hho_30_runs.csv")
    print("="*60)

if __name__ == "__main__":
    main()