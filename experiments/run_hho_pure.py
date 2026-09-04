"""
run_hho_pure.py
Menjalankan HHO PURE (tanpa surrogate model) 30 kali dengan MAX_FE = 500, L_SIM = 50.
Fitness function langsung memanggil simulasi Monte Carlo.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
from hho import hho

# ============================================================================
# ATO System Parameters (sama dengan parameter di jurnal Horng & Lin 2017)
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

# BOM (Bill of Materials) dari Tabel 4.3 skripsi (8 produk x 12 item)
BOM = [
    [0,1,0,0,1,0,1,0,1,1,0,0],  # produk 1
    [1,1,0,1,0,0,1,1,0,1,0,0],  # produk 2
    [0,1,0,0,0,1,0,0,1,1,1,1],  # produk 3
    [0,1,1,0,1,0,0,1,0,1,0,1],  # produk 4
    [1,1,1,1,0,0,1,0,1,0,1,0],  # produk 5
    [0,1,0,0,1,1,0,1,0,1,0,1],  # produk 6
    [0,1,1,0,0,1,1,0,1,0,1,0],  # produk 7
    [1,1,0,1,1,0,0,1,0,1,0,1],  # produk 8
]

# ============================================================================
# Simulasi Monte Carlo untuk sistem ATO
# ============================================================================
def simulate_ato(x, L=1000):
    """
    Melakukan simulasi Monte Carlo sistem ATO sebanyak L replikasi.
    x: vektor target inventory level (list atau array 12 integer)
    L: jumlah replikasi simulasi
    Returns: estimated expected profit per unit time
    """
    x = np.asarray(x, dtype=int)
    total_profit = 0.0
    
    for _ in range(L):
        # Inisialisasi stok awal = target inventory level
        stock = x.copy().astype(float)
        
        # Waktu simulasi
        time_now = 0.0
        # Profit dari penjualan
        sales_profit = 0.0
        # Akumulasi holding cost
        holding_cost = 0.0
        
        # Event times untuk kedatangan pesanan setiap produk
        next_arrival = []
        for i in range(8):
            rate = ARRIVAL_RATES[i]
            interarrival = np.random.exponential(1/rate) + np.random.exponential(1/rate)
            next_arrival.append(interarrival)
        
        # Event times untuk selesainya produksi (replenishment)
        production_events = [[] for _ in range(12)]
        
        # Loop utama event-driven simulation
        while time_now < SIMULATION_HORIZON:
            next_arrival_time = min(next_arrival) if next_arrival else float('inf')
            next_prod_time = float('inf')
            prod_item_idx = -1
            for j in range(12):
                if production_events[j]:
                    t = min(production_events[j])
                    if t < next_prod_time:
                        next_prod_time = t
                        prod_item_idx = j
            
            event_time = min(next_arrival_time, next_prod_time)
            if event_time >= SIMULATION_HORIZON:
                event_time = SIMULATION_HORIZON
            
            if time_now >= WARM_UP_PERIOD:
                delta_t = event_time - time_now
                for j in range(12):
                    holding_cost += stock[j] * HOLDING_COST_PER_ITEM[j] * delta_t
            
            time_now = event_time
            if time_now >= SIMULATION_HORIZON:
                break
            
            if event_time == next_arrival_time:
                product_idx = next_arrival.index(min(next_arrival))
                interarrival = np.random.exponential(1/ARRIVAL_RATES[product_idx]) + \
                               np.random.exponential(1/ARRIVAL_RATES[product_idx])
                next_arrival[product_idx] = time_now + interarrival
                
                key_items_available = True
                for j in range(10):
                    if BOM[product_idx][j] == 1 and stock[j] < 1:
                        key_items_available = False
                        break
                
                if key_items_available:
                    for j in range(12):
                        if BOM[product_idx][j] == 1 and stock[j] >= 1:
                            stock[j] -= 1
                            if time_now >= WARM_UP_PERIOD:
                                sales_profit += PROFIT_PER_ITEM[j]
                            prod_time = np.random.normal(PRODUCTION_MEANS[j], PRODUCTION_STDS[j])
                            prod_time = max(prod_time, 0)
                            finish_time = time_now + prod_time
                            production_events[j].append(finish_time)
            else:  # event_time == next_prod_time
                if production_events[prod_item_idx]:
                    production_events[prod_item_idx].remove(next_prod_time)
                    stock[prod_item_idx] = min(stock[prod_item_idx] + 1, CAPACITY_PER_ITEM[prod_item_idx])
        
        profit_repl = (sales_profit - holding_cost) / (SIMULATION_HORIZON - WARM_UP_PERIOD)
        total_profit += profit_repl
    
    return total_profit / L

# ============================================================================
# HHO Pure Parameters (Fair Comparison: MAX_FE = 500, L_SIM = 50)
# ============================================================================
HHO_POP_SIZE = 30
MAX_FE = 500           # Sama dengan HHO improved
L_SIM = 50             # Replikasi per evaluasi (cukup stabil, waktu masih manageable)
N_RUNS = 10
START_SEED = 121

# ============================================================================
# Fungsi objektif untuk HHO PURE
# ============================================================================
class PureObjectiveWrapper:
    def __init__(self, L_sim):
        self.L_sim = L_sim
        self.n_FE = 0
    def reset(self):
        self.n_FE = 0
    def __call__(self, x):
        self.n_FE += 1
        x = np.asarray(x, dtype=int).tolist()
        profit = simulate_ato(x, L=self.L_sim)
        return profit

# ============================================================================
# Main
# ============================================================================
def main():
    print("="*60)
    print(f"HHO PURE (tanpa surrogate model) - FAIR COMPARISON")
    print(f"  Populasi = {HHO_POP_SIZE}")
    print(f"  Max_FE = {MAX_FE} (sama dengan HHO improved)")
    print(f"  L_sim per evaluasi = {L_SIM}")
    print(f"  Total replikasi per run = {MAX_FE * L_SIM}")
    print(f"  Runs = {N_RUNS}, seeds {START_SEED}..{START_SEED+N_RUNS-1}")
    print("="*60)
    
    lb = [0]*12
    ub = [30]*12
    results = []
    
    for i in range(N_RUNS):
        seed = START_SEED + i
        print(f"Run {i+1}/{N_RUNS} (seed={seed})...", end=" ", flush=True)
        
        obj = PureObjectiveWrapper(L_sim=L_SIM)
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
        
        # Verifikasi profit riil dengan L=1000 (sama seperti HHO improved)
        profit_real = simulate_ato(x_best, L=1000)
        
        print(f"profit_during={f_best:.2f}, profit_real={profit_real:.2f}, FE={n_fe}, time={t_elapsed:.2f}s")
        
        results.append({
            'run': i+1,
            'seed': seed,
            'x_best': ' '.join(map(str, x_best)),
            'profit_during_search': f_best,
            'profit_real': profit_real,
            'n_FE_used': n_fe,
            'time_seconds': t_elapsed
        })
        
        pd.DataFrame(results).to_csv('hho_pure_30_runs.csv', index=False)
    
    print("HHO PURE 30 runs selesai.")
    print("   Hasil disimpan ke 'hho_pure_30_runs.csv'")
    print("="*60)

if __name__ == "__main__":
    main()