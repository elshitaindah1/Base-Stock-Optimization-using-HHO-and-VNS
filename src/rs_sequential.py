import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import math
import time
import joblib

# ============================================================================
# Load surrogate model (Random Forest)
# ============================================================================
def load_rf_model(path="best_random_forest_surrogate.joblib"):
    return joblib.load(path)

try:
    rf_model = load_rf_model()
except:
    rf_model = None
    print("[WARNING] File 'best_random_forest_surrogate.joblib' tidak ditemukan, prediksi surrogate dilewati.")

def predict_profit(x):
    """Prediksi profit menggunakan surrogate model"""
    if rf_model is None:
        return 0.0
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    return float(rf_model.predict(x)[0])

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
# Monte Carlo Simulation (REAL simulation)
# ============================================================================
def simulate_ato(x, L=1000):
    """Melakukan simulasi Monte Carlo sistem ATO sebanyak L replikasi."""
    x = np.asarray(x, dtype=int)
    total_profit = 0.0
    
    for _ in range(L):
        stock = x.copy().astype(float)
        time_now = 0.0
        sales_profit = 0.0
        holding_cost = 0.0
        
        next_arrival = []
        for i in range(8):
            rate = ARRIVAL_RATES[i]
            interarrival = np.random.exponential(1/rate) + np.random.exponential(1/rate)
            next_arrival.append(interarrival)
        
        production_events = [[] for _ in range(12)]
        
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
            else:
                if production_events[prod_item_idx]:
                    production_events[prod_item_idx].remove(next_prod_time)
                    stock[prod_item_idx] = min(stock[prod_item_idx] + 1, CAPACITY_PER_ITEM[prod_item_idx])
        
        profit_repl = (sales_profit - holding_cost) / (SIMULATION_HORIZON - WARM_UP_PERIOD)
        total_profit += profit_repl
    
    return total_profit / L

def simulate_ato_real(x, L, verbose=False):
    return simulate_ato(x, L=L)

# ============================================================================
# Sequential Ranking & Selection (R&S)
# ============================================================================
def sequential_ranking_selection(candidates, L0=100, e=2.718, Le=1000, N_min=2, 
                                 simulate_func=None, verbose=True):
    if simulate_func is None:
        raise ValueError("simulate_func harus disediakan")
    
    candidates = [np.array(c) for c in candidates]
    current_list = candidates.copy()
    history = {'stage_results': []}
    stage = 1
    
    while len(current_list) > N_min:
        L_stage = min(round(L0 * (e ** (stage - 1))), Le)
        
        if verbose:
            print(f"  -> Substage {stage} | Aktif: {len(current_list)} solusi | Replikasi: {L_stage}")
        
        profits = []
        for idx, x in enumerate(current_list):
            profit_est = simulate_func(x, L_stage)
            profits.append(profit_est)
        
        sorted_indices = np.argsort(profits)[::-1]
        sorted_candidates = [current_list[i] for i in sorted_indices]
        sorted_profits = [profits[i] for i in sorted_indices]
        
        N_next = max(round(len(current_list) / e), N_min)
        current_list = sorted_candidates[:N_next]
        
        history['stage_results'].append({
            'stage': stage,
            'L': L_stage,
            'N_before': len(sorted_candidates),
            'N_after': len(current_list),
            'top_profits': sorted_profits[:5]
        })
        
        stage += 1
        if L_stage >= Le:
            break
    
    if verbose:
        print(f"  -> Final Validation Stage (Menguji {len(current_list)} solusi terbaik dengan L = {Le})...")
    
    final_profits = []
    for idx, x in enumerate(current_list):
        profit_final = simulate_func(x, Le)
        final_profits.append(profit_final)
        if verbose:
            print(f"     Kandidat Akhir {idx+1} {x.tolist()}: Profit Riil (L=1000) = {profit_final:.4f}")
    
    best_idx = np.argmax(final_profits)
    x_best = current_list[best_idx]
    profit_best = final_profits[best_idx]
    
    history['final_candidates'] = current_list
    history['final_profits'] = final_profits
    history['best_index'] = best_idx
    
    return x_best, profit_best, history

# ============================================================================
# MAIN PROGRAM
# ============================================================================
def main():
    print("="*75)
    print("   PROSEDUR VALIDASI R&S SEPARASI: KELOMPOK HHO VS KELOMPOK VNS")
    print("="*75)
    
    TOP_K = 5
    
    # 1. Ambil TOP kandidat dari HHO
    print("\n[STEP 1] Memuat Top 5 Hasil HHO dari hho_30_runs.csv...")
    try:
        hho_results = pd.read_csv('hho_30_runs.csv')
        hho_results_sorted = hho_results.sort_values('profit_predicted', ascending=False)
        hho_top_x = []
        for i in range(min(TOP_K, len(hho_results_sorted))):
            x_str = hho_results_sorted.iloc[i]['x_best']
            x_vec = [int(float(v)) for v in x_str.split()]
            hho_top_x.append(x_vec)
            print(f"   HHO Rank {i+1} (Run {hho_results_sorted.iloc[i]['run']}): profit_pred = {hho_results_sorted.iloc[i]['profit_predicted']:.2f}")
    except Exception as e:
        print(f"   [ERROR] Gagal memuat hho_30_runs.csv: {e}")
        return

    # 2. Ambil TOP kandidat dari VNS
    print("\n[STEP 2] Memuat Top 5 Hasil VNS dari vns_30_runs.csv...")
    try:
        vns_results = pd.read_csv('vns_30_runs.csv')
        vns_results_sorted = vns_results.sort_values('profit_predicted', ascending=False)
        vns_top_x = []
        for i in range(min(TOP_K, len(vns_results_sorted))):
            x_str = vns_results_sorted.iloc[i]['x_best']
            x_vec = [int(float(v)) for v in x_str.split()]
            vns_top_x.append(x_vec)
            print(f"   VNS Rank {i+1} (Run {vns_results_sorted.iloc[i]['run']}): profit_pred = {vns_results_sorted.iloc[i]['profit_predicted']:.2f}")
    except Exception as e:
        print(f"   [ERROR] Gagal memuat vns_30_runs.csv: {e}")
        return
    
    # 3. Jalankan Arena R&S untuk Kelompok HHO
    print("\n[STEP 3] Menjalankan Prosedur R&S untuk Kelompok HHO...")
    t_start_hho = time.perf_counter()
    best_x_hho, best_profit_hho, _ = sequential_ranking_selection(
        candidates=hho_top_x, L0=100, e=2.718, Le=1000, N_min=2,
        simulate_func=simulate_ato_real, verbose=True
    )
    t_hho = time.perf_counter() - t_start_hho
    print(f"-> Selesai dalam {t_hho:.2f} detik.")
    
    # 4. Jalankan Arena R&S untuk Kelompok VNS
    print("\n[STEP 4] Menjalankan Prosedur R&S untuk Kelompok VNS...")
    t_start_vns = time.perf_counter()
    best_x_vns, best_profit_vns, _ = sequential_ranking_selection(
        candidates=vns_top_x, L0=100, e=2.718, Le=1000, N_min=2,
        simulate_func=simulate_ato_real, verbose=True
    )
    t_vns = time.perf_counter() - t_start_vns
    print(f"-> Selesai dalam {t_vns:.2f} detik.")
    
    # 5. RINGKASAN AKHIR & KESIMPULAN JAWARA
    print("\n" + "="*75)
    print("           REKAPITULASI HASIL TERBAIK VALIDASI R&S")
    print("="*75)
    print(f"  * JAWARA VALID HHO : Profit Riil (L=1000) = {best_profit_hho:.4f}")
    print(f"                       Kombinasi Base Stock = {best_x_hho.tolist()}")
    print("-"*75)
    print(f"  * JAWARA VALID VNS : Profit Riil (L=1000) = {best_profit_vns:.4f}")
    print(f"                       Kombinasi Base Stock = {best_x_vns.tolist()}")
    print("="*75)
    
    if best_profit_hho > best_profit_vns:
        diff = best_profit_hho - best_profit_vns
        print(f"  Kesimpulan Uji Validasi: HHO lebih unggul sebesar {diff:.4f} unit moneter.")
    else:
        diff = vns_profit_best - hho_profit_best
        print(f"  Kesimpulan Uji Validasi: VNS lebih unggul sebesar {diff:.4f} unit moneter.")
        
    print("  Kedua konfigurasi di atas siap digunakan sebagai input pada Bab Uji Durabilitas.")
    
    # Simpan laporan final ke CSV untuk kebutuhan lampiran tabel skripsi
    try:
        summary_data = [
            {
                'Algoritma': 'Harris Hawks Optimization (HHO)',
                'Vektor_Base_Stock_Optimal': ' '.join(map(str, best_x_hho.tolist())),
                'Profit_Riil_Valid_L1000': best_profit_hho,
                'Waktu_R_S_Detik': t_hho
            },
            {
                'Algoritma': 'Variable Neighborhood Search (VNS)',
                'Vektor_Base_Stock_Optimal': ' '.join(map(str, best_x_vns.tolist())),
                'Profit_Riil_Valid_L1000': best_profit_vns,
                'Waktu_R_S_Detik': t_vns
            }
        ]
        df_out = pd.DataFrame(summary_data)
        df_out.to_csv('hho_vns_rs_separasi_summary.csv', index=False)
        print("\n[SUCCESS] Laporan rekapitulasi disimpan di 'hho_vns_rs_separasi_summary.csv'")
    except Exception as e:
        print(f"   [WARNING] Gagal menyimpan file laporan komparasi: {e}")

if __name__ == "__main__":
    main()
    