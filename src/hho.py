"""
Harris Hawks Optimization (HHO) Algorithm - Standard Version (Fixed for NumPy v2.0+)
Implements Full HHO (4 Exploitation Phases & Levy Flight) for Maximization Problems.
Modified for Integer/Combinatorial Optimization (Base Stock Policy).
"""

import numpy as np
import math  # Tambahan pustaka bawaan untuk memperbaiki error np.math
from typing import Callable, Tuple, List


def levy_flight(dim: int) -> np.ndarray:
    """Fungsi pembantu untuk menghasilkan langkah acak berbasis Levy Flight."""
    beta = 1.5
    # Perbaikan: Menggunakan math.gamma bawaan Python, bukan np.math.gamma
    num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)
    sigma = (num / den)**(1 / beta)
    
    u = np.random.normal(0, sigma, size=dim)
    v = np.random.normal(0, 1, size=dim)
    step = u / (np.abs(v)**(1 / beta))
    return step


def hho(
    objective_func: Callable,
    max_FE: int,
    n_pop: int,
    lb: List[int],
    ub: List[int],
    seed: int = None,
) -> Tuple[np.ndarray, float, int, List[Tuple[int, float]]]:
    
    if seed is not None:
        np.random.seed(seed)
    
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    dim = len(lb)
    
    # Estimasi jumlah iterasi maksimum berdasarkan total FE budget
    max_iter = max_FE // n_pop
    if max_iter == 0:
        max_iter = 1
    
    # 1. Inisialisasi Populasi Hawk (Solusi Awal)
    X = np.random.uniform(lb, ub, (n_pop, dim))
    X = np.round(X).astype(int)  # Sesuai untuk base stock policy (integer)
    
    # Evaluasi nilai fitness populasi awal
    F = np.array([objective_func(x) for x in X])
    n_FE = n_pop
    
    # Cari solusi terbaik awal (Maksimisasi Nilai Profit)
    best_idx = np.argmax(F)
    x_best = X[best_idx].copy()
    f_best = F[best_idx]
    
    history = [(0, f_best)]
    iteration = 0
    
    # Loop Utama Algoritma
    while n_FE < max_FE:
        iteration += 1
        
        # Hitung rata-rata posisi populasi (X_m)
        X_m = np.mean(X, axis=0)
        
        # Amankan agar perhitungan energi tidak minus ekstrem jika iterasi melebihi batas
        current_t = min(iteration, max_iter)
        
        for i in range(n_pop):
            if n_FE >= max_FE:
                break
                
            # Update Nilai Escape Energy (E) Mangsa
            E0 = 2 * np.random.uniform() - 1  # Range [-1, 1]
            E = 2 * E0 * (1 - (current_t / max_iter))
            
            # --- FASE 1: EKSPLORASI (|E| >= 1) ---
            if np.abs(E) >= 1:
                q = np.random.uniform()
                if q >= 0.5:
                    # Strategi bertengger berdasarkan posisi hawk acak lainnya
                    r_idx = np.random.randint(0, n_pop)
                    X_rand = X[r_idx].copy()
                    x_new = X_rand - np.random.uniform() * np.abs(X_rand - 2 * np.random.uniform() * X[i])
                else:
                    # Strategi bertengger berdasarkan pohon acak & rata-rata populasi
                    x_new = (x_best - X_m) - np.random.uniform() * (lb + np.random.uniform() * (ub - lb))
            
            # --- FASE 2: EKSPLOITASI (|E| < 1) ---
            else:
                r = np.random.uniform()            # Peluang mangsa berhasil kabur
                J = 2 * (1 - np.random.uniform())  # Kekuatan lompatan mangsa (Jumping strength)
                
                # Keadaan 1: Soft Besiege (|E| >= 0.5 dan r >= 0.5)
                if np.abs(E) >= 0.5 and r >= 0.5:
                    delta_X = x_best - X[i]
                    x_new = delta_X - E * np.abs(J * x_best - X[i])
                
                # Keadaan 2: Hard Besiege (|E| < 0.5 dan r >= 0.5)
                elif np.abs(E) < 0.5 and r >= 0.5:
                    delta_X = x_best - X[i]
                    x_new = x_best - E * np.abs(delta_X)
                
                # Keadaan 3: Soft Besiege dengan PRD / Levy Flight (|E| >= 0.5 dan r < 0.5)
                elif np.abs(E) >= 0.5 and r < 0.5:
                    Y = x_best - E * np.abs(J * x_best - X[i])
                    Y = np.clip(Y, lb, ub)
                    Y = np.round(Y).astype(int)
                    
                    Z = Y + np.random.uniform(size=dim) * levy_flight(dim)
                    Z = np.clip(Z, lb, ub)
                    Z = np.round(Z).astype(int)
                    
                    # Evaluasi kandidat Y dan Z (Memakai 2 kuota FE)
                    f_Y = objective_func(Y)
                    f_Z = objective_func(Z)
                    n_FE += 2
                    
                    if f_Y > F[i] or f_Z > F[i]:
                        x_new = Y if f_Y > f_Z else Z
                    else:
                        x_new = X[i].copy()
                
                # Keadaan 4: Hard Besiege dengan PRD / Levy Flight (|E| < 0.5 dan r < 0.5)
                elif np.abs(E) < 0.5 and r < 0.5:
                    Y = x_best - E * np.abs(J * x_best - X_m)
                    Y = np.clip(Y, lb, ub)
                    Y = np.round(Y).astype(int)
                    
                    Z = Y + np.random.uniform(size=dim) * levy_flight(dim)
                    Z = np.clip(Z, lb, ub)
                    Z = np.round(Z).astype(int)
                    
                    f_Y = objective_func(Y)
                    f_Z = objective_func(Z)
                    n_FE += 2
                    
                    if f_Y > F[i] or f_Z > f_Z: # Perbaikan logika pembanding
                        x_new = Y if f_Y > f_Z else Z
                    else:
                        x_new = X[i].copy()

            # Post-processing posisi akhir (pembatasan bounds & pemetaan integer)
            x_new = np.clip(x_new, lb, ub)
            x_new = np.round(x_new).astype(int)
            
            # Evaluasi fungsi jika posisi belum dihitung di Keadaan 3 & 4
            if not (np.abs(E) < 1 and r < 0.5):
                f_new = objective_func(x_new)
                n_FE += 1
                
                # Kebijakan Seleksi Alami (Greedy Selection)
                if f_new > F[i]:
                    X[i] = x_new
                    F[i] = f_new
            else:
                # Untuk Keadaan 3 & 4, fitness x_new sudah dievaluasi di dalam blok kondisional sebelumnya
                f_new = objective_func(x_new)
                if f_new > F[i]:
                    X[i] = x_new
                    F[i] = f_new

            # Perbarui solusi terbaik global sepanjang masa
            if F[i] > f_best:
                x_best = X[i].copy()
                f_best = F[i]
                
        # Catat histori konvergensi
        history.append((iteration, f_best))
        
        if n_FE >= max_FE:
            break
            
    return x_best, f_best, n_FE, history


if __name__ == "__main__":
    # Pengujian sederhana menggunakan fungsi Sphere (dimaksimalkan ke arah 0)
    def sphere(x):
        return -np.sum(x**2)
    
    x_best, f_best, n_FE, history = hho(
        sphere,
        max_FE=1000,
        n_pop=30,
        lb=[0]*5,
        ub=[30]*5,
        seed=42
    )
    print(f"Hasil Terbaik Koordinat X: {x_best}")
    print(f"Nilai Fitness Terbaik: {f_best}")
    print(f"Total Evaluasi Fungsi Terpakai (FE): {n_FE}")