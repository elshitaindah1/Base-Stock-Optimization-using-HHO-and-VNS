# Base Stock Optimization using HHO and VNS

## Project Overview

Project ini merupakan implementasi Python dari penelitian skripsi mengenai optimasi kebijakan **continuous-review base stock** pada sistem inventori probabilistik **Assemble-to-Order (ATO)**.

Penelitian menggunakan pendekatan **simulation-optimization** dengan menggabungkan **Monte Carlo Simulation**, **Random Forest sebagai surrogate model**, serta dua algoritma metaheuristik, yaitu **Harris Hawks Optimization (HHO)** dan **Variable Neighborhood Search (VNS)**.

Model yang digunakan terdiri dari 8 produk dan 12 item komponen, dengan 10 key item dan 2 non-key item. Variabel keputusan berupa vektor target inventory level untuk setiap item, sedangkan objective function yang digunakan adalah memaksimalkan ekspektasi total profit per periode.

Penelitian membandingkan empat pendekatan, yaitu HHO Conventional, HHO Surrogate-Assisted, VNS Conventional, dan VNS Surrogate-Assisted. Pada pendekatan conventional, evaluasi kandidat solusi dilakukan menggunakan simulasi Monte Carlo secara langsung. Sementara itu, pendekatan surrogate-assisted menggunakan Random Forest untuk memprediksi expected profit sehingga proses optimasi dapat dilakukan dengan beban komputasi yang lebih rendah.

Setelah proses optimasi, kandidat solusi terbaik diverifikasi menggunakan **Sequential Ranking and Selection (R&S)** dengan simulasi Monte Carlo berakurasi tinggi.

## Research Problem

Optimasi target base stock pada sistem inventori probabilistik memiliki ruang pencarian yang besar dan fungsi objective yang bersifat stochastic. Evaluasi menggunakan simulasi Monte Carlo secara langsung pada setiap iterasi metaheuristik dapat meningkatkan waktu komputasi secara signifikan.

Oleh karena itu, penelitian ini mengevaluasi apakah penggunaan Random Forest sebagai surrogate model dapat meningkatkan efisiensi pencarian serta bagaimana performa HHO dibandingkan VNS.

## Objective

1. Menentukan vektor target inventory level optimal menggunakan HHO dan VNS.
2. Membandingkan pendekatan conventional dan surrogate-assisted.
3. Mengevaluasi trade-off antara kualitas solusi dan efisiensi komputasi.
4. Mengidentifikasi pendekatan yang memberikan keseimbangan terbaik antara profit dan waktu komputasi.

## Methodology

```text
Monte Carlo Simulation
          ↓
Generate Training Dataset
          ↓
Random Forest Surrogate Model
          ↓
HHO / VNS Optimization
          ↓
Candidate Solutions
          ↓
Sequential Ranking & Selection
          ↓
Final Performance Evaluation
```

Monte Carlo Simulation digunakan untuk menghasilkan dataset training dan sebagai evaluasi akhir kandidat solusi. Random Forest digunakan sebagai surrogate fitness function pada proses optimasi, sedangkan HHO dan VNS digunakan untuk mengeksplorasi ruang solusi.

## Experimental Design

* Products: 8
* Components: 12
* Key Items: 10
* Non-Key Items: 2
* Monte Carlo training samples: 2,000
* Training/testing split: 80/20
* Maximum Function Evaluations: 500
* HHO population size: 30
* VNS maximum neighborhood: 5
* Final Monte Carlo replication: 1,000

Dataset hasil simulasi terdiri dari 2.000 kombinasi vektor base stock dan nilai profit per bulan.

## Results

### Algorithm Performance Comparison

Performance evaluation was conducted using four algorithm variants: HHO Conventional, HHO Surrogate-Assisted, VNS Conventional, and VNS Surrogate-Assisted. The comparison considers two criteria: monthly profit and computational time.

| Algorithm Variant      | Profit (USD/month) | Computational Time (s) |
| ---------------------- | -----------------: | ---------------------: |
| HHO Conventional       |             243.86 |                 860.37 |
| HHO Surrogate-Assisted |             225.34 |                  96.93 |
| VNS Conventional       |             205.71 |                 830.38 |
| VNS Surrogate-Assisted |             191.50 |                  92.61 |

### Optimal Target Inventory Level

The optimization process generated a 12-dimensional optimal target inventory level vector for each algorithm variant. Each element represents the target inventory level of an item in the probabilistic Assemble-to-Order (ATO) inventory system.

| Varian Algoritma | x₁ | x₂ | x₃ | x₄ | x₅ | x₆ | x₇ | x₈ | x₉ | x₁₀ | x₁₁ | x₁₂ | Profit (USD/bulan) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HHO Conventional | 2 | 4 | 3 | 3 | 4 | 4 | 3 | 3 | 3 | 4 | 3 | 3 | 243.86 |
| HHO Surrogate-Assisted | 4 | 3 | 6 | 2 | 6 | 7 | 6 | 11 | 5 | 6 | 4 | 6 | 225.34 |
| VNS Conventional | 4 | 6 | 22 | 14 | 15 | 3 | 3 | 16 | 4 | 5 | 4 | 4 | 205.71 |
| VNS Surrogate-Assisted | 15 | 4 | 5 | 2 | 7 | 16 | 23 | 18 | 10 | 3 | 9 | 3 | 191.50 |

### TOPSIS Evaluation

TOPSIS was used to evaluate the overall performance of each algorithm based on profit and computational time. A higher relative closeness value (ξᵢ) indicates a solution that is closer to the positive ideal solution and farther from the negative ideal solution.

| Algorithm Variant          |     Profit |  Time (s) |           d⁺ |           d⁻ |           ξᵢ |
| -------------------------- | ---------: | --------: | -----------: | -----------: | -----------: |
| HHO Conventional           |     243.86 |    860.37 |     420.5195 |     43.80752 |     0.094346 |
| **HHO Surrogate-Assisted** | **225.34** | **96.93** | **15.67457** | **419.1107** | **0.963949** |
| VNS Conventional           |     205.71 |    830.38 |     405.3519 |     20.27725 |     0.047641 |
| VNS Surrogate-Assisted     |     191.50 |     92.61 |     43.80752 |     420.5195 |     0.905654 |

Based on the TOPSIS results, **HHO Surrogate-Assisted achieved the highest relative closeness value (ξᵢ = 0.963949)**. Although HHO Conventional produced the highest profit, it required substantially longer computational time. HHO Surrogate-Assisted achieved a competitive profit with significantly lower computational time, resulting in the best overall trade-off between solution quality and computational efficiency.

### Key Findings

* **HHO Surrogate-Assisted** achieved the highest TOPSIS score of **0.963949**.
* HHO Conventional generated the highest real profit at **243.86 USD/month**, with an optimal target inventory vector of **[2, 4, 3, 3, 4, 4, 3, 3, 3, 4, 3, 3]**.
* HHO Surrogate-Assisted generated a profit of **225.34 USD/month** with an optimal vector of **[4, 3, 6, 2, 6, 7, 6, 11, 5, 6, 4, 6]**.
* Surrogate-assisted approaches substantially reduced computational time compared with their conventional counterparts.
* **HHO Surrogate-Assisted provided the best balance between profitability and computational efficiency**.
* Based on the TOPSIS evaluation, **HHO Surrogate-Assisted was selected as the preferred approach** among the four algorithm variants.


## Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* Joblib
* Matplotlib
* Visual Studio Code

## Repository Structure

```text
data/          → Dataset hasil simulasi
src/           → Implementasi metode dan algoritma
experiments/   → Script eksperimen
models/        → Model Random Forest
results/       → Hasil eksperimen
figures/       → Visualisasi hasil
docs/          → Dokumen penelitian
```

## Reference

Horng, S. C., & Lin, C. Y. (2017). Simulation optimization of inventory systems using metaheuristic and machine learning approaches.

## Author

**Elshita Indah Cahyani**
Industrial Engineering — Universitas Muhammadiyah Malang
