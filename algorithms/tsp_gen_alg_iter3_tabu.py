"""
Tabu Search (TS) tabanlı TSP çözücü + Parametre Tarayıcı
-------------------------------------------------------
 • Ham TXT veya TSPLIB dosyasını okur.
 • Verilen parametre ızgarasını (GA→TS eşleştirmesi) dolaşarak
   her kombinasyonu REPEATS kez çalıştırır.
 • En iyi/ortalama mesafe ve süre istatistiklerini CSV'ye kaydeder.
"""

# ─────────────────────────── KÜTÜPHANELER ──────────────────────────────
import math
import random
import time
import itertools
import csv
from collections import deque
from statistics import mean
from pathlib import Path
import matplotlib.pyplot as plt

# ───────────────────────── VERİ OKUMA FONKSİYONU ────────────────────────
def read_tsp_file(filename: str):
    """Ham TXT (ilk satır şehir sayısı) veya TSPLIB dosyasını koordinat listesine dönüştürür."""
    coords = []
    with open(filename, "r", encoding="utf-8") as file:
        first = file.readline().strip()
        if first.isdigit():  # Ham TXT
            for _ in range(int(first)):
                x, y = map(float, file.readline().split()[:2])
                coords.append((x, y))
        else:  # TSPLIB
            lines = [first] + file.readlines()
            start = False
            for line in lines:
                if "NODE_COORD_SECTION" in line:
                    start = True
                    continue
                if not start or "EOF" in line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    if not coords:
        raise ValueError("Dosya biçimi tanınmadı / koordinat bulunamadı.")
    return coords

# ───────────────────────── YARDIMCI HESAPLAR ───────────────────────────
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def total_distance(tour, cities):
    return sum(
        euclidean_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
        for i in range(len(tour))
    )

def two_opt_swap(tour, i, k):
    """[i, k] aralığını ters çevirerek 2-opt komşusu üretir."""
    return tour[:i] + tour[i : k + 1][::-1] + tour[k + 1 :]

def nearest_neighbour_start(cities):
    """Yakın-komşu sezgiseli başlangıç turu."""
    n = len(cities)
    unvisited = set(range(1, n))
    tour = [0]
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda j: euclidean_distance(cities[last], cities[j]))
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour

# ─────────────────────────── TABU SEARCH ───────────────────────────────
def tabu_search(
    cities,
    tabu_size=50,
    max_iter=10_000,
    neighbor_sample=200,
    aspiration=True,
    seed=None,
    log_every=500,
):
    """
    Basit 2-opt Tabu Search.
    Dönen: (en_iyi_tur, geçen_süre_s)
    """
    if seed is not None:
        random.seed(seed)

    n = len(cities)
    current = nearest_neighbour_start(cities)
    best = current[:]
    best_len = total_distance(best, cities)

    tabu_list = deque(maxlen=tabu_size)
    t_start = time.perf_counter()
    lengths = []

    for iteration in range(1, max_iter + 1):
        candidate = None
        candidate_len = float("inf")
        move_chosen = None

        # ─ Komşuluk örneklemesi ─
        for _ in range(neighbor_sample):
            i, k = sorted(random.sample(range(1, n), 2))
            if (i, k) in tabu_list and not aspiration:
                continue
            neighbor = two_opt_swap(current, i, k)
            dist = total_distance(neighbor, cities)
            if dist < candidate_len or (
                aspiration and dist < best_len and (i, k) in tabu_list
            ):
                candidate, candidate_len, move_chosen = neighbor, dist, (i, k)

        if candidate is None:
            break  # Tüm hamleler tabu ve aspirasyon devre dışıysa

        current = candidate
        tabu_list.append(move_chosen)

        if candidate_len < best_len:
            best, best_len = candidate[:], candidate_len

        lengths.append(candidate_len)

        if iteration % log_every == 0:
            print(
                f"İter {iteration:>6} | "
                f"En İyi={best_len:8.2f} | "
                f"Ortalama={mean(lengths[-log_every:]):8.2f} | "
                f"Güncel={candidate_len:8.2f} | "
                f"Tabu={len(tabu_list)}"
            )

    elapsed = time.perf_counter() - t_start
    return best, elapsed

# ───────────────────────── PARAMETRE IZGARASI ──────────────────────────
# GA odaklı grid, TS parametrelerine eşlendi
PARAM_GRID_TS = {
    "neighbor_sample": [50, 100, 150],   # GA'daki population_size
    "max_iter":        [300, 600],       # GA'daki generations
    "tabu_size":       [1, 5],           # GA'daki mutation_rate yüzdeleri → küçük tamsayı
}
REPEATS   = 5
SEED_BASE = 2025
TSP_FILE  = "data/tsp_100_1"        # Probleminiz
OUT_CSV   = "ts_param_test_results.csv"

# ───────────────────────── GRID-SEARCH FONKSİYONLARI ───────────────────
def run_grid_ts():
    cities = read_tsp_file(TSP_FILE)
    results = []

    for ns, mi, ts in itertools.product(
        PARAM_GRID_TS["neighbor_sample"],
        PARAM_GRID_TS["max_iter"],
        PARAM_GRID_TS["tabu_size"]
    ):
        print(f"\n==> neighbor_sample={ns}, max_iter={mi}, tabu_size={ts}")
        best_run_dist = float("inf")
        dists, times  = [], []

        for r in range(REPEATS):
            seed = SEED_BASE + r
            t0   = time.perf_counter()
            best_tour, _ = tabu_search(
                cities,
                tabu_size=ts,
                max_iter=mi,
                neighbor_sample=ns,
                seed=seed,
                log_every=mi + 1,      # sessiz
            )
            elapsed = time.perf_counter() - t0

            dist = total_distance(best_tour, cities)
            dists.append(dist)
            times.append(elapsed)
            best_run_dist = min(best_run_dist, dist)

            print(f"  ▸ repeat {r+1}/{REPEATS}: dist={dist:.2f}, time={elapsed:.2f}s")

        results.append({
            "neighbor_sample": ns,
            "max_iter":        mi,
            "tabu_size":       ts,
            "best_dist":       best_run_dist,
            "mean_dist":       sum(dists)/REPEATS,
            "std_dist":        (sum((d - sum(dists)/REPEATS)**2 for d in dists)/(REPEATS-1))**0.5,
            "mean_time_s":     sum(times)/REPEATS,
        })
    return results

def save_results_csv(rows, out_path=OUT_CSV):
    fieldnames = [
        "neighbor_sample", "max_iter", "tabu_size",
        "best_dist", "mean_dist", "std_dist", "mean_time_s",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n📄 Sonuçlar '{out_path}' dosyasına kaydedildi.")

# ───────────────────────────────── MAIN ─────────────────────────────────
if __name__ == "__main__":
    all_results = run_grid_ts()
    save_results_csv(all_results)
