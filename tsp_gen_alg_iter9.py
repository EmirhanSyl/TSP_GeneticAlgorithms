"""
Simulated Annealing (SA) tabanlı TSP çözücü
Yapı, önceki GA ve LKH örnekleriyle uyumludur.
"""

import math
import random
import time
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

# ----------------------------- Veri Girişi -----------------------------
def read_tsp_file(filename):
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


# ----------------------- Yardımcı Fonksiyonlar -------------------------
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def total_distance(tour, cities):
    return sum(
        euclidean_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
        for i in range(len(tour))
    )


def two_opt_swap(tour, i, k):
    """Turun [i,k] aralığını ters çevirir (2-opt komşusu)."""
    return tour[:i] + tour[i : k + 1][::-1] + tour[k + 1 :]


# ----------------------- Simulated Annealing ---------------------------
def simulated_annealing(
    cities,
    T0=10_000.0,
    alpha=0.995,
    stop_T=1e-3,
    max_iter=100_000,
    seed=None,
    log_every=1_000,
):
    """
    Temel SA algoritması.
    Dönen → (en_iyi_tur, geçen_süre_s)
    """
    if seed is not None:
        random.seed(seed)

    n = len(cities)
    # Başlangıç turu: 0,1,2,…,n-1
    current = list(range(n))
    best = current[:]
    best_len = total_distance(best, cities)

    T = T0
    t_start = time.perf_counter()

    lengths = []  # istatistik için

    for iteration in range(1, max_iter + 1):
        # --- Komşu üret ---
        i, k = sorted(random.sample(range(1, n), 2))
        neighbor = two_opt_swap(current, i, k)

        delta = total_distance(neighbor, cities) - total_distance(current, cities)
        # --- Kabul ölçütü ---
        if delta < 0 or math.exp(-delta / T) > random.random():
            current = neighbor

        # En iyiyi güncelle
        curr_len = total_distance(current, cities)
        if curr_len < best_len:
            best, best_len = current[:], curr_len

        lengths.append(curr_len)

        # --- Günlük ---
        if iteration % log_every == 0:
            print(
                f"İter {iteration:>6} | T={T:8.2f} | "
                f"En İyi={best_len:8.2f} | "
                f"Ortalama={mean(lengths[-log_every:]):8.2f} | "
                f"Güncel={curr_len:8.2f}"
            )

        # Sıcaklığı azalt
        T *= alpha
        if T < stop_T:
            break

    elapsed = time.perf_counter() - t_start
    print(f"\nAlgoritma tamamlandı. Toplam süre: {elapsed:.2f} saniye")
    return best, elapsed


# ------------------------- Sonuç Görselleştirme -------------------------
def plot_tour(tour, cities, title="En iyi bulunan tur (Simulated Annealing)"):
    xs = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    ys = [cities[i][1] for i in tour] + [cities[tour[0]][1]]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------- Örnek Çalıştırma ---------------------------
if __name__ == "__main__":
    """
    1. Veri oku
    2. Simulated Annealing ile çöz
    3. Mesafe, süre raporla
    4. Tur çiz
    """
    tsp_file = "data/tsp_100_1"  # Dosyanızın adı
    cities = read_tsp_file(tsp_file)

    print(f"{len(cities)} şehir yüklendi. Simulated Annealing başlatılıyor...")
    best_tour, duration = simulated_annealing(
        cities,
        T0=10_000,
        alpha=0.995,
        stop_T=1e-3,
        max_iter=50_000,
        seed=42,
        log_every=1_000,
    )

    dist = total_distance(best_tour, cities)
    print(f"En iyi tur mesafesi: {dist:.2f}")
    plot_tour(best_tour, cities)