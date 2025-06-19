"""
Lin-Kernighan-Helsgaun (KKH/LKH) tabanlı TSP çözücü
Ön-koşul: 1) lkh/LKH ikilisi   veya 2) python-lkh pip paketi
"""

import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import mean
from shutil import which

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


# --------------------------- Yardımcı Fonksiyonlar ----------------------
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def total_distance(tour, cities):
    return sum(
        euclidean_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
        for i in range(len(tour))
    )


def write_tsplib_file(cities, path, name="Problem"):
    """Koordinat listesini temel TSPLIB .tsp formatına yazar."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"NAME: {name}\nTYPE: TSP\nDIMENSION: {len(cities)}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(cities, 1):
            f.write(f"{idx} {int(x)} {int(y)}\n")
        f.write("EOF\n")


# -------------------------- LKH Çözümleyicisi ---------------------------
def solve_with_lkh(cities, runs=1, seed=42, verbose=False):
    """
    LKH ikilisi veya python-lkh paketini kullanarak en iyi turu döndürür.
    Dönen: (tour_list, süre_saniye)
    """
    # --- Geçici çalışma klasörü ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tsp_path = tmp / "instance.tsp"
        write_tsplib_file(cities, tsp_path)

        # Prefer native LKH binary if available
        lkh_bin = which("lkh") or which("LKH")
        t_start = time.perf_counter()

        if lkh_bin:
            # --- Parametre dosyası ---
            par_file = tmp / "params.par"
            tour_file = tmp / "result.tour"
            with open(par_file, "w", encoding="utf-8") as f:
                f.write(
                    f"PROBLEM_FILE = {tsp_path}\n"
                    f"OUTPUT_TOUR_FILE = {tour_file}\n"
                    f"RUNS = {runs}\n"
                    f"SEED = {seed}\n"
                )
            # --- LKH'yi çalıştır ---
            subprocess.run(
                [lkh_bin, str(par_file)],
                check=True,
                stdout=subprocess.DEVNULL if not verbose else None,
            )
            # --- Çıktı turunu oku ---
            tour = []
            with open(tour_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().isdigit():
                        tour.append(int(line.strip()) - 1)  # 0-index
            if len(tour) != len(cities):
                raise RuntimeError("LKH çıktı turu boyutu hatalı.")
        else:
            # ---- python-lkh fallback ----
            try:
                from lkh import LKH
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Ne yerel LKH ikilisi ne de python-lkh paketi bulundu. "
                    "Lütfen `pip install python-lkh` veya LKH C sürümünü kurun."
                ) from e

            solver = LKH(runs=runs, seed=seed, verbose=verbose)
            tour = solver.solve([(int(x), int(y)) for x, y in cities])

        elapsed = time.perf_counter() - t_start
        return tour, elapsed


# ------------------------- Sonuç Görselleştirme --------------------------
def plot_tour(tour, cities, title="En iyi bulunan tur (KKH/LKH)"):
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
    - Kod akışı, önceki GA örneğiyle bire bir aynı:
        1. Veri oku
        2. Çöz
        3. Mesafe, süre, istatistikler
        4. Tur çiz
    """
    tsp_file = "data/tsp_100_1"  # Veri dosyanızın adı
    cities = read_tsp_file(tsp_file)

    print(f"{len(cities)} şehir yüklendi. LKH çözümü başlatılıyor...")
    try:
        tour, elapsed = solve_with_lkh(
            cities,
            runs=5,    # Aynı problemi birkaç kez koşup iyisini alır
            seed=42,
            verbose=False,
        )
    except RuntimeError as err:
        print(err)
        exit(1)

    dist = total_distance(tour, cities)
    print(
        f"En iyi tur mesafesi: {dist:.2f}\n"
        f"Toplam çalışma süresi: {elapsed:.2f} saniye"
    )
    plot_tour(tour, cities)

