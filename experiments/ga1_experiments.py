import csv
import time
import itertools
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------
# VAR OLAN KODUNUZDAN İÇE AKTARILANLAR
#   - read_tsp_file
#   - total_distance
#   - genetic_algorithm
# NOT: Bu fonksiyonları "ga_core.py" gibi ayrı dosyaya koyup import
# ederseniz temiz olur. Örnek amaçlı doğrudan buraya kopyalayabilirsiniz.
# --------------------------------------------------------------
from algorithms.tsp_gen_alg_iter1_baseline import (
    read_tsp_file,
    total_distance,
    genetic_algorithm,
)

# --------------------------------------------------------------
# 1) Deney parametre ızgarasını tanımlayın
# --------------------------------------------------------------
PARAM_GRID = {
    "population_size": [50, 100, 150],
    "generations":     [300, 600],
    "mutation_rate":   [0.01, 0.05],
}

REPEATS_PER_CONFIG = 3          # Her parametre setini kaç kez koşalım?
RANDOM_SEED_BASE    = 1234      # Tekrarlanabilirlik için
RESULTS_CSV         = "ga_experiments.csv"

# --------------------------------------------------------------
def run_experiments(cities):
    """Tüm kombinasyonları deneyip sonuçları döndürür."""
    keys, values = zip(*PARAM_GRID.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    total_runs = len(configs) * REPEATS_PER_CONFIG
    run_no = 0

    for cfg in configs:
        for rep in range(REPEATS_PER_CONFIG):
            run_no += 1
            seed = RANDOM_SEED_BASE + run_no         # her koşu için farklı seed
            print(f"[{run_no:>3}/{total_runs}] Çalışıyor: {cfg}, tekrar={rep+1}")

            best_route, elapsed = genetic_algorithm(
                cities,
                population_size=cfg["population_size"],
                generations=cfg["generations"],
                mutation_rate=cfg["mutation_rate"],
                seed=seed,
                log_every=cfg["generations"] + 1,    # log kapalı
            )

            best_len = total_distance(best_route, cities)
            results.append({
                **cfg,
                "repeat":        rep + 1,
                "best_distance": best_len,
                "exec_time":     elapsed,
            })

    return results


def save_results_csv(results, path):
    """Sonuçları CSV olarak kaydeder."""
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSonuçlar kaydedildi → {path}")


def print_summary(results):
    """Aynı parametre kombinasyonlarının ortalama performansını özetler."""
    df = pd.DataFrame(results)
    agg = (
        df.groupby(["population_size", "generations", "mutation_rate"])
          .agg(best_mean=("best_distance", "mean"),
               best_std=("best_distance", "std"),
               time_mean=("exec_time", "mean"))
          .reset_index()
    )
    print("\n=========== ÖZET TABLO ===========")
    print(agg.to_string(index=False, formatters={
        "best_mean": "{:.2f}".format,
        "best_std":  "{:.2f}".format,
        "time_mean": "{:.2f}".format,
    }))

    # Graphs
    fig, ax = plt.subplots()
    ax.scatter(agg["time_mean"], agg["best_mean"])
    for _, row in agg.iterrows():
        label = f"P{row.population_size}|G{row.generations}|M{row.mutation_rate}"
        ax.annotate(label, (row.time_mean, row.best_mean))
    ax.set_xlabel("Ortalama Süre (sn)")
    ax.set_ylabel("Ortalama En İyi Mesafe")
    ax.set_title("GA Parametre Karşılaştırması")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GA parametre taraması ve karşılaştırma")

    input_file = "../data/tsp_100_1"
    parser.add_argument("--csv", default=RESULTS_CSV,
                        help=f"Çıktı CSV (varsayılan: {RESULTS_CSV})")
    args = parser.parse_args()

    if not Path(input_file).exists():
        parser.error(f"{input_file} bulunamadı!")

    cities = read_tsp_file(input_file)

    t0 = time.perf_counter()
    results = run_experiments(cities)
    save_results_csv(results, args.csv)
    print_summary(results)
    print(f"\nToplam deneme süresi: {time.perf_counter() - t0:.1f} sn")


if __name__ == "__main__":
    main()
