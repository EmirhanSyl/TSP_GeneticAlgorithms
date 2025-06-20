import csv, itertools, time, argparse
from pathlib import Path
import pandas as pd

from algorithms.tsp_gen_alg_iter2_memetic import solve_tsp_ga


# ---------- Parametre ızgarası ----------
GRID = {
    "pop_size":      [50, 100, 150],
    "max_gen":       [300, 600],
    "mutation_base": [0.10, 0.20],
}

REPEAT      = 3
BASE_SEED   = 2024
CSV_OUTPUT  = "ga_results_70.csv"


def run_experiments(filepath: str):
    combos = list(itertools.product(*GRID.values()))
    keys   = list(GRID.keys())
    total  = len(combos) * REPEAT
    rows   = []
    run_id = 0

    for values in combos:
        params = dict(zip(keys, values))
        for rep in range(REPEAT):
            run_id += 1
            seed = BASE_SEED + run_id
            print(f"[{run_id:>3}/{total}]  {params} | rep={rep+1}")

            best, elapsed = solve_tsp_ga(
                filepath,
                **params,
                elite_count=1,
                seed=seed,
                verbose=False,
                log_interval=params["max_gen"] + 1
            )
            rows.append({
                **params,
                "repeat":        rep + 1,
                "best_distance": best,
                "exec_time":     elapsed,
            })
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV → {path}")


def show_summary(rows):
    df = pd.DataFrame(rows)
    agg = (df.groupby(["pop_size", "max_gen", "mutation_base"])
             .agg(best_mean=("best_distance", "mean"),
                  best_std =("best_distance", "std"),
                  time_mean=("exec_time", "mean"))
             .reset_index())
    print("\n=========== ÖZET ===========")
    print(agg.to_string(index=False, formatters={
        "best_mean": "{:.1f}".format,
        "best_std":  "{:.1f}".format,
        "time_mean": "{:.2f}".format,
    }))


def main():
    ap = argparse.ArgumentParser(description="GA parametre tarayıcısı")
    input_file = "../data/tsp_70_1"
    ap.add_argument("--csv", default=CSV_OUTPUT,
                    help=f"Çıktı CSV (varsayılan {CSV_OUTPUT})")
    args = ap.parse_args()

    if not Path(input_file).exists():
        ap.error("Girdi dosyası bulunamadı!")

    t0 = time.perf_counter()
    rows = run_experiments(input_file)
    save_csv(rows, args.csv)
    show_summary(rows)
    print(f"\nTüm deneyler tamamlandı → {time.perf_counter() - t0:.1f} sn")


if __name__ == "__main__":
    main()
