# TSP_GeneticAlgorithms
Heuristic Algorithms

# TSP / TCP Heuristic Suite 🚚✨
A collection of **five iterative meta-heuristics** (GA → Memetic GA → Tabu Search → GA + 2-opt → Simulated Annealing) for solving Euclidean Travelling-Salesperson / Courier instances of 5 – 100 cities.

---

## Repository Map
```

emirhansyl-tsp\_geneticalgorithms/
├── algorithms/             ← standalone solvers (Iter-1 … Iter-5)
├── data/                   ← three benchmark instances (5, 70, 100)
├── experiments/            ← grid-search & plotting scripts
├── experiment\_results/     ← CSVs produced by the experiments
└── README.md               ← (this file)

````

| Iter | File | Heuristic | Key extras |
|------|------|-----------|------------|
| 1 | `tsp_gen_alg_iter1_baseline.py` | **Baseline GA** | roulette selection, order CX |
| 2 | `tsp_gen_alg_iter2_memetic.py` | **Memetic GA** | elitism + PMX + 2-opt + adaptive mutation |
| 3 | `tsp_gen_alg_iter3_tabu.py` | **Tabu Search** | sampled 2-opt, aspiration rule |
| 4 | `tsp_gen_alg_iter4_twopt.py` | **GA + full 2-opt** | large grid search, CLI flags |
| 5 | `tsp_gen_alg_iter5_annualing.py` | **Simulated Annealing** | cooling schedule, 2-opt neighbours |

---

## Quick Start

```bash
# 0) create & activate a virtual-env (optional)
python -m venv venv
source venv/bin/activate

# 1) install minimal deps
pip install matplotlib pandas

# 2) solve the 100-city instance with the memetic GA (Iter-2)
python algorithms/tsp_gen_alg_iter2_memetic.py data/tsp_100_1 \
       --pop_size 100 --max_gen 300 --mutation_base 0.1 --verbose

# 3) launch a full grid search for the 70-city case (Iter-4)
python algorithms/tsp_gen_alg_iter4_twopt.py \
       --file data/tsp_70_1 --twoopt --pop 100 --gens 300 --mut 0.05
````

### Reproduce Published Experiments

All CSVs in `experiment_results/` were generated with the scripts in `experiments/`.

```bash
# baseline GA grid on 100-city
python experiments/ga1_experiments.py --csv experiment_results/ga1_experiments_100.csv

# memetic GA grid on 70-city
python experiments/ga2_experiments.py --csv experiment_results/ga2_experiments_70.csv
```

Jupyter notebook `experiments/comperison_graphs.ipynb` renders the scatter plots used in the report.

---

## Data Format

Plain text, first line = **N**, followed by *x y* pairs (one-based city indices are implied):

```
5
0   0
0   0.5
0   1
1   1
1   0
```

Distance = ⌊√((Δx)²+(Δy)²)+0.5⌋ (TSPLIB `EUC_2D`).

---

## Results Snapshot

| Instance | Best mean tour ↑ | Method              | Runtime (s, mean) |
| -------- | ---------------: | ------------------- | ----------------: |
| 5-city   |            **5** | reached by all      |            < 0.25 |
| 70-city  |        **665.4** | GA + 2-opt (Iter-4) |               1.4 |
| 100-city |       **21 285** | GA + 2-opt (Iter-4) |               508 |

*Tabu Search is ≈ 5 % longer on average but 4-5× faster; details in `/experiment_results/`.*

---

## Project Notes

* **Modularity** – every solver is import-safe and offers a CLI; you can mix-and-match pieces in new experiments.
* **Reproducibility** – all scripts accept `--seed` and fixed grids; CSVs capture every trial.
* **Visualisation** – each algorithm ships with `plot_tour()` for quick visual sanity checks.
* **Extensibility** – add constraints (time windows, capacities) by swapping the distance / feasibility functions without touching GA scaffolding.

---

Enjoy routing! 🚀

```
