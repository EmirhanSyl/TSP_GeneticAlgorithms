#!/usr/bin/env python3
# ga_tsp_clean.py  –  GA for TSP with bullet-proof distance evaluation
import argparse, math, random, sys, time
from pathlib import Path
from typing import List, Tuple

# ─────────── data loader (simple TXT only: first line = n, then n x-y lines) ─────
def load_coords(path: Path) -> List[Tuple[float, float]]:
    lines = path.read_text().strip().split()
    n = int(lines[0])
    coords = list(zip(map(float, lines[1::2]), map(float, lines[2::2])))[:n]
    if len(coords) != n:
        raise ValueError(f"{path}: expected {n} coords, got {len(coords)}")
    return coords

# ─────────── helpers ────────────────────────────────────────────────────────────
def euclid(p, q) -> float:                     # one edge
    return math.hypot(p[0]-q[0], p[1]-q[1])

def tour_len(tour: List[int], pts: List[Tuple[float,float]]) -> float:
    d = 0.0
    for i in range(len(tour)):
        d += euclid(pts[tour[i]], pts[tour[(i+1)%len(tour)]])
    return d                                     # no NumPy, no broadcast, ever

def crossover(p1: List[int], p2: List[int]) -> List[int]:
    n=len(p1); a,b=sorted(random.sample(range(n),2))
    child=[-1]*n; child[a:b]=p1[a:b]
    pos=0
    for g in p2:
        if g not in child:
            while child[pos]!=-1: pos+=1
            child[pos]=g
    return child

def mutate(tour: List[int], rate: float):
    if random.random()<rate:
        i,j=random.sample(range(len(tour)),2)
        tour[i],tour[j]=tour[j],tour[i]

# 2-opt local search
def two_opt(tour, pts):
    improved=True
    while improved:
        improved=False
        for i in range(1,len(tour)-2):
            for j in range(i+1,len(tour)):
                if j-i==1: continue
                a,b = tour[i-1],tour[i]
                c,d = tour[j-1],tour[j%len(tour)]
                if euclid(pts[a],pts[b]) + euclid(pts[c],pts[d]) > \
                   euclid(pts[a],pts[c]) + euclid(pts[b],pts[d]):
                    tour[i:j] = reversed(tour[i:j])
                    improved=True

# ─────────── GA main ────────────────────────────────────────────────────────────
def ga(coords, pop=200, gens=3000, mut=0.02, two_opt_flag=False, log=0):
    pts = coords
    n   = len(pts)
    P   = [random.sample(range(n), n) for _ in range(pop)]
    F   = [tour_len(t, pts) for t in P]
    best = min(F)

    for g in range(gens):
        # --- log
        if log and (g % log == 0 or g == gens-1):
            print(f"Gen {g:4d}: best={min(F):.2f}  avg={sum(F)/pop:.2f}  worst={max(F):.2f}")

        # --- elitism
        idx = min(range(pop), key=F.__getitem__)
        elite = P[idx][:]

        # --- create next generation
        newP=[elite]
        while len(newP)<pop:
            p1,p2=random.sample(P,2)
            child=crossover(p1,p2)
            mutate(child, mut)
            if two_opt_flag: two_opt(child, pts)
            newP.append(child)

        P=newP
        F=[tour_len(t, pts) for t in P]

    idx=min(range(pop), key=F.__getitem__)
    return P[idx], F[idx]



# ─────────── CLI ────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("file",type=Path)
    ap.add_argument("--pop",type=int,default=200)
    ap.add_argument("--gens",type=int,default=3000)
    ap.add_argument("--mut",type=float,default=0.02)
    ap.add_argument("--twoopt",action="store_true")
    ap.add_argument("--log",type=int,default=0,help="print stats every N gens")
    ap.add_argument("--seed",type=int)
    args=ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    coords = load_coords(args.file)
    t0=time.perf_counter()
    tour,best=ga(coords,args.pop,args.gens,args.mut,args.twoopt,args.log)
    print(f"\nFinal best length = {best:.3f}  (time {time.perf_counter()-t0:.2f}s)")

# if __name__=="__main__":
#     main()


def grid_search_tsp(
    filepath: str,
    param_grid: dict,
    repeat: int = 1,         # run each combo 'repeat' times
    csv_path: str = "grid_results.csv",
) -> None:
    """
    Run GA for every parameter combo and save results to CSV.

    Columns: pop, gens, mut, run, best_len, seconds
    """
    coords = load_coords(Path(filepath))
    rows = [("pop", "gens", "mut", "run", "best_len", "seconds")]

    combos = [
        (pop, gen, mut)
        for pop in param_grid["population_size"]
        for gen in param_grid["generations"]
        for mut in param_grid["mutation_rate"]
    ]

    total = len(combos) * repeat
    print(f"Running {total} GA runs …")

    run_idx = 1
    for pop, gens, mut in combos:
        for r in range(1, repeat + 1):
            t0 = time.perf_counter()
            _, best = ga(coords, pop, gens, mut,
                         two_opt_flag=True,   # keep if you want 2-opt
                         log=0)
            dt = time.perf_counter() - t0
            rows.append((pop, gens, mut, r, round(best, 3), round(dt, 2)))
            print(f"[{run_idx:>3}/{total}] pop={pop:<3} gens={gens:<4}"
                  f" mut={mut:<4}  ->  best={best:.3f}  ({dt:.2f}s)")
            run_idx += 1

    # write CSV
    with open(csv_path, "w", newline="") as fh:
        for row in rows:
            fh.write(",".join(map(str, row)) + "\n")
    print(f"\nResults written to {csv_path}")


PARAM_GRID = {
    "population_size": [50, 100, 150],
    "generations":     [300, 600],
    "mutation_rate":   [0.01, 0.05],
}

if __name__ == "__main__":
    grid_search_tsp(
        filepath="/content/drive/MyDrive/Colab/tsp_100_1",   # <- your instance
        param_grid=PARAM_GRID,
        repeat=3,                    # 3 independent runs per combo
        csv_path="/content/drive/MyDrive/Colab/grid_results.csv",
    )