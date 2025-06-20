#!/usr/bin/env python3
# tsp_ga.py
# ------------------------------------------------------------
# Genetik Algoritma ile TSP (PMX + 2-opt + adaptif mutasyon)
# Parametreleştirilebilir sürüm – en iyi mesafe & süre döndürür
# ------------------------------------------------------------
from __future__ import annotations

import sys
import math
import random
import time
from pathlib import Path
from typing import List, Tuple


# ─────────────────────  Yardımcı fonksiyon  ──────────────────────
def _log(msg: str, *, verbose: bool) -> None:
    if verbose:
        print(msg, file=sys.stderr)


# ─────────────────────  Ana çözüm fonksiyonu  ──────────────────────
def solve_tsp_ga(
    filepath: str = "input.txt",
    *,
    pop_size: int = 50,
    max_gen: int = 500,
    mutation_base: float = 0.20,
    elite_count: int = 1,
    seed: int | None = None,
    verbose: bool = True,
    log_interval: int = 10,
) -> Tuple[float, float]:
    """
    GA tabanlı TSP çözücüsü.

    Dönüş:
        best_distance (float),
        elapsed_time (float, saniye)
    """
    if seed is not None:
        random.seed(seed)

    t0 = time.perf_counter()

    # 1. Veri okuma ---------------------------------------------------
    file = Path(filepath)
    if not file.exists():
        raise FileNotFoundError(filepath)

    _log(f"[1/8] '{filepath}' okunuyor…", verbose=verbose)
    tokens = file.read_text().strip().split()
    if not tokens:
        raise ValueError("Girdi dosyası boş.")

    N = int(tokens[0])
    expected = 1 + 2 * N
    if len(tokens) < expected:
        raise ValueError(
            f"Girdi hatalı: {N} şehir için {2*N} koordinat gerekir, "
            f"ancak {len(tokens)-1} değer bulundu."
        )
    coords = list(map(float, tokens[1:expected]))
    cities: List[Tuple[float, float]] = list(zip(coords[::2], coords[1::2]))
    _log(f"--> {N} şehir yüklendi.", verbose=verbose)

    # 2. Mesafe matrisi ----------------------------------------------
    _log("[2/8] Mesafe matrisi hesaplanıyor…", verbose=verbose)
    dist = [[0] * N for _ in range(N)]
    for i in range(N):
        x1, y1 = cities[i]
        for j in range(i + 1, N):
            x2, y2 = cities[j]
            dij = int(math.floor(math.hypot(x1 - x2, y1 - y2) + 0.5))
            dist[i][j] = dist[j][i] = dij
    _log("--> Mesafe matrisi hazır.", verbose=verbose)

    # 3. Başlangıç popülasyonu ---------------------------------------
    _log("[3/8] Başlangıç popülasyonu…", verbose=verbose)

    def nearest_neighbor(start: int = 0) -> List[int]:
        unvisited = set(range(N))
        tour = [start]
        unvisited.remove(start)
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda j: dist[cur][j])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour

    population: List[List[int]] = [nearest_neighbor()] if N else []
    while len(population) < pop_size:
        p = list(range(N))
        random.shuffle(p)
        population.append(p)
    _log(f"--> Popülasyon boyutu: {pop_size}", verbose=verbose)

    # 4. Yardımcılar ---------------------------------------------------
    def route_len(tour: List[int]) -> int:
        return sum(dist[tour[i]][tour[(i + 1) % N]] for i in range(N))

    def tournament() -> int:
        k = 3
        idxs = random.sample(range(len(population)), k)
        return min(idxs, key=lambda idx: fitness[idx])  # index

    def pmx(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        a, b = sorted(random.sample(range(N), 2))
        c1, c2 = [-1] * N, [-1] * N
        c1[a : b + 1] = p1[a : b + 1]
        c2[a : b + 1] = p2[a : b + 1]

        def fill(child, other):
            for i in range(a, b + 1):
                gene = other[i]
                if gene not in child:
                    pos = i
                    while child[pos] != -1:
                        gene_in_child = child[pos]
                        pos = other.index(gene_in_child)
                    child[pos] = gene

        fill(c1, p2)
        fill(c2, p1)
        for i in range(N):
            if c1[i] == -1:
                c1[i] = p2[i]
            if c2[i] == -1:
                c2[i] = p1[i]
        return c1, c2

    def mutate(route: List[int]) -> None:
        i, j = random.sample(range(N), 2)
        route[i], route[j] = route[j], route[i]

    def two_opt(route: List[int]) -> int:
        improved, d_tot = True, route_len(route)
        while improved:
            improved = False
            for i in range(1, N - 1):
                for j in range(i + 2, N):
                    a, b = route[i - 1], route[i]
                    c, d = route[j], route[(j + 1) % N]
                    if c == route[0] and d == route[-1]:
                        continue
                    old = dist[a][b] + dist[c][d]
                    new = dist[a][c] + dist[b][d]
                    if new < old:
                        route[i : j + 1] = reversed(route[i : j + 1])
                        d_tot += new - old
                        improved = True
                        break
                if improved:
                    break
        return d_tot

    # 5. İlk değerlendirme --------------------------------------------
    fitness = [1.0 / route_len(t) for t in population]
    best_idx = max(range(pop_size), key=lambda i: fitness[i])
    best_tour = population[best_idx][:]
    best_dist = 1.0 / fitness[best_idx]
    _log(f"[4/8] İlk en iyi mesafe = {best_dist}", verbose=verbose)

    # 6. GA döngüsü ----------------------------------------------------
    mut_prob = mutation_base
    stagnation = 0
    for gen in range(1, max_gen + 1):
        new_pop, new_fit = [], []

        # Elit
        for _ in range(elite_count):
            new_pop.append(best_tour[:])
            new_fit.append(1.0 / best_dist)

        # Çocuklar
        while len(new_pop) < pop_size:
            p1, p2 = population[tournament()], population[tournament()]
            c1, c2 = pmx(p1, p2)
            if random.random() < mut_prob:
                mutate(c1)
            if random.random() < mut_prob:
                mutate(c2)
            d1, d2 = two_opt(c1), two_opt(c2)
            new_pop.extend([c1, c2])
            new_fit.extend([1.0 / d1, 1.0 / d2])

        # Popülasyon güncelle
        population, fitness = new_pop[:pop_size], new_fit[:pop_size]

        # En iyi güncelle
        curr_idx = max(range(pop_size), key=lambda i: fitness[i])
        curr_dist = 1.0 / fitness[curr_idx]
        if curr_dist < best_dist:
            best_dist = curr_dist
            best_tour = population[curr_idx][:]
            mut_prob = mutation_base
            stagnation = 0
            _log(f"[Gen {gen}] İyileşme! → {best_dist}", verbose=verbose)
        else:
            stagnation += 1
            if stagnation == 5:
                mut_prob = min(1.0, mutation_base * 2)
            elif stagnation == 20:
                mut_prob = min(1.0, mutation_base * 4)

        if gen % log_interval == 0:
            _log(f"[Gen {gen}] En iyi mesafe: {best_dist}", verbose=verbose)

    # 7. Sonuç ---------------------------------------------------------
    print(f"{best_dist} 0")
    print(" ".join(map(str, best_tour)))
    elapsed = time.perf_counter() - t0
    _log(f"Toplam süre: {elapsed:.2f} sn", verbose=verbose)
    return best_dist, elapsed


# ───── CLI ‐ tek başına çalıştırıldığında ─────
if __name__ == "__main__":
    # Varsayılan: data/tsp_100_1 gibi bir dosya verin
    solve_tsp_ga("../data/tsp_100_1")
