#!/usr/bin/env python
# tsp_ga.py
# Genetik Algoritma ile TSP – ayrıntılı log + süre ölçümü

import sys
import math
import random
import time
from pathlib import Path
from typing import List, Tuple


# ─────────────────────  Yardımcı fonksiyonlar  ──────────────────────
def log(msg: str, *, verbose: bool) -> None:
    """İstenirse stderr’e mesaj basar."""
    if verbose:
        print(msg, file=sys.stderr)


# ─────────────────────  Ana çözüm fonksiyonu  ──────────────────────
def solve_tsp_ga(
    filepath: str = "input.txt",
    *,
    verbose: bool = True,
    log_interval: int = 10,
) -> None:
    """Genetik Algoritma ile TSP çözümü (PMX + adaptif mutasyon + 2-opt)."""
    # 0. Süre ölçümünü başlat
    t0 = time.perf_counter()

    # 1. Girdiyi oku
    file = Path(filepath)
    if not file.exists():
        print(f"Hata: '{filepath}' bulunamadı.", file=sys.stderr)
        sys.exit(1)

    log(f"[1/8] '{filepath}' okunuyor…", verbose=verbose)
    tokens = file.read_text().strip().split()
    if not tokens:
        print("Girdi dosyası boş.", file=sys.stderr)
        sys.exit(1)

    try:
        N = int(tokens[0])
    except ValueError:
        print("İlk değer şehir sayısı olmalı.", file=sys.stderr)
        sys.exit(1)

    expected = 1 + 2 * N
    if len(tokens) < expected:
        print(
            f"Girdi hatalı: {N} şehir için {2 * N} koordinat gerekir,"
            f" ancak {len(tokens)-1} değer bulundu.",
            file=sys.stderr,
        )
        sys.exit(1)

    coords = list(map(float, tokens[1:expected]))
    cities: List[Tuple[float, float]] = list(zip(coords[::2], coords[1::2]))
    log(f"--> {N} şehir yüklendi.", verbose=verbose)

    # 2. Mesafe matrisi
    log("[2/8] Mesafe matrisi hesaplanıyor…", verbose=verbose)
    dist = [[0] * N for _ in range(N)]
    for i in range(N):
        x1, y1 = cities[i]
        for j in range(i + 1, N):
            x2, y2 = cities[j]
            d = math.hypot(x1 - x2, y1 - y2)
            dij = int(math.floor(d + 0.5))
            dist[i][j] = dij
            dist[j][i] = dij
    log("--> Mesafe matrisi hazır.", verbose=verbose)

    # 3. GA parametreleri
    POP_SIZE = 50
    ELITE_COUNT = 1
    MAX_GEN = 500
    base_mut = 0.2
    mut_prob = base_mut
    no_improve = 0

    # 4. Başlangıç popülasyonu
    log("[3/8] Başlangıç popülasyonu oluşturuluyor…", verbose=verbose)
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
    while len(population) < POP_SIZE:
        t = list(range(N))
        random.shuffle(t)
        population.append(t)
    log(f"--> Popülasyon boyutu: {POP_SIZE}", verbose=verbose)

    # 5. Yardımcılar
    def route_distance(tour: List[int]) -> int:
        return sum(dist[tour[i]][tour[(i + 1) % N]] for i in range(N))

    def tournament_select() -> List[int]:
        k = 3
        idxs = random.sample(range(len(population)), k)
        return min(idxs, key=lambda idx: distances[idx])  # index döner

    def pmx_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        a, b = sorted(random.sample(range(N), 2))
        c1, c2 = [-1] * N, [-1] * N
        c1[a : b + 1] = p1[a : b + 1]
        c2[a : b + 1] = p2[a : b + 1]

        def pmx_fill(child, other):
            for i in range(a, b + 1):
                gene = other[i]
                if gene not in child:
                    pos = i
                    while child[pos] != -1:
                        gene_in_child = child[pos]
                        pos = other.index(gene_in_child)
                    child[pos] = gene

        pmx_fill(c1, p2)
        pmx_fill(c2, p1)

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
        improved = True
        d_total = route_distance(route)
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
                        d_total += new - old
                        improved = True
                        break
                if improved:
                    break
        return d_total

    # 6. Başlangıç değerlendirme
    distances = [route_distance(t) for t in population]
    best_idx = min(range(len(distances)), key=distances.__getitem__)
    best_tour = population[best_idx][:]
    best_dist = distances[best_idx]
    log(f"[4/8] GA başlıyor: İlk en iyi mesafe = {best_dist}", verbose=verbose)

    # 7. GA ana döngüsü
    for gen in range(1, MAX_GEN + 1):
        new_pop, new_dist = [], []

        # 7.1 Elit
        new_pop.append(best_tour[:])
        new_dist.append(best_dist)

        # 7.2 Çocuk üretimi
        while len(new_pop) < POP_SIZE:
            p1 = population[tournament_select()]
            p2 = population[tournament_select()]
            c1, c2 = pmx_crossover(p1, p2)

            if random.random() < mut_prob:
                mutate(c1)
            if random.random() < mut_prob:
                mutate(c2)

            d1 = two_opt(c1)
            d2 = two_opt(c2)

            new_pop.extend([c1, c2])
            new_dist.extend([d1, d2])

        # 7.3 Güncelle
        population = new_pop[:POP_SIZE]
        distances = new_dist[:POP_SIZE]

        # 7.4 En iyiyi güncelle
        curr_idx = min(range(len(distances)), key=distances.__getitem__)
        if distances[curr_idx] < best_dist:
            best_dist = distances[curr_idx]
            best_tour = population[curr_idx][:]
            no_improve = 0
            mut_prob = base_mut
            log(f"[Gen {gen}] İyileşme! Yeni en iyi = {best_dist}", verbose=verbose)
        else:
            no_improve += 1
            if no_improve == 5:
                mut_prob = min(1.0, base_mut * 2)
            elif no_improve == 20:
                mut_prob = min(1.0, base_mut * 4)

        # 7.5 Periyodik rapor
        if gen % log_interval == 0:
            log(f"[Gen {gen}] En iyi mesafe: {best_dist}", verbose=verbose)

    # 8. Sonuç yazdır
    log("[8/8] Çalışma tamamlandı, sonuç yazdırılıyor…", verbose=verbose)
    print(f"{best_dist} 0")
    print(" ".join(map(str, best_tour)))

    elapsed = time.perf_counter() - t0
    log(f"Toplam süre: {elapsed:.2f} saniye", verbose=verbose)


# ─────────────────────  Komut satırı arabirimi  ──────────────────────
if __name__ == "__main__":
    solve_tsp_ga("data/tsp_100_1")
