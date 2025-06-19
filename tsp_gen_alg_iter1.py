import random
import math
import time
from statistics import mean
import matplotlib.pyplot as plt

# ----------------------------- Veri Girişi -----------------------------
def read_tsp_file(filename):
    """
    İki farklı TSP dosya biçimini destekler:
    1) İlk satırı şehir sayısı, ardından 'x y' koordinatları içeren ham TXT.
    2) TSPLIB biçimindeki NODE_COORD_SECTION ... EOF bloğu.
    Dönen değer -> [(x1, y1), (x2, y2), ...]
    """
    coords = []
    with open(filename, 'r', encoding="utf-8") as file:
        first_line = file.readline().strip()

        # --- Ham TXT biçimi (örnek dosya) ---
        if first_line.isdigit():
            city_count = int(first_line)
            for _ in range(city_count):
                parts = file.readline().strip().split()
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1])))

        # --- TSPLIB biçimi ---
        else:
            lines = [first_line] + file.readlines()
            start = False
            for line in lines:
                if 'NODE_COORD_SECTION' in line:
                    start = True
                    continue
                if 'EOF' in line or not start:
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))

    if not coords:
        raise ValueError(
            "Dosya biçimi tanımlanamadı veya koordinat bulunamadı: "
            f"{filename}"
        )
    return coords

# ----------------------- Yardımcı Matematik Fonksiyonları ----------------
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def total_distance(tour, cities):
    return sum(
        euclidean_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
        for i in range(len(tour))
    )

# ----------------------- Genetik Algoritma Bileşenleri -------------------
def initial_population(size, city_count):
    return [random.sample(range(city_count), city_count) for _ in range(size)]

def fitness(individual, cities):
    # Mesafe küçüldükçe fitness büyür
    return 1.0 / total_distance(individual, cities)

def selection(population, fitnesses):
    # İki ebeveyn; fitness oranında olasılıkla seçilir
    parents = random.choices(population, weights=fitnesses, k=2)
    return parents[0], parents[1]

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end + 1] = parent1[start:end + 1]

    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = gene
    return child

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]

# --------------------------- Ana GA Döngüsü ------------------------------
def genetic_algorithm(
    cities,
    population_size=100,
    generations=500,
    mutation_rate=0.01,
    seed=None,
    log_every=1,
):
    """
    cities         : [(x, y), ...]
    population_size: popülasyon büyüklüğü
    generations    : nesil sayısı
    mutation_rate  : mutasyon olasılığı
    seed           : tekrarlanabilirlik için isteğe bağlı random seed
    log_every      : kaç nesilde bir log basılsın (varsayılan: her nesil)
    """
    if seed is not None:
        random.seed(seed)

    t0 = time.perf_counter()

    population = initial_population(population_size, len(cities))
    distances = [total_distance(ind, cities) for ind in population]
    best_solution = population[distances.index(min(distances))]

    for gen in range(generations):
        # Fitness hesapla
        fitnesses = [1.0 / d for d in distances]

        # Yeni popülasyon oluştur
        new_population = []
        for _ in range(population_size):
            p1, p2 = selection(population, fitnesses)
            child = crossover(p1, p2)
            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        distances = [total_distance(ind, cities) for ind in population]

        # En iyiyi güncelle
        current_best = population[distances.index(min(distances))]
        if total_distance(current_best, cities) < total_distance(
            best_solution, cities
        ):
            best_solution = current_best

        # --- LOG ---
        if gen % log_every == 0:
            print(
                f"Nesil {gen:>4}: "
                f"En İyi = {min(distances):.2f}, "
                f"Ortalama = {mean(distances):.2f}, "
                f"En Kötü = {max(distances):.2f}"
            )

    elapsed = time.perf_counter() - t0
    print(f"\nAlgoritma tamamlandı. Toplam süre: {elapsed:.2f} saniye")
    return best_solution, elapsed

# ------------------------- Sonuç Görselleştirme --------------------------
def plot_tour(tour, cities, title="En iyi bulunan tur"):
    xs = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    ys = [cities[i][1] for i in tour] + [cities[tour[0]][1]]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------- Örnek Çalıştırma ---------------------------
if __name__ == "__main__":
    # TXT biçimindeki örnek problemi okuma
    tsp_file = "data/tsp_100_1"  # Dosyanızın adı
    cities = read_tsp_file(tsp_file)

    best, duration = genetic_algorithm(
        cities,
        population_size=150,
        generations=1000,
        mutation_rate=0.01,
        seed=42,          # İsteğe bağlı
        log_every=10      # İsteğe bağlı: her 10 nesilde bir raporla
    )

    print(f"En iyi tur mesafesi: {total_distance(best, cities):.2f}")
    plot_tour(best, cities)
