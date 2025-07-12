import os
import re
import random as rd
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

# Number of parallel workers
MAX_WORKERS = 16

# =============================================================================
# Helper for parallel execution
# =============================================================================
def _run_one(args):
    """Helper for parallel execution of a single TS run."""
    category, filepath, tsize, run_id, PENAL_WEIGHT, N_MOVES = args

    # Load dataset
    data = load_dataset(filepath)
    N = data["N"]
    jobs = list(range(1, N + 1))
    machines = [1, 2]
    speeds = [1, 2, 3]
    p = {j + 1: {1: data["p1j"][j], 2: data["p2j"][j]} for j in range(N)}
    v = {i + 1: data["vl"][i] for i in range(len(data["vl"]))}
    conv = {i + 1: data["convl"][i] for i in range(len(data["convl"]))}
    gamma = {i + 1: data["idleconv_i"][i] for i in range(len(data["idleconv_i"]))}
    S1_raw = data["S1"]
    S2_raw = data["S2"]
    rho = data["rho"]

    # Seed variation per run
    seed = int(time.time() * 1e6) % 10000000 + run_id

    # Execute Tabu Search
    ts = FlowshopTS(
        jobs, machines, speeds,
        p, v, conv, gamma, S1_raw, S2_raw, rho,
        seed, tsize, PENAL_WEIGHT, n_moves=N_MOVES
    )

    # Build history DataFrame
    hist_df = pd.DataFrame(ts.history)
    hist_df["filename"] = os.path.basename(filepath)
    hist_df["tabu_size"] = tsize
    hist_df["category"] = category
    hist_df["run_id"] = run_id

    return run_id, ts.best_fitness, hist_df


# =============================================================================
# 1) Parser: load_dataset
# =============================================================================
def load_dataset(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    def parse_array(line):
        return [float(x) if "." in x else int(x)
                for x in re.split(r"[\t ]+", line.split(":")[1].strip())]

    N = M = L = None
    p1j, p2j, vl, convl, idleconv_i, pi_i = [], [], [], [], [], []
    S1, S2 = [], []
    state = None

    for line in lines:
        if line.startswith("N:"):
            N = int(line.split(":")[1].strip())
        elif line.startswith("M:"):
            M = int(line.split(":")[1].strip())
        elif line.startswith("L:"):
            L = int(line.split(":")[1].strip())
        elif line.startswith("p1j:"):
            p1j = parse_array(line)
        elif line.startswith("p2j:"):
            p2j = parse_array(line)
        elif line.startswith("vl:"):
            vl = parse_array(line)
        elif line.startswith("Conv_l:"):
            convl = parse_array(line)
        elif line.startswith("IdleConv_i:"):
            idleconv_i = parse_array(line)
        elif line.startswith("pi_i:"):
            pi_i = parse_array(line)
        elif line.startswith("S1jk:"):
            state = "S1"
        elif line.startswith("S2jk:"):
            state = "S2"
        elif state == "S1":
            if line[0].isdigit() or line[0] == '-':
                S1.append(parse_array(":" + line))
            else:
                state = None
        elif state == "S2":
            if line[0].isdigit() or line[0] == '-':
                S2.append(parse_array(":" + line))
            else:
                state = None

    rho = pi_i if pi_i else [1.0, 1.0]
    if isinstance(rho, (float, int)):
        rho = [rho, rho]
    if len(rho) == 1:
        rho = [rho[0], rho[0]]

    return {
        "N": N, "M": M, "L": L,
        "p1j": p1j, "p2j": p2j,
        "vl": vl, "convl": convl,
        "idleconv_i": idleconv_i,
        "rho": rho,
        "S1": S1, "S2": S2
    }


# =============================================================================
# 2) FlowshopTS class (Tabu Search implementation)
# =============================================================================
class FlowshopTS:
    def __init__(self, jobs, machines, speeds, p, v, conv, gamma,
                 S1_raw, S2_raw, rho, seed, tabu_tenure,
                 penalization_weight, n_moves=30):
        self.jobs = list(jobs)
        self.machines = list(machines)
        self.speeds = speeds
        self.p = p
        self.v = v
        self.conv = conv
        self.gamma = gamma
        self.S1 = S1_raw
        self.S2 = S2_raw
        self.rho = rho
        self.seed = seed
        self.tabu_tenure = tabu_tenure
        self.penalization_weight = penalization_weight
        self.n_moves = n_moves
        self.N = len(self.jobs)
        self.initial_solution = self.get_InitialSolution()
        (self.tabu_str, self.best_solution,
         self.best_objvals, self.best_fitness,
         self.best_iter) = self.TSearch()

    def get_tabuestructure(self):
        job_pairs = list(combinations(self.jobs, 2))
        rd.seed(self.seed + int(time.time() * 1e6) % 1000000)
        sampled_pairs = rd.sample(job_pairs, min(self.n_moves, len(job_pairs)))
        d = {}
        for swap in sampled_pairs:
            d[swap] = {'tabu_time': 0, 'MoveValue': 0,
                       'freq': 0, 'Penalized_MV': 0,
                       'raw_obj': None}
        return d

    def get_InitialSolution(self, show=False):
        sol = self.jobs.copy()
        rd.seed(self.seed)
        rd.shuffle(sol)
        if show:
            print("Initial Random Solution:", sol)
        return sol

    def Objfun(self, sequence, show=False):
        n = len(sequence)
        m = len(self.machines)
        C = [[0 for _ in range(m)] for _ in range(n)]
        proc_time = 0
        busy_time = [0 for _ in range(m)]
        speed_idx = 2

        for i, job in enumerate(sequence):
            for m_idx, mach in enumerate(self.machines):
                if mach == 1:
                    setup = (self.S1[sequence[i - 1] - 1][job - 1]
                             if i > 0 else self.S1[job - 1][job - 1])
                    proc = self.p[job][mach] / self.v[speed_idx]
                    C[i][m_idx] = (C[i - 1][m_idx] if i > 0 else 0) + setup + proc
                    proc_time += proc
                    busy_time[m_idx] += proc
                else:
                    setup = (self.S2[sequence[i - 1] - 1][job - 1]
                             if i > 0 else self.S2[job - 1][job - 1])
                    prevC_job = C[i][m_idx - 1]
                    prevC_mach = C[i - 1][m_idx] if i > 0 else 0
                    proc = self.p[job][mach] / self.v[speed_idx]
                    C[i][m_idx] = max(prevC_job, prevC_mach) + setup + proc
                    proc_time += proc
                    busy_time[m_idx] += proc

        makespan = C[n - 1][m - 1]
        idle_times = [makespan - busy_time[k] for k in range(m)]
        total_idle = sum(idle_times)

        e_proc = 0
        for i, job in enumerate(sequence):
            for idx, mach in enumerate(self.machines):
                proc = (self.p[job][mach] / (60 * self.v[speed_idx])
                        * self.conv[speed_idx] * self.rho[idx])
                e_proc += proc
        e_idle = sum(self.gamma[mach] * self.rho[idx] * (idle_times[idx] / 60)
                     for idx, mach in enumerate(self.machines))
        total_energy = e_proc + e_idle

        if show:
            print(f"Seq: {sequence} | Makespan: {makespan:.2f} | "
                  f"Proc: {proc_time:.2f} | Energy: {total_energy:.2f} | Idle: {total_idle:.2f}")
        return makespan, proc_time, total_energy, total_idle

    def fitness(self, objvals, bestvals):
        return 0.25 * sum(obj / best for obj, best in zip(objvals, bestvals))

    def SwapMove(self, solution, i, j):
        sol = solution.copy()
        idx_i = sol.index(i)
        idx_j = sol.index(j)
        sol[idx_i], sol[idx_j] = sol[idx_j], sol[idx_i]
        return sol

    def TSearch(self):
        best_solution = self.initial_solution.copy()
        best_objvals = self.Objfun(best_solution)
        best_fitness = self.fitness(best_objvals, best_objvals)
        current_solution = self.initial_solution.copy()
        best_iter = 1
        start_time = time.time()
        time_limit = 1.5      # Max allowed time in seconds
        stop_time = 1.3       # Early stopping time in seconds
        max_iter = 3000       # iteration cap

        self.history = [{
            "iteration": 0,
            "time": 0.0,
            "fitness": best_fitness,
            "makespan": best_objvals[0],
            "processing": best_objvals[1],
            "energy": best_objvals[2],
            "idle": best_objvals[3]
        }]

        iter_count = 1
        while (time.time() - start_time) < time_limit and iter_count <= max_iter:
            elapsed = time.time() - start_time
            if elapsed >= stop_time:
                # Early stopping triggered
                break

            tabu_structure = self.get_tabuestructure()

            for move in list(tabu_structure.keys()):
                candidate_solution = self.SwapMove(current_solution, move[0], move[1])
                raw_obj = self.Objfun(candidate_solution)
                mv = sum(raw_obj)
                p_mv = mv + tabu_structure[move]['freq'] * self.penalization_weight
                tabu_structure[move].update({
                    'MoveValue': mv,
                    'Penalized_MV': p_mv,
                    'raw_obj': raw_obj
                })

            while True:
                best_move = min(tabu_structure, key=lambda x: tabu_structure[x]['Penalized_MV'])
                raw_obj = tabu_structure[best_move]['raw_obj']
                current_fitness = self.fitness(raw_obj, best_objvals)
                tabu_time = tabu_structure[best_move]['tabu_time']

                if tabu_time < iter_count or current_fitness < best_fitness:
                    current_solution = self.SwapMove(current_solution, best_move[0], best_move[1])
                    if current_fitness < best_fitness:
                        best_solution = current_solution.copy()
                        best_objvals = raw_obj
                        best_fitness = current_fitness
                        best_iter = iter_count

                    tabu_structure[best_move]['tabu_time'] = iter_count + self.tabu_tenure
                    tabu_structure[best_move]['freq'] += 1

                    elapsed = time.time() - start_time
                    self.history.append({
                        "iteration": iter_count,
                        "time": elapsed,
                        "fitness": best_fitness,
                        "makespan": best_objvals[0],
                        "processing": best_objvals[1],
                        "energy": best_objvals[2],
                        "idle": best_objvals[3]
                    })
                    iter_count += 1
                    break
                else:
                    tabu_structure[best_move]['Penalized_MV'] = float('inf')

        return tabu_structure, best_solution, best_objvals, best_fitness, best_iter


# =============================================================================
# 3) Main: Evaluate, collect, and save all datapoints; plot time-based convergence
# =============================================================================
if __name__ == "__main__":
    DATA_DIR = r"/home/isaaca/Desktop/selected_datasets"

    easy_list = [
        'ps4j2m-setup125_1.31_0.63_63321.txt',
        'ps6j2m-setup99_1.23_0.64_41468.txt',
        'ps6j2m-setup25_1.31_0.63_63321.txt',
        'ps5j2m-setup125_1.25_0.64_33919.txt',
        'ps4j2m-setup50_1.27_0.72_79727.txt'
    ]
    medium_list = [
        'ps20j2m-setup25_1.24_0.56_80879.txt',
        'ps20j2m-setup25_1.25_0.55_38999.txt',
        'ps50j2m-setup99_1.34_0.73_65114.txt',
        'ps50j2m-setup50_1.08_0.59_57394.txt',
        'ps50j2m-setup99_1.35_0.83_93184.txt'
    ]
    hard_list = [
        'ps120j2m-setup125_1.3_0.79_68968.txt',
        'ps80j2m-setup99_1.12_0.71_16039.txt',
        'ps80j2m-setup50_1.28_0.65_49292.txt',
        'ps120j2m-setup25_1.14_0.52_49633.txt',
        'ps120j2m-setup50_1.25_0.74_73248.txt'
    ]

    easy_paths = [os.path.join(DATA_DIR, f) for f in easy_list]
    medium_paths = [os.path.join(DATA_DIR, f) for f in medium_list]
    hard_paths = [os.path.join(DATA_DIR, f) for f in hard_list]

    all_sets = {
        "Easy": easy_paths,
        "Medium": medium_paths,
        "Hard": hard_paths
    }

    tabu_sizes = [5, 10, 15, 20, 25, 30, 35, 40]
    runs_per_setting = 3
    PENAL_WEIGHT = 10
    N_MOVES = 30

    results_df = pd.DataFrame(
        index=["Easy", "Medium", "Hard"],
        columns=tabu_sizes, dtype=float
    )

    best_histories = {
        "Easy": {tsize: [] for tsize in tabu_sizes},
        "Medium": {tsize: [] for tsize in tabu_sizes},
        "Hard": {tsize: [] for tsize in tabu_sizes}
    }

    all_histories = []
    rd.seed(42)

    for category in ["Easy", "Medium", "Hard"]:
        filepaths = all_sets[category]
        category_best_fit_per_tabu = {tsize: [] for tsize in tabu_sizes}

        print(f"\n===== Category: {category} =====")
        for idx, filepath in enumerate(filepaths):
            fname = os.path.basename(filepath)
            if not os.path.exists(filepath):
                print(f"  → WARNING: {fname} not found in {DATA_DIR}. Skipping.")
                continue

            print(f"  » Benchmark [{idx+1}/{len(filepaths)}]: {fname}")

            for tsize in tabu_sizes:
                best_fitness_over_5 = np.inf
                best_history_for_file = None

                args_list = [
                    (category, filepath, tsize, run_id, PENAL_WEIGHT, N_MOVES)
                    for run_id in range(runs_per_setting)
                ]

                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    for run_id, fitness_val, hist_df in executor.map(_run_one, args_list):
                        all_histories.append(hist_df)
                        if fitness_val < best_fitness_over_5:
                            best_fitness_over_5 = fitness_val
                            best_history_for_file = hist_df

                category_best_fit_per_tabu[tsize].append(best_fitness_over_5)
                best_histories[category][tsize].append(best_history_for_file)
                print(f"      → tabu_size={tsize:2d}, best-of-{runs_per_setting} fitness={best_fitness_over_5:.6f}")

        for tsize in tabu_sizes:
            vals = category_best_fit_per_tabu[tsize]
            results_df.at[category, tsize] = float(np.mean(vals)) if vals else np.nan

        print(f"→ Completed {category}. Partial table:\n{results_df.loc[category]}\n")

    # === Final results and convergence plotting ===
    results_df = results_df.round(6)
    print("===== FINAL BEST-OF-5 FITNESS TABLE =====")
    print(results_df)

    out_csv = os.path.join(DATA_DIR, "tabu122_size_vs_best_fitness.csv")
    results_df.to_csv(out_csv, index=True)
    print(f"\nSaved table to: {out_csv}")

    histories_csv = os.path.join(DATA_DIR, "overall2_tabu_histories.csv")
    all_hist_df = pd.concat(all_histories, ignore_index=True)
    all_hist_df.to_csv(histories_csv, index=False)
    print(f"Saved all run histories to: {histories_csv}")

    # ----------------------------------------------------------------------------
    # 4) Convergence Plots
    # ----------------------------------------------------------------------------
    best_tabu_per_category = results_df.idxmin(axis=1)
    plt.figure(figsize=(10, 6))
    colors = {"Easy": "tab:blue", "Medium": "tab:green", "Hard": "tab:red"}

    # Fixed time‐limit and grid
    time_limit = 2.0
    t_grid = np.linspace(0, time_limit, 250)

    for category in ["Easy", "Medium", "Hard"]:
        best_tsize = int(best_tabu_per_category[category])
        hist_list = best_histories[category][best_tsize]
        if not hist_list:
            continue

        all_curves = []
        for df in hist_list:
            times = df["time"].tolist()
            fitnesses = df["fitness"].tolist()
            if times[-1] < time_limit:
                times.append(time_limit)
                fitnesses.append(fitnesses[-1])
            interp_fitness = np.interp(t_grid, times, fitnesses)
            all_curves.append(np.minimum.accumulate(interp_fitness))

        mean_curve = np.mean(all_curves, axis=0)
        plt.plot(
            t_grid,
            mean_curve,
            label=f"{category} (tabu={best_tsize})",
            linewidth=2,
            color=colors[category]
        )
    plt.axvline(x= 0.8, color='black', linestyle='--', linewidth=1.0, label="t = 0.8s", ymax=0.8)
    plt.xlabel("Time (seconds)", fontsize=13)
    plt.ylabel("Fitness", fontsize=13)
    plt.title("Tabu Search Convergence (Average Best-So-Far Fitness vs Time)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{DATA_DIR}/Tabu_convergence.pdf')
    plt.show()
