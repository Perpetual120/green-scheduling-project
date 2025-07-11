import os
import re
import random as rd
import time
import glob
import csv
from itertools import combinations

# --- Data Loader ---
def load_dataset(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    def parse_array(line):
        # Remove name, then split on whitespace, convert to float or int
        return [float(x) if "." in x else int(x) for x in re.split(r"[\t ]+", line.split(":")[1].strip())]

    N, M, L = None, None, None
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
                S1.append(parse_array(":"+line))
            else:
                state = None
        elif state == "S2":
            if line[0].isdigit() or line[0] == '-':
                S2.append(parse_array(":"+line))
            else:
                state = None

    # Handle pi_i (rho): default to [1, 1] if missing
    rho = pi_i if pi_i else [1.0, 1.0]
    # Some datasets use single value
    if isinstance(rho, float) or isinstance(rho, int):
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

# --- Tabu Search Class ---
class FlowshopTS():
    def __init__(self, jobs, machines, speeds, p, v, conv, gamma, S1_raw, S2_raw, rho, seed, tabu_tenure, penalization_weight, n_moves=30):
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
        self.n_moves = n_moves # number of random moves for neighborhood
        self.N = len(self.jobs)
        self.history = []
        self.initial_solution = self.get_InitialSolution()
        self.tabu_str, self.best_solution, self.best_objvals, self.best_fitness, self.best_iter = self.TSearch()

    def get_tabuestructure(self):
        # Use a random sample of possible swaps for each iteration
        job_pairs = list(combinations(self.jobs, 2))
        rd.seed(self.seed)
        sampled_pairs = rd.sample(job_pairs, min(self.n_moves, len(job_pairs)))
        d = {}
        for swap in sampled_pairs:
            d[swap] = {'tabu_time': 0, 'MoveValue': 0, 'freq': 0, 'Penalized_MV': 0, 'raw_obj': None}
        return d

    def get_InitialSolution(self, show=False):
        initial_solution = self.jobs.copy()
        rd.seed(self.seed)
        rd.shuffle(initial_solution)
        if show:
            print("Initial Random Solution:", initial_solution)
        return initial_solution

    def Objfun(self, sequence, show=False):
        n = len(sequence)
        m = len(self.machines)
        C = [[0 for _ in self.machines] for _ in range(n)]
        proc_time = 0
        busy_time = [0 for _ in self.machines]
        speed_idx = 2  # Use normal speed (v=1.0, index 2)

        for i, job in enumerate(sequence):
            for m_idx, m in enumerate(self.machines):
                if m == 1:
                    setup = self.S1[sequence[i-1]-1][job-1] if i > 0 else self.S1[job-1][job-1]
                    proc = self.p[job][m] / self.v[speed_idx]
                    C[i][m-1] = (C[i-1][m-1] if i > 0 else 0) + setup + proc
                    proc_time += proc
                    busy_time[m-1] += proc
                else:  # m==2
                    setup = self.S2[sequence[i-1]-1][job-1] if i > 0 else self.S2[job-1][job-1]
                    prevC_job = C[i][m-2]
                    prevC_mach = C[i-1][m-1] if i > 0 else 0
                    proc = self.p[job][m] / self.v[speed_idx]
                    C[i][m-1] = max(prevC_job, prevC_mach) + setup + proc
                    proc_time += proc
                    busy_time[m-1] += proc

        makespan = C[n-1][1]
        idle_times = [makespan - busy_time[m_idx] for m_idx in range(m)]
        total_idle = sum(idle_times)
        # Energy: job processing + idle (use speed=1.0, index 2, conversion, rho)
        e_proc = 0
        for i, job in enumerate(sequence):
            for idx, m in enumerate(self.machines):
                proc = self.p[job][m] / (60 * self.v[speed_idx]) * self.conv[speed_idx] * self.rho[idx]
                e_proc += proc
        e_idle = sum(self.gamma[m] * self.rho[m-1] * (idle_times[m-1] / 60) for m in self.machines)
        total_energy = e_proc + e_idle

        if show:
            print(f"Seq: {sequence} | Makespan: {makespan:.2f} | Processing: {proc_time:.2f} | Energy: {total_energy:.2f} | Idle: {total_idle:.2f}")
        return makespan, proc_time, total_energy, total_idle

    def fitness(self, objvals, bestvals):
        return 0.25 * sum(obj / best for obj, best in zip(objvals, bestvals))

    def SwapMove(self, solution, i, j):
        solution = solution.copy()
        i_index = solution.index(i)
        j_index = solution.index(j)
        solution[i_index], solution[j_index] = solution[j_index], solution[i_index]
        return solution

    def TSearch(self):
        tenure = self.tabu_tenure
        best_solution = self.initial_solution
        best_objvals = self.Objfun(best_solution)
        best_fitness = self.fitness(best_objvals, best_objvals)  # Should be 1.0 at start
        current_solution = self.initial_solution
        iter = 1
        Terminate = 0
        best_iter = 1
        start_time = time.time()
        time_limit = 30  # seconds

        while (time.time() - start_time) < time_limit:
            tabu_structure = self.get_tabuestructure()  # regenerate random moves
            for move in tabu_structure:
                candidate_solution = self.SwapMove(current_solution, move[0], move[1])
                raw_obj = self.Objfun(candidate_solution)
                tabu_structure[move]['MoveValue'] = sum(raw_obj)
                tabu_structure[move]['Penalized_MV'] = tabu_structure[move]['MoveValue'] + (tabu_structure[move]['freq'] * self.penalization_weight)
                tabu_structure[move]['raw_obj'] = raw_obj

            while True:
                best_move = min(tabu_structure, key=lambda x: tabu_structure[x]['Penalized_MV'])
                MoveValue = tabu_structure[best_move]["MoveValue"]
                tabu_time = tabu_structure[best_move]["tabu_time"]
                raw_obj = tabu_structure[best_move]["raw_obj"]
                current_fitness = self.fitness(raw_obj, best_objvals)
                if tabu_time < iter:
                    current_solution = self.SwapMove(current_solution, best_move[0], best_move[1])
                    if current_fitness < best_fitness:
                        best_solution = current_solution.copy()
                        best_objvals = raw_obj
                        best_fitness = current_fitness
                        best_iter = iter
                        Terminate = 0
                    else:
                        Terminate += 1
                    tabu_structure[best_move]['tabu_time'] = iter + tenure
                    tabu_structure[best_move]['freq'] += 1
                    iter += 1
                    break
                else:
                    if current_fitness < best_fitness:
                        current_solution = self.SwapMove(current_solution, best_move[0], best_move[1])
                        best_solution = current_solution.copy()
                        best_objvals = raw_obj
                        best_fitness = current_fitness
                        best_iter = iter
                        tabu_structure[best_move]['freq'] += 1
                        Terminate = 0
                        iter += 1
                        break
                    else:
                        tabu_structure[best_move]['Penalized_MV'] = float('inf')
                        continue
        return tabu_structure, best_solution, best_objvals, best_fitness, best_iter

# --- Main Batch Evaluation ---
def main():
    dataset_dir = "./data" # set path where your datasets are stored
    pattern = os.path.join(dataset_dir, "ps*j2m*.txt")
    dataset_files = sorted(glob.glob(pattern))

    output_file = "tabu_results.csv"
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "dataset", "run", "speed_level", "setup_size",
            "init_makespan", "init_processing", "init_energy", "init_idle",
            "best_makespan", "best_processing", "best_energy", "best_idle",
            "fitness", "best_sequence", "iterations"
        ])

        for dataset_path in dataset_files:
            data = load_dataset(dataset_path)
            N = data["N"]
            jobs = list(range(1, N+1))
            machines = [1, 2]
            speeds = [1, 2, 3]
            # Parse speed level, setup size from file name if needed
            m = re.search(r"setup(\d+)", os.path.basename(dataset_path))
            setup_size = int(m.group(1)) if m else ""
            speed_level = "" # If speed info is in filename, parse here

            p = {
                j+1: {1: data["p1j"][j], 2: data["p2j"][j]}
                for j in range(N)
            }
            v = {i+1: data["vl"][i] for i in range(len(data["vl"]))}
            conv = {i+1: data["convl"][i] for i in range(len(data["convl"]))}
            gamma = {i+1: data["idleconv_i"][i] for i in range(len(data["idleconv_i"]))}
            S1_raw = data["S1"]
            S2_raw = data["S2"]
            rho = data["rho"]

            for run in range(1, 11):
                seed = int(time.time()) + run
                tabu_tenure = 5
                penalization_weight = 10
                ts = FlowshopTS(
                    jobs, machines, speeds, p, v, conv, gamma, S1_raw, S2_raw, rho,
                    seed, tabu_tenure, penalization_weight, n_moves=30
                )

                init_obj = ts.Objfun(ts.initial_solution)
                writer.writerow([
                    os.path.basename(dataset_path), run, speed_level, setup_size,
                    *["%.2f" % val for val in init_obj],
                    *["%.2f" % val for val in ts.best_objvals],
                    "%.6f" % ts.best_fitness,
                    " ".join(map(str, ts.best_solution)),
                    ts.best_iter
                ])
                csvfile.flush()

if __name__ == "__main__":
    main()
