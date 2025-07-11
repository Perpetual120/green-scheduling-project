import os
import re
import random
import math
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def load_dataset(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    def parse_array(line):
        return [float(x) if "." in x else int(x)
                for x in re.split(r"[\t ]+", line.split(":")[1].strip())]
    N = M = L = None
    p1j = p2j = vl = convl = idleconv_i = pi_i = []
    S1 = S2 = []
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
        elif state == "S1" and (line[0].isdigit() or line[0] == '-'):
            S1.append(parse_array(":" + line))
        elif state == "S2" and (line[0].isdigit() or line[0] == '-'):
            S2.append(parse_array(":" + line))
        else:
            state = None
    rho = pi_i or [1.0] * (M or 2)
    if len(rho) == 1:
        rho = rho * (M or 2)
    idle = idleconv_i or [0.0] * (M or 2)
    return {"N": N, "M": M, "L": L,
            "p1j": p1j, "p2j": p2j,
            "vl": vl, "convl": convl,
            "idleconv_i": idle, "rho": rho,
            "S1": S1, "S2": S2}

class SimulatedAnnealingFlowshop:
    def __init__(self, instance_data, initial_temp, cooling_rate, bestvals, time_limit, max_iter=None):
        self.machines = [1, 2]
        self.p = {j+1: {1: instance_data['p1j'][j], 2: instance_data['p2j'][j]}
                  for j in range(instance_data['N'])}
        self.S1 = instance_data['S1']
        self.S2 = instance_data['S2']
        self.v = instance_data['vl']
        self.conv = instance_data['convl']
        self.rho = instance_data['rho']
        self.gamma = {1: instance_data['idleconv_i'][0], 2: instance_data['idleconv_i'][1]}
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.bestvals = bestvals
        self.time_limit = time_limit
        self.max_iter = max_iter

    def objfun(self, seq):
        n, m = len(seq), len(self.machines)
        C = [[0]*m for _ in range(n)]
        busy = [0]*m
        speed_idx = len(self.v)-1
        for i, job in enumerate(seq):
            for midx, mach in enumerate(self.machines):
                prev_job = seq[i-1] if i>0 else job
                S = self.S1 if mach==1 else self.S2
                setup = S[prev_job-1][job-1]
                ptime = self.p[job][mach]/self.v[speed_idx]
                if mach==1:
                    C[i][midx] = (C[i-1][midx] if i>0 else 0) + setup + ptime
                else:
                    C[i][midx] = max(C[i][midx-1], (C[i-1][midx] if i>0 else 0)) + setup + ptime
                busy[midx] += ptime
        makespan = C[-1][-1]
        idle_times = [makespan-b for b in busy]
        total_idle = sum(idle_times)
        e_proc = sum((self.p[job][mach]/(60*self.v[speed_idx])) *
                     self.conv[speed_idx] * self.rho[idx]
                     for i, job in enumerate(seq)
                     for idx, mach in enumerate(self.machines))
        e_idle = sum(self.gamma[mach]*self.rho[idx]*(idle_times[idx]/60)
                     for idx, mach in enumerate(self.machines))
        return makespan, sum(busy), e_proc+e_idle, total_idle

    def fitness(self, objvals):
        return 0.25 * sum(o/(b+1e-8) for o,b in zip(objvals, self.bestvals))

    def run(self):
        start = time.time()
        T = self.initial_temp
        curr = list(self.p.keys())
        random.shuffle(curr)
        curr_obj = self.objfun(curr)
        curr_fit = self.fitness(curr_obj)
        best_seq, best_obj, best_fit = curr, curr_obj, curr_fit
        iter_count = 0
        while (self.max_iter is None and time.time() - start < self.time_limit) or (self.max_iter is not None and iter_count < self.max_iter):
            for _ in range(5):
                cand = curr.copy()
                i,j = random.sample(range(len(cand)),2)
                cand[i],cand[j] = cand[j],cand[i]
                cand_obj = self.objfun(cand)
                cand_fit = self.fitness(cand_obj)
                delta = cand_fit - curr_fit
                try:
                    mprob = math.exp(-delta/T) if T > 1e-8 else 0
                except OverflowError:
                    mprob = float('inf')
                if delta<=0 or random.random()<mprob:
                    curr, curr_obj, curr_fit = cand, cand_obj, cand_fit
                    if curr_fit<best_fit:
                        best_seq, best_obj, best_fit = curr, curr_obj, curr_fit
                iter_count += 1
                if self.max_iter is not None and iter_count >= self.max_iter:
                    break
            T *= self.cooling_rate
            if self.max_iter is not None and iter_count >= self.max_iter:
                break
        return best_seq, best_obj, best_fit, iter_count

def get_baseline(filename, bvals_df):
    # Normalize both sides for robust matching
    fname = os.path.basename(filename).strip().lower()
    if 'dataset_norm' not in bvals_df.columns:
        bvals_df['dataset_norm'] = bvals_df['dataset'].str.strip().str.lower()
    row = bvals_df[bvals_df['dataset_norm'] == fname]
    if row.empty:
        print(f"Skip {fname}: no baseline")
        return None, None
    dataset_name = row.iloc[0]['dataset']  # exact name as in CSV
    baseline = row.iloc[0][['best_makespan','best_processing','best_energy','best_idle']].tolist()
    return dataset_name, baseline

def get_group_and_params(n_jobs):
    if n_jobs in (4, 5, 6):
        return 'easy', 0.6, 0.72, None
    elif n_jobs in (20, 50):
        return 'medium', 0.95, 0.99, None
    elif n_jobs in (80, 120):
        return 'hard', 0.80, 0.95, None
    else:
        return None, None, None, None

def get_max_iter(n_jobs):
    if n_jobs in (4, 5, 6):
        return 1000
    elif n_jobs in (20, 50):
        return 2000
    elif n_jobs in (80, 120):
        return 3000
    else:
        return None

def worker(args):
    filepath, bvals_df = args
    data = load_dataset(filepath)
    n_jobs = data['N']
    group, initial_temp, cooling_rate, _ = get_group_and_params(n_jobs)
    if group is None:
        return None
    dataset_name, baseline = get_baseline(filepath, bvals_df)
    if baseline is None:
        return None
    max_iter = get_max_iter(n_jobs)
    best_fit = float('inf')
    best_result = None
    for run in range(10):
        sa = SimulatedAnnealingFlowshop(
            data, initial_temp=initial_temp, cooling_rate=cooling_rate,
            bestvals=baseline, time_limit=2, max_iter=max_iter
        )
        seq, obj, fit, iterations = sa.run()
        if fit < best_fit:
            makespan, proc_time, energy, idle_time = obj
            best_result = {
                "instance": dataset_name,  # Exact name from CSV
                "group": group,
                "run": run+1,
                "makespan": round(makespan, 4),
                "proc_time": round(proc_time, 4),
                "energy": round(energy, 4),
                "idle_time": round(idle_time, 4),
                "b1(make)": round(baseline[0], 4),
                "b2(proc)": round(baseline[1], 4),
                "b3(energy)": round(baseline[2], 4),
                "b4(idle)": round(baseline[3], 4),
                "fitness": round(fit, 4),
                "iterations": iterations,
                "sequence": seq,
                "n_jobs": n_jobs,
                "init_temp": initial_temp,
                "alpha": cooling_rate
            }
            best_fit = fit
    return best_result

if __name__ == "__main__":
    basepath = "/green-scheduling-project/benchmark"               # directory for the dataset
    bvals_csv = "/green-scheduling-project/simulated_annealing/final_best_values.csv"
    bvals_df = pd.read_csv(bvals_csv)
    files = [os.path.join(basepath, f) for f in os.listdir(basepath) if f.endswith('.txt')]
    results = []
    with ProcessPoolExecutor(max_workers=1) as pool:
        for res in pool.map(worker, [(filepath, bvals_df) for filepath in files]):
            if res is not None:
                results.append(res)
    df = pd.DataFrame(results)
    df = df.sort_values('n_jobs')
    df.to_csv("sa_fshop_best_runs.csv", index=False)
    print("Results saved to sa_flowshop_best_runs.csv")
