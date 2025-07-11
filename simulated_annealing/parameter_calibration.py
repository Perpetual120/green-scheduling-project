import os
import re
import random
import math
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------------
# Runtime configuration
# -------------------------------
MAX_RUNTIME = 2.0  # seconds per SA run
NUM_WORKERS = 1   # number of parallel workers

# -------------------------------
# 1. DATA LOADING
# -------------------------------
def load_instance(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    def parse_array(line):
        return [float(x) if "." in x else int(x)
                for x in re.split(r"[\t ]+", line.split(":")[1].strip())]
    p1j, p2j, vl, convl, idleconv_i, pi_i = [], [], [], [], [], []
    S1, S2 = [], []
    N = M = L = None
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
    rho = pi_i if len(pi_i) else [1.0] * (M if M else 2)
    if len(rho) == 1:
        rho = rho * (M if M else 2)
    idle = idleconv_i if len(idleconv_i) else [0.0] * (M if M else 2)
    return {"N": N, "M": M, "L": L,
            "p1j": p1j, "p2j": p2j,
            "vl": vl, "convl": convl,
            "idleconv_i": idle, "rho": rho,
            "S1": S1, "S2": S2}

# -------------------------------
# 2. SIMULATED ANNEALING
# -------------------------------
def simulated_annealing(data, T0, alpha, max_time=MAX_RUNTIME,
                        baseline=None, return_history=False):
    machines = [1, 2]
    p = {j+1: {1: data['p1j'][j], 2: data['p2j'][j]} for j in range(data['N'])}
    S1, S2 = data['S1'], data['S2']
    v, conv, rho = data['vl'], data['convl'], data['rho']
    gamma = {1: data['idleconv_i'][0], 2: data['idleconv_i'][1]}
    bestvals = baseline
    curr = list(p.keys()); random.shuffle(curr)
    start = time.time(); T = T0
    fitness_hist, time_hist = [], []
    def objfun(seq):
        C = [[0]*2 for _ in seq]; busy=[0,0]
        for i, job in enumerate(seq):
            for midx, mach in enumerate(machines):
                prev = seq[i-1] if i>0 else job
                S = S1 if mach==1 else S2
                setup=S[prev-1][job-1]
                ptime=p[job][mach]/v[-1]
                if mach==1:
                    C[i][midx]=(C[i-1][midx] if i>0 else 0)+setup+ptime
                else:
                    C[i][midx]=max(C[i][midx-1],C[i-1][midx] if i>0 else 0)+setup+ptime
                busy[midx]+=ptime
        makespan=C[-1][-1]; idle_times=[makespan-b for b in busy]
        e_proc=sum((p[j][mach]/(60*v[-1]))*conv[-1]*rho[midx]
                   for j in seq for midx,mach in enumerate(machines))
        e_idle=sum(gamma[mach]*rho[midx]*(idle_times[midx]/60)
                   for midx,mach in enumerate(machines))
        return makespan, sum(busy), e_proc+e_idle, sum(idle_times)
    def fitness(vals): return 0.25*sum(o/(b+1e-8) for o,b in zip(vals,bestvals))
    curr_obj=objfun(curr); curr_fit=fitness(curr_obj)
    best_seq, best_obj, best_fit=curr, curr_obj, curr_fit
    while time.time()-start<max_time:
        for _ in range(5):
            cand=curr.copy(); i,j=random.sample(range(len(cand)),2)
            cand[i],cand[j]=cand[j],cand[i]
            co=objfun(cand); cf=fitness(co); delta=cf-curr_fit
            if delta<=0 or random.random()<math.exp(-delta/T): curr, curr_fit, curr_obj=cand,cf,co; 
            if curr_fit<best_fit: best_seq,best_obj,best_fit=curr,curr_obj,curr_fit
            if return_history: fitness_hist.append(curr_fit); time_hist.append(time.time()-start)
        T*=alpha
    return (best_seq,best_obj,best_fit,fitness_hist,time_hist) if return_history else (best_seq,best_obj,best_fit)

# -------------------------------
# 3. GRID CELL EVALUATION
# -------------------------------
def run_sa_grid_cell(filenames, T0, alpha, DATA_DIR, df_b, difficulty):
    vals=[]
    for fn in filenames:
        data=load_instance(os.path.join(DATA_DIR,fn))
        match=df_b[df_b['dataset']==fn]
        if match.empty:
            raise ValueError(f"No baseline for {fn}")
        baseline=match.iloc[0][['best_makespan','best_processing','best_energy','best_idle']].tolist()
        bf=float('inf')
        for _ in range(5): bf=min(bf, simulated_annealing(data,T0,alpha,baseline=baseline)[2])
        vals.append(bf)
    return np.mean(vals), T0, alpha

# -------------------------------
# 4. CONVERGENCE CURVE
# -------------------------------
def get_best_convergence_curve(filenames, T0, alpha, DATA_DIR, df_b, difficulty):
    hist_list, time_list=[],[]
    for fn in filenames:
        data=load_instance(os.path.join(DATA_DIR,fn)); match=df_b[df_b['dataset']==fn]
        if match.empty:
            raise ValueError(f"No baseline for {fn}")
        baseline=match.iloc[0][['best_makespan','best_processing','best_energy','best_idle']].tolist()
        bf=float('inf'); bh=([],[])
        for _ in range(5):
            _,_,fit,hist,times=simulated_annealing(data,T0,alpha,max_time=MAX_RUNTIME,baseline=baseline,return_history=True)
            if fit<bf: bf, bh = fit, (hist,times)
        hist_list.append(bh[0]); time_list.append(bh[1])
    m=min(max(t) for t in time_list); grid=np.linspace(0,m,200)
    arrs=[np.interp(grid,t,h,h[-1]) for h,t in zip(hist_list,time_list)]
    return grid, np.mean(arrs,axis=0)

# -------------------------------
# 5. MAIN: PARALLEL GRID SEARCH & DATA SAVE
# -------------------------------
def main():
    DATA_DIR = "/green-scheduling-project/benchmark"       # directory of the dataset
    df_b = pd.read_csv("/green-scheduling-project/simulated_annealing/final_best_values.csv")
    easy = [
        'ps4j2m-setup125_1.31_0.63_63321.txt',
        'ps6j2m-setup99_1.23_0.64_41468.txt',
        'ps6j2m-setup25_1.31_0.63_63321.txt',
        'ps5j2m-setup125_1.25_0.64_33919.txt',
        'ps4j2m-setup50_1.27_0.72_79727.txt'
    ]
    medium = [
        'ps20j2m-setup25_1.24_0.56_80879.txt',
        'ps20j2m-setup25_1.25_0.55_38999.txt',
        'ps50j2m-setup99_1.34_0.73_65114.txt',
        'ps50j2m-setup50_1.08_0.59_57394.txt',
        'ps50j2m-setup99_1.35_0.83_93184.txt'
    ]
    hard = [
        'ps120j2m-setup125_1.3_0.79_68968.txt',
        'ps80j2m-setup99_1.12_0.71_16039.txt',
        'ps80j2m-setup50_1.28_0.65_49292.txt',
        'ps120j2m-setup25_1.14_0.52_49633.txt',
        'ps120j2m-setup50_1.25_0.74_73248.txt'
    ]
    groups = {'easy': easy, 'medium': medium, 'hard': hard}
    T0_list = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    alpha_list = [0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.95, 0.99]

    best_params = {}
    for diff, files in groups.items():
        print(f"Starting grid search for {diff}...")
        jobs = [(files, T0, alpha, DATA_DIR, df_b, diff)
                for alpha in alpha_list for T0 in T0_list]
        results = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(run_sa_grid_cell, *job): job for job in jobs}
            for fut in as_completed(futures):
                avg_fit, T0, alpha = fut.result()
                results.append({'T0': T0, 'alpha': alpha, 'avg_fitness': avg_fit})
        df_grid = pd.DataFrame(results)
        # save raw grid
        grid_csv = os.path.join(DATA_DIR, f"sa_{diff}_grid.csv")
        df_grid.to_csv(grid_csv, index=False)
        print(f"Saved raw grid for {diff}: {grid_csv}")

        # pivot for display
        pivot = df_grid.pivot(index='alpha', columns='T0', values='avg_fitness')
        print(f"===== {diff.upper()} PROBLEMS GRID =====")
        print(pivot)
        print()

        # record best params
        best = pivot.stack().idxmin()
        best_params[diff] = (best[1], best[0])  # (T0, alpha)

    # save convergence values only
    for diff, files in groups.items():
        T0, alpha = best_params[diff]
        time_grid, avg_fitness = get_best_convergence_curve(
            files, T0, alpha, DATA_DIR, df_b, diff)
        df_vals = pd.DataFrame({'time': time_grid, 'avg_fitness': avg_fitness})
        val_csv = os.path.join(DATA_DIR, f"{diff}_convergence_values.csv")
        df_vals.to_csv(val_csv, index=False)
        print(f"Saved convergence values for {diff}: {val_csv}")

    print("Grid search and data export complete.")

if __name__ == "__main__":
    main()
