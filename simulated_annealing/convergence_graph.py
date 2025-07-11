import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_DIR = "/green-scheduling-project/benchmark"      # directory of the dataset

# Load the CSVs
easy_df = pd.read_csv(os.path.join(DATA_DIR, "easy_convergence_values.csv"))
medium_df = pd.read_csv(os.path.join(DATA_DIR, "medium_convergence_values.csv"))
hard_df = pd.read_csv(os.path.join(DATA_DIR, "hard_convergence_values.csv"))

plt.figure(figsize=(10,6))

plt.plot(
    easy_df['time'], easy_df['avg_fitness'],
    label='Easy (α = 0.72, T₀ = 0.60)', color=(0/255, 153/255, 0/255)   # Green
)
plt.plot(
    medium_df['time'], medium_df['avg_fitness'],
    label='Medium (α = 0.99, T₀ = 0.95)', color=(0/255, 76/255, 153/255) # Blue
)
plt.plot(
    hard_df['time'], hard_df['avg_fitness'],
    label='Hard (α = 0.80, T₀ = 0.95)', color=(200/255, 51/255, 51/255)  # Red
)

plt.axvline(1.75, color="black", linestyle="--", linewidth=2, label="t = 1.75s", ymax=0.8)

plt.xlabel("Time (seconds)")
plt.ylabel("Average Fitness")
plt.title("Simulated Annealing Convergence (Easy, Medium, Hard)")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "grouped44_convergence_curves.pdf"), bbox_inches='tight')
plt.show()
print(f"Plot saved as {os.path.join(DATA_DIR, 'grouped44_convergence_curves.pdf')}")
