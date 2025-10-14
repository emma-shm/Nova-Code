import numpy as np
import matplotlib.pyplot as plt

# Example data structure - replace with your actual data
# angles: list of 10 angles (same for all runs)
# means: list of lists - means[run][angle_index]
# errors: list of lists - errors[run][angle_index]

angles = np.array([97.075, 97.080, 97.085, 97.090, 97.095, 
                   97.100, 97.105, 97.110, 97.115, 97.120])

# Replace these with your actual measured data
means = [
    [0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.07, 0.09, 0.11, 0.13],  # Run 1
    [0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.06, 0.08, 0.10, 0.12],  # Run 2
    [0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.08, 0.10, 0.12, 0.14]   # Run 3
]

errors = [
    [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],  # Run 1 SE
    [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],  # Run 2 SE
    [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]   # Run 3 SE
]

# Create the plot
plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'green']  # Different colors for each run

for run in range(len(means)):
    # Plot line connecting the means
    plt.plot(angles, means[run], color=colors[run], linestyle='-', 
             linewidth=2, label=f'Run {run+1}', alpha=0.7)
    
    # Plot error bars at each point
    plt.errorbar(angles, means[run], yerr=errors[run], 
                fmt='o', color=colors[run], capsize=5, capthick=2,
                markersize=6, elinewidth=2, alpha=0.8)

# Customize the plot
plt.xlabel('Angle (degrees)', fontsize=12)
plt.ylabel('Power (μW)', fontsize=12)
plt.title('Polarizer Transmission Near Minimum - Multiple Runs', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Optional: zoom in on the minimum region
plt.xlim(97.075, 97.125)
plt.ylim(min(min(p for p in means)), max(max(p for p in means)) * 1.1)

plt.show()