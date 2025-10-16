import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2_contingency

# Step 1: Your data structure (replace with your actual measurements)
angles = np.array([96.960, 96.970, 96.980, 96.990, 97.000, 97.010, 97.020, 97.030, 97.040, 97.050, 97.060, 97.070, 97.080, 97.090, 97.100, 97.110, 97.120, 97.130, 97.140, 97.150, 97.160, 97.170, 97.180, 97.19])

# Data
means = np.array([
    [70.16, 69.36, 66.52, 67.78, 67.02, 66.22, 65.26, 64.62,
    64.36, 64.04, 63.96, 64.10, 64.04, 63.92, 63.80, 63.78,
    63.84, 64.04, 64.44, 65.06, 65.60, 66.54, 67.22, 67.92],  # Run 1

    [69.18, 68.46, 67.76, 67.20, 66.62, 66.10, 65.66, 65.46,
    65.44, 65.16, 64.90, 64.92, 64.86, 64.86, 64.82, 64.90,
    64.94, 64.92, 65.12, 65.30, 65.62, 66.08, 66.70, 67.38],  # Run 2

    [70.04, 69.82, 69.16, 68.48, 67.94, 67.36, 66.92, 66.46,
    66.16, 65.80, 65.54, 65.22, 65.10, 65.00, 64.94, 65.10,
    65.28, 65.50, 65.90, 66.22, 66.80, 67.38, 68.00, 68.70],   # Run 3

    [71.08, 70.22, 69.30, 72.08, 67.82, 67.20, 66.76, 66.32,
    66.00, 65.74, 65.62, 65.56, 65.30, 65.28, 65.46, 65.70,
    65.80, 66.02, 66.28, 66.58, 66.92, 67.42, 67.90, 68.66]     # Run 4
])

# Standard errors
errors = np.array([
    [0.044, 0.044, 4.337, 0.043, 0.009, 0.116, 0.046, 0.083,
    0.044, 0.044, 0.083, 0.084, 0.044, 0.071, 0.045, 0.043,
    0.044, 0.050, 0.050, 0.050, 0.000, 0.046, 0.043, 0.009],  # Standard errors for Run 1
    [0.009, 0.044, 0.050, 0.000, 0.043, 0.000, 0.044, 0.044,
    0.044, 0.050, 0.000, 0.043, 0.050, 0.044, 0.043, 0.045,
    0.044, 0.048, 0.009, 0.000, 0.071, 0.043, 0.000, 0.043],  # Run 2
    [0.084, 0.043, 0.050, 0.043, 0.044, 0.050, 0.043, 0.050, 0.050, 0.000,
    0.050, 0.009, 0.045, 0.000, 0.044, 0.000, 0.043, 0.000, 0.000, 0.009,
    0.000, 0.043, 0.000, 0.000],
    [0.043, 0.009, 0.000, 1.613, 0.009, 0.000, 0.044, 0.009, 0.000, 0.050,
    0.043, 0.083, 0.000, 0.009, 0.044, 0.000, 0.000, 0.009, 0.043, 0.043,
    0.009, 0.043, 0.000, 0.050]   # Run 3
])

# Step 2: Combine all runs into single arrays for fitting
# Step 2: Combine all runs into single arrays for fitting
all_angles = np.concatenate([angles, angles, angles, angles])
all_means = np.concatenate(means)     # 72 power values
all_errors = np.concatenate(errors)    # 72 uncertainties
# Replace zeros in error array to avoid divide-by-zero issues
all_errors[all_errors == 0] = np.min(all_errors[np.nonzero(all_errors)])


# Step 3: Define Malus's law model function
def malus(theta, Pmax, theta0, Poffset):
    return Pmax * np.cos(np.deg2rad(theta - theta0))**2 + Poffset

# Step 4: Set initial parameter guesses
p0 = [np.max(all_means), 7.1, np.min(all_means)]

# Step 5: Perform the fit with error weighting
try:
    popt, pcov = curve_fit(
        malus, 
        all_angles, 
        all_means,
        p0=p0,
        sigma=all_errors,
        maxfev=5000
    )
    
    Pmax_fit, theta0_fit, Poffset_fit = popt
    Pmax_err, theta0_err, Poffset_err = np.sqrt(np.diag(pcov))
    theta_min = theta0_fit + 90
    theta_min_err = theta0_err
    
    print("=== FIT RESULTS ===")
    print(f"Parallel angle (θ₀):     {theta0_fit:.4f}° ± {theta0_err:.4f}°")
    print(f"Minimum angle:           {theta_min:.4f}° ± {theta_min_err:.4f}°")
    print(f"Max power (P_max):       {Pmax_fit:.4f} ± {Pmax_err:.4f}")
    print(f"Offset power:            {Poffset_fit:.4f} ± {Poffset_err:.4f}")
    
    predictions = malus(all_angles, *popt)
    residuals = all_means - predictions
    chi_squared = np.sum((residuals / all_errors)**2)
    dof = len(all_means) - len(p0)
    reduced_chi2 = chi_squared / dof
    
    print(f"\nGoodness of fit:")
    print(f"χ² = {chi_squared:.2f}, reduced χ² = {reduced_chi2:.3f}")
    
    fit_success = True
    
except Exception as e:
    print(f"Fit failed: {e}")
    fit_success = False
    theta_min = None

# Step 6: Plotting
if fit_success:
    theta_smooth = np.linspace(min(all_angles)-0.01, max(all_angles)+0.01, 1000)
    P_smooth = malus(theta_smooth, *popt)
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs with proper per-point error bars
    colors = ['blue', 'red', 'green']
    for i, (run_means, run_errors) in enumerate(zip(means, errors)):
        plt.errorbar(angles, run_means, yerr=run_errors, 
                 fmt='o-', capsize=4, alpha=0.7, linewidth=1.5,
                 label=f'Run {i+1}')
    
    # Plot fitted curve
    plt.plot(theta_smooth, P_smooth, 'k-', linewidth=3, 
             label=f'Fit: P_max={Pmax_fit:.3f}, θ₀={theta0_fit:.3f}°')
    
    # Mark minimum
    plt.axvline(theta_min, color='green', linestyle='--', linewidth=2,
                label=f'Minimum: {theta_min:.4f}° ± {theta_min_err:.4f}°')
    
    plt.xlabel('Angle (degrees)', fontsize=12)
    plt.ylabel('Power', fontsize=12)
    plt.title('Malus Law Fit - Polarizer Extinction Minimum', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('malus_fit.png', dpi=300)
    plt.show()
    
    # Residuals plot
    plt.figure(figsize=(10, 4))
    plt.errorbar(all_angles, residuals, yerr=all_errors, fmt='o', alpha=0.6, markersize=4)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Residuals')
    plt.title('Fit Residuals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('malus_residuals.png', dpi=300)
    plt.show()
