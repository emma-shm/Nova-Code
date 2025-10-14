import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2_contingency

# Step 1: Your data structure (replace with your actual measurements)
angles = np.array([97.075, 97.080, 97.085, 97.090, 97.095, 
                   97.100, 97.105, 97.110, 97.115, 97.120])

# Example data - REPLACE WITH YOUR MEASURED VALUES
means = [
    [0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.07, 0.09, 0.11, 0.13],  # Run 1
    [0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.06, 0.08, 0.10, 0.12],  # Run 2
    [0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.08, 0.10, 0.12, 0.14]   # Run 3
]

errors = [
    [0.02]*10,  # Standard errors for Run 1 (replace with actual)
    [0.02]*10,  # Run 2
    [0.02]*10   # Run 3
]

# Step 2: Combine all runs into single arrays for fitting
all_angles = np.concatenate([angles, angles, angles])  # 30 total angles
all_means = np.concatenate(means)                      # 30 power values
all_errors = np.concatenate(errors)                    # 30 uncertainties

# Step 3: Define Malus's law model function
def malus(theta, Pmax, theta0, Poffset):
    """
    Malus's law: P = Pmax * cos²(θ - θ0) + Poffset
    theta: angle in degrees
    theta0: angle where polarizers are parallel (maximum)
    Minimum occurs at theta0 + 90°
    """
    return Pmax * np.cos(np.deg2rad(theta - theta0))**2 + Poffset

# Step 4: Set initial parameter guesses
# Rough estimates: Pmax ≈ max power, theta0 ≈ 7.1°, Poffset ≈ min power
p0 = [np.max(all_means), 7.1, np.min(all_means)]

# Step 5: Perform the fit with error weighting
try:
    popt, pcov = curve_fit(
        malus, 
        all_angles, 
        all_means,
        p0=p0,
        sigma=all_errors,           # Use measurement errors
        absolute_sigma=True,        # Treat errors as absolute values
        maxfev=5000                 # Allow more iterations if needed
    )
    
    # Extract fitted parameters and uncertainties
    Pmax_fit, theta0_fit, Poffset_fit = popt
    Pmax_err, theta0_err, Poffset_err = np.sqrt(np.diag(pcov))
    
    # Calculate minimum angle
    theta_min = theta0_fit + 90
    theta_min_err = theta0_err
    
    print("=== FIT RESULTS ===")
    print(f"Parallel angle (θ₀):     {theta0_fit:.4f}° ± {theta0_err:.4f}°")
    print(f"Minimum angle:           {theta_min:.4f}° ± {theta_min_err:.4f}°")
    print(f"Max power (P_max):       {Pmax_fit:.4f} ± {Pmax_err:.4f}")
    print(f"Offset power:            {Poffset_fit:.4f} ± {Poffset_err:.4f}")
    
    # Step 6: Calculate goodness of fit
    predictions = malus(all_angles, *popt)
    residuals = all_means - predictions
    chi_squared = np.sum((residuals / all_errors)**2)
    dof = len(all_means) - len(p0)  # degrees of freedom
    reduced_chi2 = chi_squared / dof
    
    print(f"\nGoodness of fit:")
    print(f"χ² = {chi_squared:.2f}, reduced χ² = {reduced_chi2:.3f}")
    print(f"(reduced χ² ≈ 1 indicates good fit)")
    
    fit_success = True
    
except Exception as e:
    print(f"Fit failed: {e}")
    print("Try adjusting initial guesses or check data quality")
    fit_success = False
    theta_min = None

# Step 7: Plotting (only if fit succeeded)
if fit_success:
    # Generate smooth theoretical curve for plotting
    theta_smooth = np.linspace(min(all_angles)-0.01, max(all_angles)+0.01, 1000)
    P_smooth = malus(theta_smooth, *popt)
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    colors = ['blue', 'red', 'green']
    for i, (run_means, run_errors) in enumerate(zip(means, errors)):
        plt.errorbar(angles, run_means, yerr=run_errors, 
                    fmt='o-', color=colors[i], capsize=4, alpha=0.7,
                    label=f'Run {i+1}', markersize=6)
    
    # Plot all data points (small markers)
    plt.scatter(all_angles, all_means, c='black', s=20, alpha=0.5, zorder=1)
    
    # Plot fitted curve
    plt.plot(theta_smooth, P_smooth, 'r-', linewidth=3, 
             label=f'Fit: P_max={Pmax_fit:.3f}, θ₀={theta0_fit:.3f}°')
    
    # Mark minimum
    plt.axvline(theta_min, color='green', linestyle='--', linewidth=2,
                label=f'Minimum: {theta_min:.4f}° ± {theta_min_err:.4f}°')
    
    # Formatting
    plt.xlabel('Angle (degrees)', fontsize=12)
    plt.ylabel('Power', fontsize=12)
    plt.title('Malus Law Fit - Polarizer Extinction Minimum', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Optional: Plot residuals
    plt.figure(figsize=(10, 4))
    plt.errorbar(all_angles, residuals, yerr=all_errors, fmt='o', alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Residuals')
    plt.title('Fit Residuals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()