import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from matplotlib.gridspec import GridSpec  # For better subplot sizing control

# Initialize Tkinter root and hide it
root = tk.Tk()
root.withdraw()

# Prompt user to select the folder containing difference_map.npy and valid_mask.npy
input_folder = filedialog.askdirectory(title="Select the folder containing difference_map.npy and valid_mask.npy")
if not input_folder:
    print("No input folder selected. Exiting.")
    exit()
print(f"Selected input folder: {input_folder}")

# Prompt user to select the 'before' folder for alignment data
before_folder = filedialog.askdirectory(title="Select the 'before' TTV data folder for alignment data")
if not before_folder:
    print("No 'before' folder selected. Exiting.")
    exit()
print(f"Selected 'before' folder: {before_folder}")

# Set output folder to be the same as input folder
output_folder = input_folder
os.makedirs(output_folder, exist_ok=True)

# Load precomputed difference map and valid mask
difference_path = os.path.join(input_folder, 'difference_map.npy')
valid_mask_path = os.path.join(input_folder, 'valid_mask.npy')

if not os.path.exists(difference_path) or not os.path.exists(valid_mask_path):
    raise FileNotFoundError("difference_map.npy or valid_mask.npy not found in the selected folder.")

difference = np.load(difference_path)
valid_mask = np.load(valid_mask_path)

# Load alignment data to get x_mm and y_mm from the 'before' folder
alignment1 = None
for file in os.listdir(before_folder):
    if file.startswith('alignment_') and file.endswith('.npy'):
        alignment1 = np.load(os.path.join(before_folder, file), allow_pickle=True).item()
        break

if alignment1 is None:
    raise FileNotFoundError("alignment_*.npy not found in the 'before' folder.")

x_mm = alignment1['x_shifted_mm']
y_mm = alignment1['y_shifted_mm']

# Compute stats from the difference map
valid_diffs = difference[valid_mask]
stats = {
    'mean': np.mean(valid_diffs) if len(valid_diffs) > 0 else 0,
    'std': np.std(valid_diffs) if len(valid_diffs) > 0 else 0,
    'min': np.min(valid_diffs) if len(valid_diffs) > 0 else 0,
    'max': np.max(valid_diffs) if len(valid_diffs) > 0 else 0,
    'range': np.ptp(valid_diffs) if len(valid_diffs) > 0 else 0,
    'valid_fraction': np.sum(valid_mask) / np.prod(valid_mask.shape)
}

print("\nDifference Map Statistics:")
print(f"  Mean: {stats['mean']:.3f} nm")
print(f"  Std:  {stats['std']:.3f} nm")
print(f"  Range: {stats['range']:.3f} nm")
print(f"  Valid data: {stats['valid_fraction']*100:.1f}%")

vmin, vmax = -200, 300

# Prompt user to decide whether to zoom into the polished region
zoom_in = messagebox.askyesno("Zoom", "Do you want to zoom into the polished region?")

# Define the range for the initial plot based on user choice
if zoom_in:
    x_start_idx, x_end_idx = 909, 1160
    y_start_idx, y_end_idx = 1023, 1261
    cropped_diff = difference[y_start_idx:y_end_idx, x_start_idx:x_end_idx]
    cropped_x = x_mm[x_start_idx:x_end_idx]  # array of x values in mm for the cropped region
    cropped_y = y_mm[y_start_idx:y_end_idx]  # array of y values in mm for the cropped region
else:
    x_start_idx, x_end_idx = 0, difference.shape[1]
    y_start_idx, y_end_idx = 0, difference.shape[0]
    cropped_diff = difference  # Use full image
    cropped_x = x_mm  # Full x-coordinates
    cropped_y = y_mm  # Full y-coordinates

# Plot 1: Initial cropped Thickness Difference
plt.figure(figsize=(10, 10))
im = plt.imshow(cropped_diff, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                extent=[cropped_x[0], cropped_x[-1], cropped_y[-1], cropped_y[0]])
if zoom_in:
    im.set_clim(vmin, vmax)
plt.title(f'Select Polished Region')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
cbar = plt.colorbar(im, label='Difference (nm)')
cbar.set_ticks([vmin, -200, -100, 0, 100, 200, vmax])

print("Click on the upper-left corner, then the bottom-right corner of the polished region.")
points = plt.ginput(2, timeout=120, show_clicks=True)
plt.close()

i=1  # Counter for output folder naming if needed
if len(points) == 2:
    ul_x, ul_y = points[0]  # assigning first selected point in UPPER LEFT CORNER to ul_x and ul_y
    br_x, br_y = points[1]  # assigning second selected point in BOTTOM RIGHT CORNER to br_x and br_y
    
    # Adjust indices to be relative to the cropped region
    ul_x_idx = np.argmin(np.abs(cropped_x - ul_x)) + x_start_idx  # finding the index of the x value in cropped_x
    ul_y_idx = np.argmin(np.abs(cropped_y - ul_y)) + y_start_idx  # same for y
    br_x_idx = np.argmin(np.abs(cropped_x - br_x)) + x_start_idx  # same for bottom right x
    br_y_idx = np.argmin(np.abs(cropped_y - br_y)) + y_start_idx  # same for bottom right y
    
    x_start = min(ul_x_idx, br_x_idx)  # getting the minimum x index
    x_end = max(ul_x_idx, br_x_idx) + 1  # getting the maximum x index, adding 1 for slicing
    y_start = min(ul_y_idx, br_y_idx)  # getting the minimum y index
    y_end = max(ul_y_idx, br_y_idx) + 1  # getting the maximum y index, adding 1 for slicing
    
    print(f"Polished region indices:")
    print(f"X start index: {x_start}, X end index: {x_end}")
    print(f"Y start index: {y_start}, Y end index: {y_end}")
    
    print(f"Corresponding mm ranges:")
    print(f"X: {x_mm[x_start]:.2f} mm to {x_mm[x_end-1]:.2f} mm")
    print(f"Y: {y_mm[y_start]:.2f} mm to {y_mm[y_end-1]:.2f} mm")
    
    # Calculate dimensions in mm
    x_dim = abs(x_mm[x_end-1] - x_mm[x_start])
    y_dim = abs(y_mm[y_end-1] - y_mm[y_start])

    x_rounded=round(x_dim)
    y_rounded=round(y_dim)
    
    # Prompt user for the new folder name
    new_folder_name = simpledialog.askstring("Input",f"Enter the name for the output folder for the selected region (default: polished_region_{x_rounded}x{y_rounded}):", parent=root)
    if not new_folder_name:
        print(f"No folder name provided. Using default name 'polished_region_{x_rounded}x{y_rounded}'.")
        new_folder_name = f'polished_region_{x_rounded}x{y_rounded}'
    
    # Create new folder in the input_folder directory
    new_output_folder = os.path.join(input_folder, new_folder_name)
    os.makedirs(new_output_folder, exist_ok=True)
    
    cropped_map = difference[y_start:y_end, x_start:x_end]  # slicing the ORIGINAL difference map
    cropped_x = x_mm[x_start:x_end]
    cropped_y = y_mm[y_start:y_end]

    if zoom_in: # only manually adjusting vmin/vmax if user zoomed in to right around the polished region before selecting, because in that case we know good vmin/vmax values for polished region; otherwise, use default
        vmin_zoomed, vmax_zoomed = 200, 400
    else:
        vmin_zoomed, vmax_zoomed = None, None  # use default scaling
    
    plt.figure(figsize=(10,10))
    plt.imshow(cropped_map, cmap='RdBu_r', vmin=vmin_zoomed, vmax=vmax_zoomed,
               extent=[cropped_x[0], cropped_x[-1], cropped_y[-1], cropped_y[0]])
    plt.title(f'Selected Polished Region ({os.path.basename(input_folder)})\nDimensions: {x_dim:.2f}mm x {y_dim:.2f}mm\n X start: {cropped_x[0]:.2f}mm, X end: {cropped_x[-1]:.2f}mm\n Y start: {cropped_y[-1]:.2f}mm, Y end: {cropped_y[0]:.2f}mm')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.colorbar(label='Difference (nm)')
    plt.savefig(os.path.join(new_output_folder, 'polished_region_difference.png'))
    plt.close()

    # 2. Line plots of mean subtractions in selected polished region
    # Mean over x (average along columns, as function of y)
    mean_over_x = np.mean(cropped_map, axis=1)

    # Mean over y (average along rows, as function of x)
    mean_over_y = np.mean(cropped_map, axis=0)

    # Plot them
    plt.figure(figsize=(12, 5))

    # Subplot for mean as function of y
    plt.subplot(1, 2, 1)
    plt.plot(cropped_y, mean_over_x)
    plt.title('Mean Difference vs Y (averaged over X) - Polished Region')
    plt.xlabel('Y (mm)')
    plt.ylabel('Mean Difference (nm)')
    plt.grid(True)

    # Subplot for mean as function of x
    plt.subplot(1, 2, 2)
    plt.plot(cropped_x, mean_over_y)
    plt.title('Mean Difference vs X (averaged over Y) - Polished Region')
    plt.xlabel('X (mm)')
    plt.ylabel('Mean Difference (nm)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(new_output_folder, 'mean_line_plots_polished_region.png'))
    plt.close()

    # 3. Line plots for four evenly spaced x-values (thickness along y-direction)
    plt.figure(figsize=(8, 6))
    x_indices = np.linspace(0, cropped_map.shape[1]-1, 4, dtype=int)  # Four evenly spaced x indices
    for x_idx in x_indices:
        x_val = cropped_x[x_idx]
        thickness = cropped_map[:, x_idx]  # Thickness values along y for fixed x
        plt.plot(cropped_y, thickness, label=f'X = {x_val:.2f} mm')
    plt.title('Thickness Profiles at Four X Positions')
    plt.xlabel('Y (mm)')
    plt.ylabel('Thickness Difference (nm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(new_output_folder, 'x_profiles_polished_region.png'))
    plt.close()

    # 4. Line plots for four evenly spaced y-values (thickness along x-direction)
    plt.figure(figsize=(8, 6))
    y_indices = np.linspace(0, cropped_map.shape[0]-1, 4, dtype=int)  # Four evenly spaced y indices
    for y_idx in y_indices:
        y_val = cropped_y[y_idx]
        thickness = cropped_map[y_idx, :]  # Thickness values along x for fixed y
        plt.plot(cropped_x, thickness, label=f'Y = {y_val:.2f} mm')
    plt.title('Thickness Profiles at Four Y Positions')
    plt.xlabel('X (mm)')
    plt.ylabel('Thickness Difference (nm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(new_output_folder, 'y_profiles_polished_region.png'))
    plt.close()
    
        # Combined figure: Polished region map + x_profiles + y_profiles side by side
    fig = plt.figure(figsize=(24, 6))  # Wide figure for 3 subplots
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.4, 1, 1], wspace=0.3, left=0.029, right=0.97)

    # Subplot 1: Polished region imshow
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(cropped_map, cmap='RdBu_r',vmin=vmin_zoomed, vmax=vmax_zoomed,
                    extent=[cropped_x[0], cropped_x[-1], cropped_y[-1], cropped_y[0]])
    ax1.set_title(f'Selected Polished Region\nDimensions: {x_dim:.2f}mm x {y_dim:.2f}mm')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    cbar1 = fig.colorbar(im, ax=ax1, shrink=0.8)
    cbar1.set_label('Difference (nm)')

    # Subplot 2: Thickness profiles at four x positions (along y)
    ax2 = fig.add_subplot(gs[0, 1])
    x_indices = np.linspace(0, cropped_map.shape[1]-1, 4, dtype=int) # shape[1] is number of columns (x direction) for cropped_map; using linspace to generate 4 evenly spaced x points
    for x_idx in x_indices: # loop over the selected x indices
        x_val = cropped_x[x_idx] # get the actual x value in mm at each index
        thickness = cropped_map[:, x_idx] # get thickness values along y for the fixed x index
        ax2.plot(cropped_y, thickness, label=f'X = {x_val:.2f} mm')
    ax2.set_title('Thickness Profiles at Four X Positions')
    ax2.set_xlabel('Y (mm)')
    ax2.set_ylabel('Thickness Difference (nm)')
    ax2.legend()
    ax2.grid(True)

    # Subplot 3: Thickness profiles at four y positions (along x)
    ax3 = fig.add_subplot(gs[0, 2])
    y_indices = np.linspace(0, cropped_map.shape[0]-1, 4, dtype=int)  # shape[0] is number of rows (y direction) for cropped_map
    for y_idx in y_indices: # loop over the selected y indices
        y_val = cropped_y[y_idx] # get the actual y value in mm at each index
        thickness = cropped_map[y_idx, :] # get thickness values along x for the fixed y index
        ax3.plot(cropped_x, thickness, label=f'Y = {y_val:.2f} mm')
    ax3.set_title('Thickness Profiles at Four Y Positions')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Thickness Difference (nm)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    combined_path = os.path.join(new_output_folder, 'combined_polished_region_profiles.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {combined_path}")
    plt.show()

    i+=1  # Increment counter for next potential folder

# Destroy Tkinter root
root.destroy()