import numpy as np
import matplotlib.pyplot as plt
import os
import json
import matplotlib.image as mpimg

# Manually set file paths
folder1_path = r"C:\Users\Emma Martignoni\Desktop\2025-10-09 FJPtest BEFORE"
folder2_path = r"C:\Users\Emma Martignoni\Desktop\2025-10-17 FJPtest AFTER"
output_folder = r"C:\Users\Emma Martignoni\Desktop\test"

# Load data
for file in os.listdir(folder1_path):
    if file.startswith('TTV_thickness_2D_') and file.endswith('.npy'):
        ttv1 = np.load(os.path.join(folder1_path, file))
        dataset1 = file.replace('TTV_thickness_2D_', '').replace('.npy', '')
    if file.startswith('alignment_') and file.endswith('.npy'):
        alignment1 = np.load(os.path.join(folder1_path, file), allow_pickle=True).item()

for file in os.listdir(folder2_path):
    if file.startswith('TTV_thickness_2D_') and file.endswith('.npy'):
        ttv2 = np.load(os.path.join(folder2_path, file))
        dataset2 = file.replace('TTV_thickness_2D_', '').replace('.npy', '')
    if file.startswith('alignment_') and file.endswith('.npy'):
        alignment2 = np.load(os.path.join(folder2_path, file), allow_pickle=True).item()

x_mm = alignment1['x_shifted_mm']
y_mm = alignment1['y_shifted_mm']

# Assume no shift needed, use ttv2 directly
ttv2_shifted = ttv2

valid_mask = (ttv1 != 0) & (ttv2_shifted != 0)
difference = np.zeros_like(ttv1)
difference[valid_mask] = ttv1[valid_mask] - ttv2_shifted[valid_mask]

valid_diffs = difference[valid_mask]
stats = {
    'mean': np.mean(valid_diffs),
    'std': np.std(valid_diffs),
    'min': np.min(valid_diffs),
    'max': np.max(valid_diffs),
    'range': np.ptp(valid_diffs),
    'valid_fraction': np.sum(valid_mask) / np.prod(valid_mask.shape)
}

print(f"\n{dataset1} - {dataset2} Difference:")
print(f"  Mean: {stats['mean']:.3f} nm")
print(f"  Std:  {stats['std']:.3f} nm")
print(f"  Range: {stats['range']:.3f} nm")
print(f"  Valid data: {stats['valid_fraction']*100:.1f}%")

vmin, vmax = -200, 300
difference = difference - np.mean(valid_diffs)

# Define the specific range for the initial plot
x_start_idx, x_end_idx = 919, 1150
y_start_idx, y_end_idx = 1033, 1251
cropped_diff = difference[y_start_idx:y_end_idx, x_start_idx:x_end_idx]
cropped_x = x_mm[x_start_idx:x_end_idx] # array of x values in mm for the cropped region
cropped_y = y_mm[y_start_idx:y_end_idx] # array of y values in mm for the cropped region

# print(f"First few cropped x values (mm): {cropped_x[:5]}")
# print(f"First few cropped y values (mm): {cropped_y[:5]}")

# Plot 1: Initial cropped Thickness Difference
plt.figure(figsize=(7, 7))
im = plt.imshow(cropped_diff, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                extent=[cropped_x[0], cropped_x[-1], cropped_y[-1], cropped_y[0]])
im.set_clim(vmin, vmax)
plt.title(f'Initial Cropped Thickness Difference\nRange: {stats["range"]:.2f} nm')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
cbar = plt.colorbar(im, label='Difference (nm)')
cbar.set_ticks([vmin, -200, -100, 0, 100, 200, vmax])

print("Click on the upper-left corner, then the bottom-right corner of the polished region.")
points = plt.ginput(2, timeout=120, show_clicks=True)
plt.close()

if len(points) == 2:
    ul_x, ul_y = points[0] # assigning first selected point in UPPER LEFT CORNER to ul_x and ul_y
    br_x, br_y = points[1] # assigning second selected point in BOTTOM RIGHT CORNER to br_x and br_y
    
    # Adjust indices to be relative to the cropped region
    ul_x_idx = np.argmin(np.abs(cropped_x - ul_x)) + x_start_idx # finding the index of the x value in cropped_x that most closely matches ul_x by using argmin on the absolute difference, then adding x_start_idx to get the index in the full array
    ul_y_idx = np.argmin(np.abs(cropped_y - ul_y)) + y_start_idx # same for y
    br_x_idx = np.argmin(np.abs(cropped_x - br_x)) + x_start_idx # same for bottom right x
    br_y_idx = np.argmin(np.abs(cropped_y - br_y)) + y_start_idx # same for bottom right y
    
    x_start = min(ul_x_idx, br_x_idx) # getting the minimum x index, which should correspond to the upper left corner
    x_end = max(ul_x_idx, br_x_idx) + 1 # getting the maximum x index, which should correspond to the bottom right corner, adding 1 for slicing
    y_start = min(ul_y_idx, br_y_idx) # getting the minimum y index
    y_end = max(ul_y_idx, br_y_idx) + 1 # getting the maximum y index, adding 1 for slicing
    
    print(f"Polished region indices:")
    print(f"X start index: {x_start}, X end index: {x_end}") # 
    print(f"Y start index: {y_start}, Y end index: {y_end}")
    
    print(f"Corresponding mm ranges:")
    print(f"X: {x_mm[x_start]:.2f} mm to {x_mm[x_end-1]:.2f} mm")
    print(f"Y: {y_mm[y_start]:.2f} mm to {y_mm[y_end-1]:.2f} mm")
    
    # Calculate dimensions in mm
    x_dim = abs(x_mm[x_end-1] - x_mm[x_start]) # calculating the physical dimension in mm by taking the absolute difference between the x value at the end and start indices for the selected region
    y_dim = abs(y_mm[y_end-1] - y_mm[y_start]) # same for y dimension
    
    cropped_map = difference[y_start:y_end, x_start:x_end] # slicing the ORIGINAL difference map to get the SELECTED polished region
    cropped_x = x_mm[x_start:x_end]
    cropped_y = y_mm[y_start:y_end]
    
    plt.figure(figsize=(7, 7))
    plt.imshow(cropped_map, cmap='RdBu_r', extent=[cropped_x[0], cropped_x[-1], cropped_y[-1], cropped_y[0]])
    plt.title(f'Selected Polished Region\nDimensions: {x_dim:.2f}mm x {y_dim:.2f}mm\n X start: {cropped_x[0]:.2f}mm, X end: {cropped_x[-1]:.2f}mm\n Y start: {cropped_y[-1]:.2f}mm, Y end: {cropped_y[0]:.2f}mm')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.colorbar(label='Difference (nm)')
    plt.savefig(os.path.join(output_folder, 'polished_region_difference.png'))
    plt.show()

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
    plt.savefig(os.path.join(output_folder, 'mean_line_plots_polished_region.png'))
    plt.show()

    # # Line plots of thickness profiles at y and x coordinates closest to mean thickness
    # # Compute mean thickness difference across the region
    # mean_thickness = np.mean(cropped_map)

    # # Find indices where thickness is closest to the mean in y-direction
    # mean_thickness_y_idx = np.argmin(np.abs(np.mean(cropped_map, axis=1) - mean_thickness))
    # mean_thickness_x_idx = np.argmin(np.abs(np.mean(cropped_map, axis=0) - mean_thickness))

    # # Extract profiles at these indices
    # profile_at_mean_y_thickness = cropped_map[mean_thickness_y_idx, :]  # X-direction profile
    # profile_at_mean_x_thickness = cropped_map[:, mean_thickness_x_idx]  # Y-direction profile

    # # Plot them
    # plt.figure(figsize=(12, 5))

    # # Subplot for profile as function of x at y closest to mean thickness
    # plt.subplot(1, 2, 1)
    # plt.plot(cropped_x, profile_at_mean_y_thickness)
    # plt.title(f'Difference Profile vs X at Y ≈ {cropped_y[mean_thickness_y_idx]:.2f} mm (Mean Thickness)')
    # plt.xlabel('X (mm)')
    # plt.ylabel('Difference (nm)')
    # plt.grid(True)

    # # Subplot for profile as function of y at x closest to mean thickness
    # plt.subplot(1, 2, 2)
    # plt.plot(cropped_y, profile_at_mean_x_thickness)
    # plt.title(f'Difference Profile vs Y at X ≈ {cropped_x[mean_thickness_x_idx]:.2f} mm (Mean Thickness)')
    # plt.xlabel('Y (mm)')
    # plt.ylabel('Difference (nm)')
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, 'profiles_at_mean_thickness.png'))
    # plt.show()




# plt.figure(figsize=(7, 7))
# im = plt.imshow(difference[y_start:y_end, x_start:x_end], cmap='RdBu_r', vmin=vmin, vmax=vmax,
#                 extent=[cropped_x[0], cropped_x[-1], cropped_y[-1], cropped_y[0]])
# plt.title("Plot of selected region but found by slicing full map \n (should match previous plot)")
# plt.xlabel('X (mm)')
# plt.ylabel('Y (mm)')
# plt.colorbar(label='Difference (nm)')
# plt.show()