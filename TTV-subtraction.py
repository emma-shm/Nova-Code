import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import shift
import matplotlib.image as mpimg

def load_ttv_data(folder1_path, folder2_path):
    """Load TTV maps and alignment data from folders."""
    ttv1 = None
    dataset1 = None
    alignment1 = None
    for file in os.listdir(folder1_path):
        if file.startswith('TTV_thickness_2D_') and file.endswith('.npy'):
            ttv1 = np.load(os.path.join(folder1_path, file))
            dataset1 = file.replace('TTV_thickness_2D_', '').replace('.npy', '')
        if file.startswith('alignment_') and file.endswith('.npy'):
            alignment1 = np.load(os.path.join(folder1_path, file), allow_pickle=True).item()
    
    ttv2 = None
    dataset2 = None
    alignment2 = None
    for file in os.listdir(folder2_path):
        if file.startswith('TTV_thickness_2D_') and file.endswith('.npy'):
            ttv2 = np.load(os.path.join(folder2_path, file))
            dataset2 = file.replace('TTV_thickness_2D_', '').replace('.npy', '')
        if file.startswith('alignment_') and file.endswith('.npy'):
            alignment2 = np.load(os.path.join(folder2_path, file), allow_pickle=True).item()
    
    if ttv1 is None or ttv2 is None:
        raise FileNotFoundError("TTV_thickness_2D_*.npy not found in one or both folders.")
    if alignment1 is None or alignment2 is None:
        raise FileNotFoundError("alignment_*.npy not found in one or both folders.")
    
    if ttv1.shape != ttv2.shape:
        raise ValueError(f"Map shapes don't match: {ttv1.shape} vs {ttv2.shape}")
    
    return ttv1, ttv2, alignment1, alignment2, dataset1, dataset2

# Initialize Tkinter root and hide it
root = tk.Tk()
root.withdraw()

# Prompt user to select folders
folder1 = filedialog.askdirectory(title="Select the 'before' TTV data folder (for before - after subtraction)")
if not folder1:
    print("No 'before' folder selected. Exiting.")
    exit()
print(f"Selected 'before' folder: {folder1}")

folder2 = filedialog.askdirectory(title="Select the 'after' TTV data folder (for before - after subtraction)")
if not folder2:
    print("No 'after' folder selected. Exiting.")
    exit()
print(f"Selected 'after' folder: {folder2}")

# Prompt user to select .tif image files (optional for future FFT alignment)
img1_path = filedialog.askopenfilename(
    title="Select the 'before' reference .tif image",
    initialdir=folder1,
    filetypes=[("TIFF files", "*.tif *.tiff")]
)
if not img1_path:
    print("No 'before' image selected.")
else:
    print(f"Selected 'before' image: {img1_path}")

img2_path = filedialog.askopenfilename(
    title="Select the 'after' reference .tif image",
    initialdir=folder2,
    filetypes=[("TIFF files", "*.tif *.tiff")]
)
if not img2_path:
    print("No 'after' image selected.")
else:
    print(f"Selected 'after' image: {img2_path}")

# Prompt for output folder name
output_folder_name = input("Enter the name for the output folder on the Desktop: ")
output_dir = os.path.join(os.path.expanduser("~"), "Desktop", output_folder_name)
os.makedirs(output_dir, exist_ok=True)

# Load data
ttv1, ttv2, alignment1, alignment2, dataset1, dataset2 = load_ttv_data(folder1, folder2)
x_mm = alignment1['x_shifted_mm']
y_mm = alignment1['y_shifted_mm']

# Metadata-based center shifts (calculated but applied only if significant; starting with no shift as per current setup)
center_shift_x = alignment1['x_center'] - alignment2['x_center']
center_shift_y = alignment1['y_center'] - alignment2['y_center']
print(f"Metadata-based center alignment: dx={center_shift_x}, dy={center_shift_y} pixels")

# Start with no initial shift (as per "no shifts" in current code); manual adjustments will be added iteratively
current_shift_x = 0.0
current_shift_y = 0.0
print("Starting with no automatic shift for manual alignment adjustments.")

fft_shift = None  # Placeholder for potential FFT-based shift if images are used in the future

# Interactive loop for subtraction, visualization, and manual shifting
while True:
    print(f"Applying current shift to second map (ttv2): dy={current_shift_y}, dx={current_shift_x} pixels")
    ttv2_shifted = shift(ttv2, (current_shift_y, current_shift_x), mode='constant', cval=0.0)

    valid_mask = (ttv1 != 0) & (ttv2_shifted != 0)
    difference = np.zeros_like(ttv1)
    difference[valid_mask] = ttv1[valid_mask] - ttv2_shifted[valid_mask]
    
    valid_diffs = difference[valid_mask]
    stats = {
        'mean': np.mean(valid_diffs) if len(valid_diffs) > 0 else 0,
        'std': np.std(valid_diffs) if len(valid_diffs) > 0 else 0,
        'min': np.min(valid_diffs) if len(valid_diffs) > 0 else 0,
        'max': np.max(valid_diffs) if len(valid_diffs) > 0 else 0,
        'range': np.ptp(valid_diffs) if len(valid_diffs) > 0 else 0,
        'valid_fraction': np.sum(valid_mask) / np.prod(valid_mask.shape)
    }

    print(f"\n{dataset1} - {dataset2} Difference:")
    print(f"  Mean: {stats['mean']:.3f} nm")
    print(f"  Std:  {stats['std']:.3f} nm")
    print(f"  Range: {stats['range']:.3f} nm")
    print(f"  Valid data: {stats['valid_fraction']*100:.1f}%")

    # Adjust difference to account for potential offset in zero levels
    mean_diff = stats['mean']
    difference[valid_mask] -= mean_diff  # Apply adjustment only on valid pixels

    vmin, vmax = -300, 300

    # Plot 1: Thickness Difference (shown interactively)
    plt.figure(figsize=(10,10))
    im = plt.imshow(difference, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                    extent=[x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]])
    plt.title(f'Thickness Difference\nRange: {stats["range"]:.2f} nm (mean adjusted by {mean_diff:.3f} nm)\ndx={current_shift_x:.2f}, dy={current_shift_y:.2f} pixels')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    im.set_clim(vmin, vmax)
    cbar = plt.colorbar(im, label='Difference (nm)')
    cbar.set_ticks([vmin, -200, -100, 0, 100, 200, vmax])
    plt.savefig(os.path.join(output_dir, 'thickness_difference.png'), dpi=300, bbox_inches='tight')
    plt.show()  # Blocks until user closes the window
    plt.close()

    # Plot 2: Valid Subtraction Region (saved only)
    plt.figure(figsize=(10,10))
    plt.imshow(valid_mask, cmap='gray', extent=[x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]])
    plt.title('Valid Subtraction Region')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(os.path.join(output_dir, 'valid_subtraction_region.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Prompt for manual shift
    response = input("Would you like to shift the images for better alignment? (y/n): ").strip().lower()
    if response != 'y':
        # Finalize: Plot and save cross-sections comparison (hardcoded paths; done once at the end)
        png_files = [
            r"C:\Users\Emma Martignoni\Desktop\2025-10-09 FJPtest BEFORE\2025-10-09 FJPtest BEFORE_Plots_10_16_2025\Plot_5_mm_Horizontal_Thickness.png",
            r"C:\Users\Emma Martignoni\Desktop\2025-10-09 FJPtest BEFORE\2025-10-09 FJPtest BEFORE_Plots_10_16_2025\Plot_6_mm_Vertical_Thickness.png",
            r"C:\Users\Emma Martignoni\Desktop\2025-10-17 FJPtest AFTER\2025-10-17 FJPtest AFTER_Plots_10_17_2025\Plot_5_mm_Horizontal_Thickness.png",
            r"C:\Users\Emma Martignoni\Desktop\2025-10-17 FJPtest AFTER\2025-10-17 FJPtest AFTER_Plots_10_17_2025\Plot_6_mm_Vertical_Thickness.png"
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for ax, png_file in zip(axes.flat, png_files):
            if os.path.exists(png_file):
                img = mpimg.imread(png_file)
                ax.imshow(img)
            ax.axis('off')  # Hide axes for cleaner display
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'thickness_cross_sections_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Save intermediate NumPy files (overwritten each iteration)
        np.save(os.path.join(output_dir, 'difference_map.npy'), difference)
        np.save(os.path.join(output_dir, 'valid_mask.npy'), valid_mask)

        # Save final summary (includes applied shifts)
        summary = {
            **stats,
            'dataset1': dataset1,
            'dataset2': dataset2,
            'center_shift': (center_shift_x, center_shift_y),
            'applied_shift': (current_shift_x, current_shift_y),
            'fft_shift': fft_shift
        }
        with open(os.path.join(output_dir, 'difference_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"Results saved to: {output_dir}")
        break
    else:
        # Get manual shift inputs (added to current totals; supports sub-pixel with float)
        try:
            dx = float(input("Enter pixels to shift second image left/right (positive: shift right, negative: shift left): "))
            dy = float(input("Enter pixels to shift second image up/down (positive: shift down, negative: shift up): "))
            current_shift_x += dx
            current_shift_y += dy
            print(f"Added shift: dx={dx}, dy={dy}. New totals: dx={current_shift_x}, dy={current_shift_y}")
        except ValueError:
            print("Invalid input; no shift applied this iteration.")
        # Loop continues: re-compute subtraction with updated shifts