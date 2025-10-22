import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tkinter as tk
from tkinter import filedialog

def subtract_ttv_maps(folder1_path, folder2_path, output_folder, img1_path=None, img2_path=None):
    """Subtract two TTV thickness maps, using FFT-based shift if images are provided."""
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
    
    if 'ttv1' not in locals() or 'ttv2' not in locals():
        raise FileNotFoundError("TTV_thickness_2D_*.npy not found in one or both folders.")
    if 'alignment1' not in locals() or 'alignment2' not in locals():
        raise FileNotFoundError("alignment_*.npy not found in one or both folders.")
    
    if ttv1.shape != ttv2.shape:
        raise ValueError(f"Map shapes don't match: {ttv1.shape} vs {ttv2.shape}")
    
    x_mm = alignment1['x_shifted_mm']
    y_mm = alignment1['y_shifted_mm']
    
    # Initialize shifts
    final_shift_x, final_shift_y = 0, 0
    fft_shift = None
    
    center_shift_x = alignment1['x_center'] - alignment2['x_center']
    center_shift_y = alignment1['y_center'] - alignment2['y_center']
    print(f"Metadata-based center alignment: dx={center_shift_x}, dy={center_shift_y} pixels")
    if abs(center_shift_x) > 1 or abs(center_shift_y) > 1:
        final_shift_x, final_shift_y = center_shift_x, center_shift_y
    
    # Apply shift if significant
    if abs(final_shift_x) > 1 or abs(final_shift_y) > 1:
        from scipy.ndimage import shift
        print(f"Applying shift: dx={final_shift_x}, dy={final_shift_y} pixels")
        ttv2_shifted = shift(ttv2, (final_shift_y, final_shift_x), mode='constant', cval=0)
    else:
        ttv2_shifted = ttv2


    valid_mask = (ttv1 != 0) & (ttv2_shifted != 0) #
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

    vmin, vmax = -300, 300

    difference = difference - np.mean(valid_diffs) # adjusting to account for fact that zero for the two TTV maps may not correspond to same physical thickness

    # Plot 1: Thickness Difference
    plt.figure(figsize=(7, 7))
    im = plt.imshow(difference, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                    extent=[x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]])
    plt.title(f'Thickness Difference\nRange: {stats["range"]:.2f} nm')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    im.set_clim(vmin, vmax)
    cbar = plt.colorbar(im, label='Difference (nm)')
    cbar.set_ticks([vmin, -200,-100, 0, 100, 200, vmax])
    plt.savefig(os.path.join(output_folder, 'thickness_difference.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot 2: Valid Subtraction Region
    plt.figure(figsize=(7, 7))
    plt.imshow(valid_mask, cmap='gray', extent=[x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]])
    plt.title('Valid Subtraction Region')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(os.path.join(output_folder, 'valid_subtraction_region.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Difference Distribution
    plt.figure(figsize=(7, 7))
    plt.hist(valid_diffs, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(stats['mean'], color='red', linestyle='--',
                label=f'Mean: {stats["mean"]:.2f} nm')
    plt.xlabel('Difference (nm)')
    plt.ylabel('Frequency')
    plt.title('Difference Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'difference_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Center Cross-sections
    plt.figure(figsize=(7, 7))
    center_y = len(y_mm) // 2
    center_x = len(x_mm) // 2
    plt.plot(x_mm, difference[center_y, :], 'b-', label='Horizontal')
    plt.plot(y_mm, difference[:, center_x], 'r-', label='Vertical')
    plt.xlabel('Position (mm)')
    plt.ylabel('Difference (nm)')
    plt.title('Center Cross-sections')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, 'center_cross_sections.png'), dpi=300, bbox_inches='tight')
    plt.close()

    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, 'difference_map.npy'), difference)
    np.save(os.path.join(output_folder, 'valid_mask.npy'), valid_mask)

    import matplotlib.image as mpimg

    # List of PNG file paths (replace with your actual file paths)
    png_files = [
        r"C:\Users\Emma Martignoni\Desktop\2025-10-09 FJPtest BEFORE\2025-10-09 FJPtest BEFORE_Plots_10_16_2025\Plot_5_mm_Horizontal_Thickness.png",
        r"C:\Users\Emma Martignoni\Desktop\2025-10-09 FJPtest BEFORE\2025-10-09 FJPtest BEFORE_Plots_10_16_2025\Plot_6_mm_Vertical_Thickness.png",
        r"C:\Users\Emma Martignoni\Desktop\2025-10-17 FJPtest AFTER\2025-10-17 FJPtest AFTER_Plots_10_17_2025\Plot_5_mm_Horizontal_Thickness.png",
        r"C:\Users\Emma Martignoni\Desktop\2025-10-17 FJPtest AFTER\2025-10-17 FJPtest AFTER_Plots_10_17_2025\Plot_6_mm_Vertical_Thickness.png"
    ]

    # Create a 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, png_file in zip(axes.flat, png_files):
        img = mpimg.imread(png_file)
        ax.imshow(img)
        ax.axis('off')  # Hide axes for cleaner display
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'thickness_cross_sections_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

    summary = {
        **stats,
        'dataset1': dataset1,
        'dataset2': dataset2,
        'center_shift': (alignment1['x_center'] - alignment2['x_center'], alignment1['y_center'] - alignment2['y_center']),
        'fft_shift': fft_shift if fft_shift else None
    }
    with open(os.path.join(output_folder, 'difference_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Results saved to: {output_folder}")
    return difference, stats

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    
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
    
    # Prompt user to select .tif image files
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
    
    # Prompt for output folder name
    output_folder_name = input("Enter the name for the output folder on the Desktop: ")
    output_dir = os.path.join(os.path.expanduser("~"), "Desktop", output_folder_name)
    
    # Run TTV subtraction with image paths
    difference, stats = subtract_ttv_maps(folder1, folder2, output_dir, img1_path, img2_path)