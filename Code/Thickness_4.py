import os
import numpy as np
import cv2  
import tkinter as tk
from tkinter import filedialog
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_images(folder_path):
    images, filenames, image_numbers = [], [], []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".tif") and "ref" not in filename.lower():
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            images.append(img)
            filenames.append(filename)
            match = re.search(r'\d+', filename)
            if match:
                image_numbers.append(int(match.group()))

    sorted_data = sorted(zip(image_numbers, images, filenames), key=lambda x: x[0])
    image_numbers, images, filenames = zip(*sorted_data)
    print(f"Images loaded: {len(images)}\n\nImage numbers: {image_numbers}\n")
    return list(images), list(filenames), list(image_numbers)


def read_settings_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select the settings file")
    print(f"Selected file: {file_path}")

    with open(file_path, 'r') as fle:
        descript = fle.readline()
        nframesTot = int(fle.readline().split()[0])
        nframes_start = int(fle.readline().split()[0])
        nframes_analyze = int(fle.readline().split()[0])
        x_center = int(fle.readline().split()[0])
        y_center = int(fle.readline().split()[0])
        radius = int(fle.readline().split()[0])
        smoothing = int(fle.readline().split()[0])
        sub = fle.readline().split()[0]
        fle.readline()  # Skip blank line

        wv, pw = [], []
        for line in fle:
            parts = line.split()
            if len(parts) >= 2:
                wv.append(float(parts[0]))
                pw.append(float(parts[1]))

    print(f"Wavelengths loaded: {len(wv)}")
    return {
        'descript': descript,
        'nframesTot': nframesTot,
        'wavelength': np.array(wv),
        'power': np.array(pw),
        'nframes_start': nframes_start,
        'nframes_analyze': nframes_analyze,
        'x_center': x_center,
        'y_center': y_center,
        'radius': radius,
        'smoothing': smoothing,
        'sub': sub
    }


def folder_path():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title="Select a folder with data")
    print(f"Selected folder: {path}")
    return path


def plot_center_pixel_intensity(frames_array, center_x, center_y, wavelengths, image_numbers, powers):
    patch_size = 4
    center_intensities = [
        np.mean(frame[center_y - patch_size:center_y + patch_size, center_x - patch_size:center_x + patch_size])
        for frame in frames_array
    ]

    plt.figure(figsize=(18, 4))
    plt.plot(wavelengths, center_intensities, 'o', markersize=5)

    for x, y, num in zip(wavelengths, center_intensities, image_numbers):
        plt.text(x, y, str(num), fontsize=8, ha='right', va='bottom')

    plt.title("Average Intensity of Center Pixels vs Wavelength")
    plt.xlabel('Wavelength (nm)')
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    plt.ylabel('Intensity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return center_intensities


def print_intensity_and_wavelength(frames_array, center_x, center_y, wavelengths, image_numbers, powers):
    patch_size = 4
    scaled_intensities = []

    print("\nImage Intensity and Wavelengths:")
    print("Image Number\tWavelength(nm)\t  Average Intensity")
    print("-" * 50)

    for frame, wl, num, power in zip(frames_array, wavelengths, image_numbers, powers):
        try:
            patch = frame[center_y - patch_size:center_y + patch_size, center_x - patch_size:center_x + patch_size]
            avg_intensity = np.mean(patch)
            scaled = avg_intensity / power
            scaled_intensities.append((num, wl, scaled))
        except Exception as e:
            print(f"Error processing image {num}, wavelength {wl:.4f}: {e}")

    if not scaled_intensities:
        print("No valid data to process.")
        return

    max_intensity = max(scaled for _, _, scaled in scaled_intensities)

    for num, wl, scaled in scaled_intensities:
        normalized = scaled / max_intensity * 100
        print(f"{num}\t\t{wl:.4f}\t\t{normalized:.4f}")

    return normalized


# --------- Main Execution --------- #
plt.close('all')

selected_folder = folder_path()
out = read_settings_file()

images, filenames, image_numbers = load_images(selected_folder)

if len(images) < 2:
    print("Error: At least two images are required.")
else:
    normalized = print_intensity_and_wavelength(
        images,
        out['x_center'],
        out['y_center'],
        out['wavelength'],
        image_numbers,
        out['power']
    )
    center_intensities = plot_center_pixel_intensity(
        images,
        out['x_center'],
        out['y_center'],
        out['wavelength'],
        image_numbers,
        out['power']
    )
