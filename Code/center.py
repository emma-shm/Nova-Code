import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os


def plot_circle_on_grayscale_image(image, center, radius):
    color=(255, 0, 0)
    thickness=4
    
    # Convert grayscale to RGB for colored drawing
    if image.dtype == 'uint16':
        image_display = (image / 256).astype('uint8')  # Convert 16-bit to 8-bit for display
    else:
        image_display = image

    # Convert single channel grayscale image to three-channel RGB
    image_rgb = cv2.cvtColor(image_display, cv2.COLOR_GRAY2RGB)

    # Draw a red circle on the image (color in RGB)
    cv2.circle(image_rgb, center, radius, color, thickness)

    print(f"\n")

    # Display the image with the circle
    plt.figure(radius)
    plt.title(f"Center (x: {center[0]}, y: {center[1]}), R =  {radius}")
    plt.imshow(image_rgb)
    plt.axis('on')  
    plt.show()


def find_circle_center(image_path, downscale_factor=2, min_radius=0, max_radius=0):
    # Load the 16-bit grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return None

    # Downscale the image to reduce processing time
    image_resized = cv2.resize(image, 
                               (image.shape[1] // downscale_factor, image.shape[0] // downscale_factor))

    # Normalize the 16-bit image to an 8-bit image for processing
    image_8bit = cv2.normalize(image_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(image_8bit, (5, 5), 2)

    # Use the Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, 
                               param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)

    # If at least one circle is detected
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Adjust the center coordinates according to the downscale factor
            center_x = x * downscale_factor
            center_y = y * downscale_factor
            print(f"Circle center: (x: {center_x}, y: {center_y})\nRadius: {r * downscale_factor}\n")

            center = (center_x, center_y)
            plot_circle_on_grayscale_image(image, center, r*downscale_factor)
            return (center_x, center_y)
    else:
        print("No circles were detected.")
        return None

    #return (image)

def get_user_input():
    # Get center point (x, y) from the user
    x = int(input("Enter the x-coordinate of the center: "))
    y = int(input("Enter the y-coordinate of the center: "))
    center = (x, y)

    # Get the radius from the user
    radius = int(input("Enter the radius of the circle: "))

    return center, radius

# Create a Tkinter root window
# Open a file dialog to select a file
root = tk.Tk()
root.withdraw()  # Hide the root window
file = filedialog.askopenfilename(title="Select a Ref File")
file_name = os.path.basename(os.path.dirname(file)) # extract folder name

print(f"Selected file: {file}")
print(f"\nWafer: {file_name}")
c = find_circle_center(file)
image = cv2.imread(file, cv2.IMREAD_UNCHANGED)


# Main loop to prompt the user to run again
def main():

    run_again = 'y'
    

    
    while run_again.lower() == 'y':


        # Ask the user if they want to run again
        run_again = input("Do you want to run it again? (y/n): ")
        if run_again == 'n':
            break


        # Prompt the user for center and radius
        center, radius = get_user_input()
        

        # Plot the image with the user-defined circle
        plot_circle_on_grayscale_image(image, center, radius)



# Run the main function
if __name__ == "__main__":
    main()
