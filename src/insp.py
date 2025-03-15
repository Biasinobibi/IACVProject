import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_alpha_matte(image):
    """
    Computes the alpha matte of a motion-blurred image.
    White represents the moving object (alpha = 1),
    Black represents the background (alpha = 0).

    Parameters:
    - image: Blurred input image (BGR format)

    Returns:
    - alpha_matte: Computed alpha matte (grayscale transparency map)
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Estimate background as the brightest pixel in the image (static background)
    background_intensity = np.max(gray_image)

    # To avoid divide by zero errors, we use np.maximum to ensure the denominator is at least 1
    alpha_matte = gray_image / np.maximum(gray_image + background_intensity, 1)

    # Clip values to be between 0 and 1
    alpha_matte = np.clip(alpha_matte, 0, 1)

    # Apply Gaussian Blur to smooth the edges of the alpha matte (increased kernel size for more smoothing)
    alpha_matte = cv2.GaussianBlur(alpha_matte, (31, 31), 4)

    # Use Canny edge detection to find the edges of the object
    edges = cv2.Canny((alpha_matte * 255).astype(np.uint8), 100, 200)

    # Apply the edges as a mask to highlight the object edges
    alpha_matte_with_edges = alpha_matte.copy()

    # Increase alpha values near the edges for transparency
    alpha_matte_with_edges[edges > 0] = 22.5  # Modify this value to control edge transparency

    # Apply a threshold to further clean up the matte
    _, alpha_matte_with_edges = cv2.threshold(alpha_matte_with_edges, 0.38, 1, cv2.THRESH_BINARY)

    return alpha_matte_with_edges


def visualize_results(image, alpha_matte):
    """
    Visualizes the original image and the computed alpha matte.
    """
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    # Alpha matte
    plt.subplot(1, 2, 2)
    plt.imshow(alpha_matte, cmap='gray')
    plt.title("Alpha Matte with Edge Shades")
    plt.axis("off")

    plt.show()


# Example Usage
if __name__ == "__main__":
    # Load the motion-blurred image (use your own path or image)
    image = cv2.imread("../assets/images/walkingman.png")

    # Compute alpha matte with edge highlighting
    alpha_matte = compute_alpha_matte(image)

    # Visualize the results
    visualize_results(image, alpha_matte)

#Gaussian Blur (cv2.GaussianBlur): Increased the kernel size to (31, 31) and set the standard deviation to 4 to create a more pronounced blur, smoothing out the edges of the moving object.

#Edge Detection (cv2.Canny): Used the Canny Edge Detector to find the edges in the alpha matte. The edges are highlighted in the alpha matte by increasing their transparency. You can adjust the threshold values (100 and 200) for more or less sensitivity to edges.

#Edge Highlighting: After finding the edges, I applied a mask to the alpha matte, setting the alpha value to 0.5 for the edges to make them semi-transparent. You can modify the 0.5 value to adjust how transparent or opaque the edges should appear. A lower value will make them more transparent.

#Thresholding: After applying the edge mask, I applied a thresholding step (0.3) to further distinguish the foreground from the background. You can adjust the threshold to fine-tune the transparency level.