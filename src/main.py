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

    # Compute the alpha matte (alpha = foreground intensity / (foreground + background))
    alpha_matte = gray_image / (gray_image + background_intensity)

    # Clip values to be between 0 and 1
    alpha_matte = np.clip(alpha_matte, 0, 1)

    return alpha_matte


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
    plt.title("Alpha Matte")
    plt.axis("off")

    plt.show()


# Example Usage
if __name__ == "__main__":
    # Load the motion-blurred image (use your own path or image)
    image = cv2.imread("../assets/images/flyingball.PNG")

    # Compute alpha matte
    alpha_matte = compute_alpha_matte(image)

    # Visualize the results
    visualize_results(image, alpha_matte)
