import cv2
import numpy as np
import os
import sys
from src.Davide import closed_form_matting


def process_alpha_matte(image_path, scribbles_path, max_dimension=500, display=True, save=True):
    """
    Process images to generate an alpha matte using closed form matting.
    
    Args:
        image_path (str): Path to the original image
        scribbles_path (str): Path to the scribbles image
        max_dimension (int): Maximum allowed dimension (width or height) for processing
                            Set to None to disable resizing
        display (bool): Whether to display the result in a window
        save (bool): Whether to save the result to file
    
    Returns:
        numpy.ndarray: The computed alpha matte (at original size)
    """
    # Load the original image and scribbles
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_scribbles = cv2.imread(scribbles_path, cv2.IMREAD_COLOR)
    
    if original_image is None or original_scribbles is None:
        raise ValueError(f"Could not load image files: {image_path} or {scribbles_path}")
    
    # Save original dimensions for later
    original_height, original_width = original_image.shape[:2]
    scale_factor = 1.0
    
    # Resize if necessary
    if max_dimension is not None:
        # Check if resizing is needed
        if original_height > max_dimension or original_width > max_dimension:
            # Calculate scale factor to match max dimension
            scale_factor = max_dimension / max(original_height, original_width)
            
            # Compute new dimensions
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            print(f"Resizing images from {original_width}x{original_height} to {new_width}x{new_height} to avoid memory issues")
            
            # Resize images for processing
            image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            scribbles = cv2.resize(original_scribbles, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            # No resizing needed
            image = original_image.copy()
            scribbles = original_scribbles.copy()
    else:
        # No resizing requested
        image = original_image.copy()
        scribbles = original_scribbles.copy()
    
    # Normalize to [0, 1] range for matting
    image = image / 255.0
    scribbles = scribbles / 255.0
    
    # Compute the alpha matte
    print("Computing alpha matte...")
    alpha = closed_form_matting.closed_form_matting_with_scribbles(image, scribbles)
    
    # Resize alpha back to original dimensions if it was resized
    if scale_factor < 1.0:
        print("Resizing alpha matte back to original dimensions...")
        alpha = cv2.resize(alpha, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    
    if save:
        # Extract just the filename without extension for output naming
        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create output directory if it doesn't exist
        output_dir = "../../output/alpha"
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output path
        output_path = os.path.join(output_dir, f"{image_filename}_alpha.png")
        
        # Save the alpha matte as a PNG file
        alpha_scaled = (alpha * 255).astype(np.uint8)
        cv2.imwrite(output_path, alpha_scaled)
        print(f"Alpha matte saved to: {output_path}")
    
    if display:
        # Display the alpha matte (resize for display if it's very large)
        display_alpha = alpha.copy()
        if original_height > 800 or original_width > 1200:
            display_scale = min(800 / original_height, 1200 / original_width)
            display_width = int(original_width * display_scale)
            display_height = int(original_height * display_scale)
            display_alpha = cv2.resize(display_alpha, (display_width, display_height), interpolation=cv2.INTER_AREA)
        
        cv2.imshow("Alpha Matte", display_alpha)
    
    return alpha

def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) < 3:
        print("Usage: python3 main.py image.png scribble.png [max_dimension]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    scribbles_path = sys.argv[2]
    
    # Optional max dimension parameter
    max_dimension = 500  # Default value
    if len(sys.argv) >= 4:
        try:
            max_dimension = int(sys.argv[3])
        except ValueError:
            print("Warning: max_dimension must be an integer. Using default value of 500.")
    
    process_alpha_matte(image_path, scribbles_path, max_dimension=max_dimension)

if __name__ == "__main__":
    main()