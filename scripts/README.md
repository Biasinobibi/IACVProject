# Closed-Form Matting Wrapper

A safe, command‐line wrapper around a Closed‐Form Matting implementation.  
This script reads an input image and corresponding scribbles, automatically handles orientation and sizing, computes an alpha matte, and writes the result to disk—without opening any display windows.

```bash
python process_alpha_matte.py input.jpg scribbles.png -m 800 -o output/alpha_matte.png
````

* **`input.jpg`**: Path to the original image.
* **`scribbles.png`**: Path to your scribbles mask (must match or be rotatable to match the input’s dimensions).
* **`-m 800`**: (Optional) Maximum dimension (width or height) for resizing. Here, both image and scribbles will be scaled so their largest side is 800 px. Use `-m 0` to disable resizing.
* **`-o output/alpha_matte.png`**: (Optional) Path (including filename) where the resulting alpha matte will be saved. Defaults to `alpha.png` in the current directory.
