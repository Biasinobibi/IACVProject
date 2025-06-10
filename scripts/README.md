# compute_alpha.py

A safe, command‐line wrapper around a Closed‐Form Matting implementation.  
This script reads an input image and corresponding scribbles, automatically handles orientation and sizing, computes an alpha matte, and writes the result to disk—without opening any display windows.

```bash
python compute_alpha.py input.png input_scribbles.png -m 800 -o output/alpha/input_alpha.png
````

* **`input.png`**: Path to the original image.
* **`input_scribbles.png`**: Path to your scribbles mask (must match or be rotatable to match the input’s dimensions).
* **`-m 800`**: (Optional) Maximum dimension (width or height) for resizing. Here, both image and scribbles will be scaled so their largest side is 800 px. Use `-m 0` to disable resizing.
* **`-o output/alpha/input_alpha.png`**: (Optional) Path (including filename) where the resulting alpha matte will be saved. Defaults to `alpha.png` in the current directory.
