# MNIST Handwritten Digit Recognition with an MLP

This project trains a Multi-Layer Perceptron (MLP) neural network to recognize handwritten digits using the MNIST dataset, and tests its performance on both the MNIST test set and our group's own collected handwritten digit images.

## Requirements

The project uses Python 3.10+ and the following libraries:

- `torch` (PyTorch)
- `torchvision`
- `numpy`
- `pandas`
- `scikit-learn`
- `pillow` (PIL)
- `matplotlib`
- `seaborn`
- `jupyter`

## Installation

Install all required packages with:

If running with GPU support (NVIDIA CUDA), install the CUDA build of PyTorch instead:

# Running the Project

### Option 1 — Google Colab (recommended)

1. Open `MNIST_MLP_Project.ipynb` in Google Colab.
2. Mount your Google Drive when prompted, and upload the `digits/` folder to your Drive (e.g. to `MyDrive/digits/`).
3. Update the path in the `ProjectDataLoader` call if needed:
```python
   images, labels = ProjectDataLoader('/content/drive/MyDrive/digits', invert=False)
```
4. Click **Runtime → Run all** to execute all cells.

### Option 2 — Local Jupyter Notebook

1. Place `MNIST_MLP_Project.ipynb` and the `digits/` folder in the same directory.
2. Update the path in the `ProjectDataLoader` call:
```python
   images, labels = ProjectDataLoader('./digits', invert=False)
```
3. Launch Jupyter:
4. Run all cells (**Cell → Run All**).

## Image Filename Convention

Group images must follow the naming convention:

`<digit>-<groupID>-<memberID>.png`

For example, `3-2-5.png` represents a digit "3" from group 2, member 5. The first character of the filename is parsed as the digit label. Both `.png` and `.jpg`/`.jpeg` formats are supported.

## Notebook Sections

The notebook is organized into the following sections:

1. **Imports and Setup** — Loads libraries and detects GPU availability.
2. **Load MNIST Data** — Downloads MNIST and splits it into train (50,000), validation (10,000), and test (10,000) sets.
3. **Iteration 1 — Baseline MLP** — Trains a simple two-hidden-layer MLP (128, 64).
4. **Iteration 2 — Larger Network + Dropout** — Trains a larger MLP (512, 256) with dropout regularization.
5. **Hyperparameter Sweep** — Systematically searches across 8 configurations to find the best hyperparameters.
6. **Iteration 3 — Tuned + Data Augmentation** — Uses the best hyperparameters from the sweep, combined with random rotations and translations on the training data.
7. **Iteration Comparison** — Compares MNIST test accuracy across all three iterations.
8. **Save Best Model** — Saves the best-performing model to `best_mlp.pth`.
9. **Load Group Images (Task 2)** — Defines and calls the `ProjectDataLoader` function to read group images.
10. **Evaluate on Group Images** — Tests the best model on the group's handwritten digits, including per-digit accuracy and misclassified examples.
11. **Confusion Matrix and Performance Gap Analysis** — Visualizes detailed error patterns and compares MNIST vs. group performance.
12. **Final Results Summary** — Prints overall accuracy statistics.


## ProjectDataLoader Function

To load group images from a folder into NumPy arrays:

```python
from data_loader import ProjectDataLoader

images, labels = ProjectDataLoader(
    folder_path='digits',   # path to the folder of PNG/JPG images
    invert=False,           # set True if your images are dark on light (to match MNIST style)
    verbose=True            # print a summary after loading
)
```

Returns:
- `images`: NumPy array of shape `(N, 28, 28)`, dtype `uint8`
- `labels`: NumPy array of shape `(N,)`, dtype `int64`

## Notes

- The model architecture is restricted to fully connected (Linear) layers only, no convolutional layers.
- The notebook automatically detects and uses a GPU if one is available. Otherwise it falls back to CPU.
- Results may vary slightly between runs due to random weight initialization, data shuffling, and dropout.
