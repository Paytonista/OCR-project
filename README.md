# OCR Image Processor

A Python tool that extracts text from images using [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), with multiple image preprocessing pipelines to improve recognition accuracy. Comes with both a graphical UI and a command-line interface.

---

## What it does

Raw images often produce poor OCR results due to noise, skew, or low contrast. This tool runs each image through several preprocessing techniques and shows you the OCR output for each, so you can pick whichever works best for your image.

| Preprocessing | What it does |
|---|---|
| **Inverted** | Flips pixel values — useful for white-on-dark text |
| **Grayscaled** | Strips colour to reduce noise |
| **Binarized** | Converts to pure black & white via thresholding |
| **Denoised** | Applies morphological operations to remove speckles |
| **Dilation & Erosion** | Thickens/thins text strokes to improve character shapes |
| **Deskew** *(optional)* | Auto-detects and corrects image rotation before processing |

---

## Requirements

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system
  - Windows: download from [UB-Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki)
  - macOS: `brew install tesseract`
  - Linux: `sudo apt install tesseract-ocr`

Install Python dependencies:

```bash
pip install opencv-python Pillow pytesseract matplotlib numpy
```

---

## Usage

### GUI (recommended)

```bash
python app.py
```

1. Click **Browse** to select an image file
2. Optionally check **Apply deskew** if the image appears rotated
3. Click **Process**
4. Use the dropdown to switch between preprocessing results and compare OCR output

### Command line

```bash
python main.py
```

Place your image in the `data/` folder and enter the filename when prompted. All preprocessing results are printed to the terminal.

---

## Project structure

```
OCR-project/
├── main.py      # Image processing functions + CLI entry point
├── app.py       # Tkinter GUI
├── data/        # Place input images here
└── temp/        # Intermediate debug images (auto-created)
```

---

## How it works

1. The image is loaded with OpenCV
2. Optionally deskewed using contour-based angle detection
3. Passed through each preprocessing function
4. Each preprocessed variant is fed directly into Tesseract via `pytesseract`
5. Results are displayed side-by-side in the UI (or printed in the CLI)
