from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
import os


# ----------------------------------------------DISPLAY---------------------------------------------
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.show()


# ==============================================================================================
# ----------------------------IMAGE PROCESSING--------------------------------------------------

def getSkewAngle(cvImage) -> float:
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)


# ------------------------------------------------------------------

def invert(image):
    return cv2.bitwise_not(image)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(image):
    _, binarized = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binarized


def denoise(image):
    kernel = np.ones((3, 3), np.uint8)          # fixed: was (1,1) — no-op
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


def dilation_and_erosion(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)  # fixed: was iterations=0 — no-op
    image = cv2.bitwise_not(image)
    return image


# ------------------------------------------------------------------

def _ocr(cv_image):
    """Run OCR on a cv2/numpy image without a disk round-trip."""
    pil_image = Image.fromarray(cv_image)
    return pytesseract.image_to_string(pil_image)


def run_pipeline(image):
    """Run all preprocessing steps and return {label: ocr_text}."""
    gray = grayscale(image)
    bw = binarize(gray)

    steps = {
        "Inverted":            invert(image),
        "Grayscaled":          gray,
        "Binarized":           bw,
        "Denoised":            denoise(bw),
        "Dilation & Erosion":  dilation_and_erosion(image),
    }

    return {label: _ocr(img) for label, img in steps.items()}


# ------------------------MAIN---------------------------------

if __name__ == "__main__":
    filename = input("Enter the name of the file: ")
    child_dir = "data"
    full_path = os.path.join(os.getcwd(), child_dir, filename)

    image = cv2.imread(full_path)
    if image is None:
        print(f"Error: could not load '{full_path}'")
        exit(1)

    os.makedirs("temp", exist_ok=True)

    print("\n--- Original image ---")
    for label, text in run_pipeline(image).items():
        print(f"\n[{label.upper()}]\n{text}")

    print("\n--- Deskewed image ---")
    for label, text in run_pipeline(deskew(image)).items():
        print(f"\n[{label.upper()} + DESKEW]\n{text}")
