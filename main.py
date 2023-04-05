from PIL import Image
import cv2
import numpy
from matplotlib import pyplot as plt
import pytesseract
import os



#----------------------------------------------DISPLAY---------------------------------------------
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()
#========================================================================================================
#----------------------------IMAGE_PROCESSING--FUNCTION_NAMES_ARE_SELF_EXPLANATORY--------------------------------------

#IMPORTANT

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage
# Deskew image

def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

#------------------------------------------------------------------

def invert(image):
    

    inverted_image = cv2.bitwise_not(image)
    return inverted_image

def grayscale(image):

    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscaled

def binarize(image): 
    thresh, binarized = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binarized

def denoise(image):
    kernel = numpy.ones((1,1), numpy.uint8)
    image = cv2.dilate(image, kernel, iterations=2)
    kernel = numpy.ones((1,1), numpy.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image
def dilation_and_erosion(image):
    image = cv2.bitwise_not(image)
    kernel = numpy.ones((2,2), numpy.uint8)
    image = cv2.erode(image, kernel , iterations=0)
    image = cv2.bitwise_not(image)
    return image


#------------------------MAIN---------------------------------

filename = input("enter the name of the file: ")
child_dir = "data"
full_path = os.path.join(os.getcwd(),child_dir, filename)
img = full_path
image = cv2.imread(img)

inverted_image = invert(image)
cv2.imwrite("temp/inverted.jpg", inverted_image)
grayscaled_image = grayscale(image)
cv2.imwrite("temp/grayscaled.jpg", grayscaled_image)
bw_image = binarize(grayscaled_image)
cv2.imwrite("temp/bw.jpg", bw_image)
denoised_image = denoise(bw_image)
cv2.imwrite("temp/denoised.jpg", denoised_image)
dilation_and_erosion_image = dilation_and_erosion(image)
cv2.imwrite("temp/dilation_and_erosion.jpg", dilation_and_erosion_image)


#FOR_IMAGES_THAT_NEED_ROTATION---------------------------------

image_memory = cv2.imread(img)
rotated_image = deskew(image_memory)
cv2.imwrite("temp/rotated.jpg", rotated_image)

inverted_image_rotated = invert(rotated_image)
cv2.imwrite("temp/inverted_rotated.jpg", inverted_image_rotated)
grayscaled_image_rotated = grayscale(rotated_image)
cv2.imwrite("temp/grayscaled_rotated.jpg", grayscaled_image_rotated)
bw_image_rotated = binarize(grayscaled_image_rotated)
cv2.imwrite("temp/bw_rotated.jpg", bw_image_rotated)
denoised_image_rotated = denoise(bw_image_rotated)
cv2.imwrite("temp/denoised_rotated.jpg", denoised_image_rotated)
dilation_and_erosion_image_rotated = dilation_and_erosion(rotated_image)
cv2.imwrite("temp/dilation_and_erosion_rotated.jpg", dilation_and_erosion_image_rotated)


#RESULTS_FOR_IMAGES_THAT_DONT_NEED_ROTATION
print("--------------------------------INVERTED--------------------------------")
img_memory_inverted = Image.open("temp/inverted.jpg")
ocr_result_inverted = pytesseract.image_to_string(img_memory_inverted)
print(ocr_result_inverted)
print("--------------------------------GRAYSCALED--------------------------------")
img_memory_grayscaled = Image.open("temp/grayscaled.jpg")
ocr_result_grayscaled = pytesseract.image_to_string(img_memory_grayscaled)
print(ocr_result_grayscaled)
print("--------------------------------BINARIZED--------------------------------")
img_memory_binarized = Image.open("temp/bw.jpg")
ocr_result_binarized = pytesseract.image_to_string(img_memory_binarized)
print(ocr_result_binarized)
print("--------------------------------DENOISED--------------------------------")
img_memory_denoised = Image.open("temp/denoised.jpg")
ocr_result_denoised = pytesseract.image_to_string(img_memory_denoised)
print(ocr_result_denoised)
print("--------------------------------DILATION_AND_EROSION--------------------------------")
img_memory_dilation_and_erosion = Image.open("temp/dilation_and_erosion.jpg")
ocr_result_dilation_and_erosion = pytesseract.image_to_string(img_memory_dilation_and_erosion)
print(ocr_result_dilation_and_erosion)


#RESULTS_FOR_IMAGES_THAT_NEED_ROTATION
print("--------------------------------INVERTED_ROTATED--------------------------------")
img_memory_inverted_rotated = Image.open("temp/inverted_rotated.jpg")
ocr_result_inverted_rotated = pytesseract.image_to_string(img_memory_inverted_rotated)
print(ocr_result_inverted_rotated)
print("--------------------------------GRAYSCALED_ROTATED--------------------------------")
img_memory_grayscaled_rotated = Image.open("temp/grayscaled_rotated.jpg")
ocr_result_grayscaled_rotated = pytesseract.image_to_string(img_memory_grayscaled_rotated)
print(ocr_result_grayscaled_rotated)
print("--------------------------------BINARIZED_ROTATED--------------------------------")
img_memory_binarized_rotated = Image.open("temp/bw_rotated.jpg")
ocr_result_binarized_rotated = pytesseract.image_to_string(img_memory_binarized_rotated)
print(ocr_result_binarized_rotated)
print("--------------------------------DENOISED_ROTATED--------------------------------")
img_memory_denoised_rotated = Image.open("temp/denoised_rotated.jpg")
ocr_result_denoised_rotated = pytesseract.image_to_string(img_memory_denoised_rotated)
print(ocr_result_denoised_rotated)
print("--------------------------------DILATION_AND_EROSION_ROTATED--------------------------------")
img_memory_dilation_and_erosion_rotated = Image.open("temp/dilation_and_erosion_rotated.jpg")
ocr_result_dilation_and_erosion_rotated = pytesseract.image_to_string(img_memory_dilation_and_erosion_rotated)
print(ocr_result_dilation_and_erosion_rotated)

