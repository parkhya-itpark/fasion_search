
# import cv2
# import numpy as np

# temp_image = cv2.imread('some_images/1164.jpg')

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# sharpened = cv2.filter2D(image, -1, kernel)

# cv2.imshow('output', temp_image)
# cv2.imshow('output', cv2.resize(temp_image, (512, 512)))

# cv2.imshow(‘Original Image’, image)
# cv2.imshow(‘Sharpened Image’, sharpened)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import os

def deblur_image(image_path, output_path):
    # Read the blurred image
    blurred_image = cv2.imread(image_path)

    # Apply a deblurring algorithm (e.g., Wiener deconvolution)
    deblurred_image = cv2.deconvolve(blurred_image, cv2.randn(blurred_image.shape, 0, 5))

    # Save the deblurred image
    cv2.imwrite(output_path, deblurred_image)

output_image_path = 'quality_images_1'
# Example usage
for image_path in os.listdir('some_images'):
    deblur_image(image_path, output_image_path)
    