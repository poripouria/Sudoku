import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('sample1.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to binarize the image(returns threshhold value and threshold img)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #(src, thresh_value, max_val, threshhold_type)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
'''
thresh: is the binary image that will be used to find the contours.
RETR_EXTERNAL: specifies that only the outermost contours should be returned.
CHAIN_APPROX_NONE: specifies that the contours should not be approximated.

contours: is a list of all the contours found in the image
hierarchy: is a list that describes the hierarchical relationships between the contours.
'''
# cv2.drawContours(img, contours, -1, (0, 255, 0), 2) #src, contours, -1:draw all contours, color, thickness
# plt.imshow(img2)

# Find the largest contour (the Sudoku board)
largest_contour = max(contours, key=cv2.contourArea)

# Perimeter of the largest contour found in the image
peri = cv2.arcLength(largest_contour, True)
'''
largest_contour: is the contour whose perimeter will be calculated.
closed: specifies whether the contour is closed (True) or open (False).
'''

# Get the corners of the Sudoku board
approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
'''
contour: is the contour that will be approximated.
epsilon: is the maximum distance between a point on the approximated contour and the original contour.
closed: specifies whether the approximated contour should be closed (True) or open (False).
'''
# print(approx)
# cv2.drawContours(img, approx, -1, (0, 255, 0), 10) #src, contours, -1:draw all contours, color, thickness
# plt.imshow(img2)

# Reorder the corners in a clockwise direction starting from the top-left corner
ordered_corners = np.zeros_like(approx)
sum_coords = approx.sum(axis=2)
ordered_corners[0] = approx[np.argmin(sum_coords)]
ordered_corners[2] = approx[np.argmax(sum_coords)]
diff_coords = np.diff(approx, axis=2)
ordered_corners[1] = approx[np.argmin(diff_coords)]
ordered_corners[3] = approx[np.argmax(diff_coords)]

# Reshape the ordered corners to (4, 2) shape
ordered_corners = np.float32(ordered_corners.reshape((4, 2)))

# Define the size of the output image
output_size = (450, 450)

# Define the destination points for perspective transformation
dest_points = np.float32([[0, 0], [output_size[0], 0], [output_size[0], output_size[1]], [0, output_size[1]]])
Matrix = cv2.getPerspectiveTransform(ordered_corners, dest_points)

# Apply the perspective transformation to the original image
output = cv2.warpPerspective(img, Matrix, output_size)

# Display the original image and the aligned image
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Aligned Image')
plt.axis('off')

plt.tight_layout()
plt.show()
