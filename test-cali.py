import numpy as np
import cv2

# Load the calibration results
calibration_data = np.load('camera_calibration.npz')
camera_matrix = calibration_data['camera_matrix']
distortion_coeffs = calibration_data['distortion_coeffs']

# Load an image to undistort
img = cv2.imread('air/c11.jpg')
h, w = img.shape[:2]

# Get new camera matrix
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
print("new_camera_matrix", new_camera_matrix.shape)
print("roi", roi)

# Undistort the image
undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs, None, new_camera_matrix)

# Crop the image based on the ROI
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

# Display the results
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
