import numpy as np
import cv2
import glob

# Define the dimensions of the checkerboard
CHECKERBOARD = (9, 6)  # Change to your checkerboard dimensions
square_size = 2.5  # Size of a square in your defined unit (e.g., cm)

# Prepare object points based on the checkerboard dimensions
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Load images
images = glob.glob('*.jpg')  # Change path and file type as needed

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, flags=flags)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Optionally draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
	    print("not found for ", fname)

cv2.destroyAllWindows()

# Calibrate the camera
#first_img = cv2.imread(images[0])
#gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the calibration results
np.savez('camera_calibration.npz', camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs)

# Print the camera matrix and distortion coefficients
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", distortion_coeffs)
