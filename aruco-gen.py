import cv2
import numpy as np

# Specify the dictionary and marker ID
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_id = 5  # Change this ID for different markers
marker_size = 200  # Size of the marker in pixels

# Generate the marker
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)


# Save the marker image
cv2.imwrite("aruco_marker_5.png", marker_image)

# Display the marker
cv2.imshow("Marker", marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()