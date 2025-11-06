import cv2
import numpy as np 
import matplotlib.pyplot as plt

# 1. Read and display original image
img_path = 'images/table_image.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()


# 2. Define source and destination points (manually chosen or from features)
# These are example corner coordinates of a document or rectangular object in the image
src_points = np.float32([
    [41, 228], 
    [352, 120], # top-right corner
    [444, 370], # bottom-right corner
    [767, 237]   # bottom-left corner
])

# Destination coordinates (desired straightened rectangle)
dst_points = np.float32([
    [0, 0],
    [400, 0],
    [0, 400],
    [400, 400],
])


# 3. Compute the Homography Matrix
H, status = cv2.findHomography(src_points, dst_points)
print("Homography Matrix (H):\n", H)


# 4. Validate transformation for one sample point
src_pt = np.append(src_points[0], 1)
mapped_pt = H @ src_pt
mapped_pt /= mapped_pt[2]
print("Reprojected point for first corner:", mapped_pt[:2])


# 5. Apply perspective warp using the computed matrix
warped_img = cv2.warpPerspective(img_rgb, H, (400, 400))


# 5. Display the transformed image
plt.imshow(warped_img)
plt.title("Perspective Corrected (Warped) Image")
plt.axis('off')
plt.show()
