import cv2
import numpy as np 
import matplotlib.pyplot as plt

# 1. Read and display original image
image_path = 'input_image.jpg'

image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

h, w = image_bgr.shape[:2]


# 2. Translation (shift image)
tx, ty = 200, 300 # shift by 200px right and 300px down
T = np.float32([[1, 0, tx],
                [0, 1, ty]])
translated_image = cv2.warpAffine(image_rgb, T, (w, h))

plt.imshow(translated_image)
plt.title("Translated Image (200px right, 300px down)")
plt.axis('off')
plt.show()


# 3. Rotation
M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
rotated_image = cv2.warpAffine(image_rgb, M, (w, h))

plt.imshow(rotated_image)
plt.title("Rotated Image (45°)")
plt.axis('off')
plt.show()


# 4. Scaling
scaled_image = cv2.resize(image_rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

plt.imshow(scaled_image)
plt.title("Scaled Image (1.5x)")
plt.axis('off')
plt.show()


# 5. Reflection
reflected_image = cv2.flip(image_rgb, 1)
# 0 for flipping around the x-axis (vertical flip)
# 1 for flipping around the y-axis (horizontal flip)
# -1 for flipping around both axes

plt.imshow(reflected_image)
plt.title("Reflected Image (Horizontal Flip)")
plt.axis('off')
plt.show()


# 6. Shearing
shear_factor_x, shear_factor_y = 0.3, 0.1
shear_matrix = np.float32([[1, shear_factor_x, 0],
                           [shear_factor_y, 1, 0]])
sheared_image = cv2.warpAffine(image_rgb, shear_matrix, (int(w*1.5), int(h*1.5)))

plt.imshow(sheared_image)
plt.title("Sheared Image")
plt.axis('off')
plt.show()


