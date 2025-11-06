import cv2
import matplotlib.pyplot as plt

img = cv2.imread('images/table_image.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(img, None)

sift_img = cv2.drawKeypoints(
    img, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    color=(0, 255, 0)
)

plt.figure(figsize=(8, 6))
plt.imshow(sift_img, cmap='gray')
plt.title("SIFT Feature Detection and Description (Synthetic Image)")
plt.axis('off')
plt.show()

print("Total Keypoints Detected: ", len(keypoints))
print("Descriptor Shape:", descriptors.shape if descriptors is not None else "None")
if descriptors is not None:
    print("\nExample of SIFT Descriptor (first 10 values of first descriptor):")
    print(descriptors[0][:10])
