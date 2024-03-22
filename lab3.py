import numpy as np
import cv2
import os
from skimage import io


directory = 'images/lab3'

images = []
masks = []
sum_img_pixels = []

for filename in os.listdir(directory):
    if "img" in filename:
        image = cv2.imread(os.path.join(directory, filename))
        images.append(image)
        sum_img_pixels = np.sum(image)
    elif "mask" in filename:
        mask = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
#       la fel si pt mask
# print("numarul de imagini:", len(images))
# print("numarul de masti:", len(masks))

# print(np.max(images[1]))
# np.argmax(sum_pixels)
# np.argmin(sum_pixels)

# for i, img in enumerate(images):
#     io.imshow(img.astype(np.uint8))
#     io.show()
mean_image = np.mean(images, axis=0)
sigma = np.std(images, axis=0)
print(sigma)



normalized_images = (images - mean_image) / sigma
# for i, img  in enumerate(normalized_images):
#     io.imshow(img.astype(np.uint8))
#     io.show()

# print(sigma)


for i in range(len(images)):
    image = images[i]
    mask = masks[i]

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    mask_indices = mask > 0

    image[mask_indices] = [255, 0, 0]
    io.imshow(image)
    io.show()

