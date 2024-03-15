import numpy as np
from skimage import io
import matplotlib.pyplot as plt

images = np.empty((9, 400, 600))

for idx in range(0, 9):
    file_path = f"lab1/images/car_{idx}.npy"
    image = np.load(file_path)
    images[idx-1] = image


# or just np.sum
pixels_sum = np.sum(images, axis=(0,1,2))
# print(pixels_sum)

pixels_sum_i = np.sum(images, axis=(1, 2))
pixels_sum_r = np.sum(images, axis=(0, 2))
pixels_sum_c = np.sum(images, axis=(0, 1))

print(pixels_sum_i)
# print(pixels_sum_r)
# print(pixels_sum_c)


max_image_pixels = np.argmax(pixels_sum_i)
print(max_image_pixels)


mean_image = np.mean(images, axis=0)
# io.imshow(mean_image.astype(np.uint8)) # petru a putea fi afisata
                                        # imaginea trebuie sa aiba
                                        # tipul unsigned int
# io.show()


sigma = np.std(images, axis=0)


# normalized_images = (images - mean_image) / sigma
# print(normalized_images.shape)
# for i, img  in enumerate(normalized_images):
#     io.imshow(img.astype(np.uint8))
#     io.show()

# print(sigma)

#
cropped_images = images[:, 200:300, 280:400]
for i, crp in enumerate(cropped_images):
    io.imshow(crp.astype(np.uint8))
    io.show()