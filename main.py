import numpy as np
import cv2
import os
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.set_printoptions(threshold=1000)
images = []
masks = []
sum_img_pixels = []

directory = 'images/lab3'

for filename in os.listdir(directory):
    if 'mask' in filename:
        img = cv2.imread((os.path.join(directory, filename)), cv2.IMREAD_GRAYSCALE)
        masks.append(img)
    if 'img' in filename:
        img = cv2.imread((os.path.join(directory, filename)))
        images.append(img)
    sum_img_pixels.append(np.sum(img))


np_images = np.array(images)
np_masks = np.array(masks)

mean_image = np.mean(np_images, axis=0)
sigma = np.uint8(np.std(np_images, axis=0))

num_clusters = 5
for img in np_images:
    img_norm = np.uint8((img-mean_image)/sigma)

for img, mask in zip(images, masks):
    # Aplicăm K-Means pentru fiecare imagine
    pixels = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)

    # Obținem noile culori ca medie a clusterelor
    centers = np.uint8(kmeans.cluster_centers_)
    new_pixels = centers[kmeans.labels_]

    # Reconstruim imaginea pe baza noilor culori
    cluster_image = new_pixels.reshape(img.shape)

    # Masca (dacă este necesar să o folosim pentru a modifica anumite părți ale imaginii)
    if mask is not None:
        mask_indices = mask > 0
        cluster_image[mask_indices] = [255, 0, 0]  # Aplicăm masca pentru a modifica culorile respective în roșu

    # Afișăm imaginea rezultată după aplicarea K-Means și eventuala aplicare a măștii
    io.imshow(cluster_image)
    io.show()


# normalized_images = np.uint8((img-mean_image)/sigma)
# plt.imshow(cv2.cvtColor(normalized_images[0].astype('uint8'), cv2.COLOR_BGR2RGB))
# plt.show()
