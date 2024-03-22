import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import faraway.datasets.composite

###################################################################
# Exercitiul 1
data=pd.read_csv('./csv/heart.csv')
data_train = data.drop(['output', 'trtbps'], axis=1)
# print(data.head())

# (data_train['sex'].value_counts().sort_values(ascending=False)
#     .plot(kind='pie'))
# plt.show()


mean = np.mean(data_train, axis=0)
sigma = np.std(data_train, axis=0)

normalized_data = (data_train - mean) / sigma
# print(mean)

# print(normalized_data.head())
# (normalized_data['age'].value_counts().sort_values(axis=0, ascending=False)
#     .head(20).plot(kind="bar"))
# plt.show()

# (data['age'].value_counts().sort_values(axis=0, ascending=False)
#     .head(20).plot(kind="bar"))
# plt.show()

# sns.countplot(x='cp', data=data)
# plt.title('Numărul pacienților pentru fiecare tip de durere în piept')
# plt.xlabel('Tipul durerii în piept')
# plt.ylabel('Număr de pacienți')
# plt.show()


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_train)


norm_kmeans =  KMeans(n_clusters=2, random_state=32)
norm_kmeans.fit(normalized_data)

# print(kmeans.cluster_centers_)
# print(kmeans.labels_)
# print(data['output'])


# cof_matrix = confusion_matrix(data['output'], kmeans.labels_)
# print("Matricea de confuzie:")
# print(conf_matrix)
#
# norm_conf_matrix = confusion_matrix(data['output'], norm_kmeans.labels_)
# print("Matricea de confuzie normalizata:")
# print(norm_conf_matrix)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(data_train)
    inertia.append(kmeans.inertia_)

norm_inertia = []
for k in range(1, 11):
    kmeans_norm = KMeans(n_clusters=k, random_state=32).fit(normalized_data)
    norm_inertia.append(kmeans_norm.inertia_)
    print(kmeans_norm.inertia_)

plt.plot(inertia)
plt.ylabel("Inertia")
plt.show()

plt.plot(norm_inertia)
plt.ylabel("Inertia")
plt.show()


# print(inertia, norm_inertia)
#########################################################################
# Exercitiul 2

composite = faraway.datasets.composite.load()