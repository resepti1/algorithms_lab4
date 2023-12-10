import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import os
from sklearn.cluster import KMeans
import numpy as np


def find_cluster_center(cluster_points):
    if not cluster_points:
        return None

    # перетворення списку в масив numpy
    points_array = np.array(cluster_points)

    # рахуємо центр мас кожного кластеру
    center = np.mean(points_array, axis=0)

    return center.tolist()


# встановлюємо розмір вікна
width, height = 960, 540
fig, ax = plt.subplots(figsize=(width/80, height/80))

x = []
y = []

# отримання директорії скріпта
script_directory = os.path.dirname(os.path.abspath(__file__))

# отримання шляху зберігання результату
output_image_path = os.path.join(script_directory, "output.jpg")

# зчитування координат з файлу
with open('DS7.txt', 'r') as file:
    for line in file:
        coordinates = line.split()
        y.append(int(coordinates[0]))
        x.append(int(coordinates[1]))
points = list(zip(x, y))

# знаходимо сектори за допомогою алгоритму кластеризації
KMean = KMeans(8)
labels = KMean.fit_predict(points)
unique_labels = np.unique(labels)

cluster_centers = []

# цикл знаходження центрів мас, записуємо їх в список cluster_centers
for cluster_label in unique_labels:
    cluster_points = [point for point, label in zip(
        points, labels) if label == cluster_label]
    cluster_center = find_cluster_center(cluster_points)
    cluster_centers.append(cluster_center)

# отримуємо діаграму Вороного
vor = Voronoi(cluster_centers)

# виводимо діагараму на графік matplotlib
voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=True)

plt.xlim(0, width)
plt.ylim(0, height)

# виводимо точки датасету з насиченістю 10%
ax.scatter(x, y, color='black', s=1, alpha=0.1)
# налаштування відображення графіку
plt.title('Діаграма Вороного')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# збереження графіка в форматі jpg
plt.savefig(output_image_path, format='jpg', dpi=80, bbox_inches='tight')
plt.show()
