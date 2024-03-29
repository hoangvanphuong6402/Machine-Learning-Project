import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.birch import birch
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample

sns.set(color_codes=True)

num_cluster = 5

df = pd.read_csv("StudentsPerformance.csv", index_col=0)


def plot_data(data_set, labels, number_of_cluster, centers, i=0, j=2):
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf", "#e377c2", "#bcbd22", "#7f7f7f"]
    x_points = [[] for x in range(number_of_cluster)]
    y_points = [[] for x in range(number_of_cluster)]
    x_centers = []
    y_centers = []

    for x in range(number_of_cluster):
        x_centers.append(centers[x][i])
        y_centers.append(centers[x][j])

    for index in range(len(data_set)):
        x_points[labels[index]].append(data_set[index][i])
        y_points[labels[index]].append(data_set[index][j])

    for x in range(number_of_cluster):
        plt.plot(x_points[x], y_points[x], '.', color=color_list[x])

    plt.plot(x_centers, y_centers, '.', color="black")
    plt.show()


def R_squared(data_set, labels, number_of_cluster):
    totalSum = 0
    clusters = [[] for i in range(number_of_cluster)]
    for index in range(len(data_set)):
        clusters[labels[index]].append(data_set[index])

    for i in range(number_of_cluster):
        avg = np.mean(clusters[i], 0)
        sum = 0
        for x in range(len(clusters[i])):
            sum = sum + np.square(np.linalg.norm(np.array(clusters[i][x]) - np.array(avg)))

        totalSum = totalSum + sum

    avg = np.mean(data_set, 0)

    sum = 0
    for i in range(len(data_set)):
        sum = sum + np.square(np.linalg.norm(np.array(data_set[i]) - np.array(avg)))

    return (sum - totalSum) / sum


def RMSSTD(data_set, labels, number_of_cluster):
    totalSum = 0
    totalCount = 0
    clusters = [[] for i in range(number_of_cluster)]
    for index in range(len(data_set)):
        clusters[labels[index]].append(data_set[index])

    for i in range(number_of_cluster):
        avg = np.mean(clusters[i], 0)
        sum = 0
        for x in range(len(clusters[i])):
            sum = sum + np.square(np.linalg.norm(np.array(clusters[i][x]) - np.array(avg)))

        totalSum = totalSum + sum

    totalCount = len(data_set) - number_of_cluster
    P = len(data_set[0])
    return np.sqrt(totalSum / (P * totalCount))


DIAMETER_METHODS = ['mean_cluster', 'farthest']
CLUSTER_DISTANCE_METHODS = ['nearest', 'farthest']


def inter_cluster_distances(labels, distances, method='nearest'):
    if method not in CLUSTER_DISTANCE_METHODS:
        raise ValueError(
            'method must be one of {}'.format(CLUSTER_DISTANCE_METHODS))

    if method == 'nearest':
        return __cluster_distances_by_points(labels, distances)
    elif method == 'farthest':
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                    (not farthest and
                     distances[i, ii] < cluster_distances[labels[i], labels[ii]])
                    or
                    (farthest and
                     distances[i, ii] > cluster_distances[labels[i], labels[ii]])):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = distances[i, ii]
    return cluster_distances


def diameter(labels, distances, method='farthest'):
    if method not in DIAMETER_METHODS:
        raise ValueError('method must be one of {}'.format(DIAMETER_METHODS))

    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == 'mean_cluster':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i, ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == 'farthest':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
    return diameters


def dunn(labels, your_dataset, diameter_method='farthest', cdist_method='nearest'):
    distances = pairwise_distances(your_dataset)
    labels = LabelEncoder().fit(labels).transform(labels)

    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(diameter(labels, distances, diameter_method))

    return min_distance / max_diameter


def normalize_data(data_set):
    for j in range(0, len(data_set[0])):
        col_max = np.max(np.array(data_set)[:, j])
        for i in range(0, len(data_set)):
            data_set[i][j] = data_set[i][j] / col_max

    return data_set


def outlier_removal(data_set):
    distances = pairwise_distances(data_set)
    max = 0
    min = sys.maxsize
    n = len(data_set)
    for i in range(n):
        for j in range(i + 1, n):
            if min > distances[i][j]:
                min = distances[i][j]
            if max < distances[i][j]:
                max = distances[i][j]
    threshold = (max + min) / 2

    sum = [0] * len(data_set[0])
    for i in range(len(data_set)):
        sum = np.add(sum, data_set[i])
    center = np.array(sum) / len(data_set)

    result = []
    for i in range(len(data_set)):
        if np.linalg.norm(np.array(center) - np.array(data_set[i])) < threshold:
            result.append(data_set[i])

    return result


df_ = df.to_numpy()
df_ = np.delete(df_, 3, 1)
df_ = np.delete(df_, 2, 1)
df_ = np.delete(df_, 1, 1)
df_ = np.delete(df_, 0, 1)

# default
kmeans = KMeans(n_clusters=num_cluster)
kmeans.fit(df_)
kmeans_labels = kmeans.labels_
# print(kmeans_labels)
# print(kmeans.cluster_centers_)
# print(davies_bouldin_score(df_, kmeans_labels))  # small
# print(RMSSTD(df_, kmeans_labels, num_cluster))
# print(dunn(kmeans_labels, df_))  # great
# print(R_squared(df_, kmeans_labels, num_cluster))
# print(silhouette_score(df_, kmeans_labels))
# plot_data(df_, kmeans_labels, num_cluster, kmeans.cluster_centers_, 0, 2)

# after normalizing
df_normalized = normalize_data(df_)
kmeans = KMeans(n_clusters=num_cluster)
kmeans.fit(df_normalized)
kmeans_labels = kmeans.labels_
# print(kmeans_labels)
# print(kmeans.cluster_centers_)
# print(davies_bouldin_score(df_normalized, kmeans_labels))    # small
# print(RMSSTD(df_normalized, kmeans_labels, num_cluster))
# print(dunn(kmeans_labels, df_normalized))                    # great
# print(R_squared(df_normalized, kmeans_labels, num_cluster))
# print(silhouette_score(df_normalized, kmeans_labels))
# plot_data(df_normalized, kmeans_labels, num_cluster, kmeans.cluster_centers_, 0, 2)

# after normalizing and removing outliers
df_removed = outlier_removal(df_normalized)
df_removed = np.array(df_removed)
kmeans = KMeans(n_clusters=num_cluster)
kmeans.fit(df_removed)
kmeans_labels = kmeans.labels_
print(kmeans_labels)
print(kmeans.cluster_centers_)
# print(davies_bouldin_score(df_, kmeans_labels))  # small
# print(RMSSTD(df_, kmeans_labels, num_cluster))
# print(dunn(kmeans_labels, df_))  # great
# print(R_squared(df_, kmeans_labels, num_cluster))
# print(silhouette_score(df_, kmeans_labels))
# plot_data(df_, kmeans_labels, num_cluster, kmeans.cluster_centers_, 0, 2)

num_of_iteration = 100
dbs = rmsstd = dun = rs = sil = 0
for i in range(num_of_iteration):
    kmeans.fit(df)
    dbs += davies_bouldin_score(df, kmeans.labels)
    rmsstd += RMSSTD(df, kmeans.labels_, num_cluster)
    dun += dunn(kmeans.labels, df_)
    rs += R_squared(df, kmeans.labels_, num_cluster)
    sil += silhouette_score(df, kmeans.labels_)

print(dbs / num_of_iteration)  # small
print(rmsstd / num_of_iteration)
print(dun / num_of_iteration)  # great
print(rs / num_of_iteration)
print(sil / num_of_iteration)
