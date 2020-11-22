import sys
import numpy as np
import random
from time import time
import matplotlib.pyplot as plt
## change Path ##
sys.path.append("/home/elkin/university/gradSchool/Fall2020/CS472/CS472")
from tools import arff, normalization
import itertools
from clustering import HAC, Kmeans
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from sklearn import metrics
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs

print("*******************PART 1************************************")
print("*******************EVALUATION************************************")
# arff_path_train = r"training/seismic-bumps_train.arff"
# # arff_path_train = r"training/labor-negotiations.arff"
# mat = arff.Arff(arff_path_train,label_count=0)
# raw_data = mat.data
# data = raw_data
# # data = data[...,:-1]
# ##Normalize the data##
# scaler = preprocessing.MinMaxScaler()
# scaler.fit(data)
# norm_data = scaler.transform(data)
# ### KMEANS ###
# KMEANS = Kmeans.KMEANSClustering(k=5,debug=True)
# KMEANS.fit(norm_data)
# KMEANS.save_clusters("debug_kmeans.txt")
#
# ## HAC SINGLE LINK ###
# HAC_single = HAC.HACClustering(k=5,link_type='single')
# HAC_single.fit(norm_data)
# HAC_single.save_clusters("debug_hac_single.txt")
#
# ## HAC COMPLETE LINK ###
# HAC_complete = HAC.HACClustering(k=5,link_type='complete')
# HAC_complete.fit(norm_data)
# HAC_complete.save_clusters("debug_hac_complete.txt")
# clustering = AgglomerativeClustering(n_clusters = 5, linkage= 'single').fit(norm_data)
print("*******************PART 2************************************")

def graphsSSE(type,norm_data,file_name):
    save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/clustering/plots/"
    k = []
    sse = []
    for i in range(2,8):
        if type == 'k_means':
            KMEANS = Kmeans.KMEANSClustering(k=i,debug=False)
            KMEANS.fit(norm_data)
            k.append(i)
            sse.append(KMEANS.totalSSE)
        elif type == 'single':
            HAC_single = HAC.HACClustering(k=i,link_type='single')
            HAC_single.fit(norm_data)
            k.append(i)
            sse.append(HAC_single.totalSSE)
        elif type == 'complete':
            HAC_complete = HAC.HACClustering(k=i,link_type='complete')
            HAC_complete.fit(norm_data)
            k.append(i)
            sse.append(HAC_complete.totalSSE)
    x = np.arange(len(k))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    sses = ax.bar(x - width/2, sse, width, label='SSE')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('SSE')
    ax.set_title('Total SSE for different k-values')
    ax.set_xticks(x)
    ax.set_xticklabels(k)
    ax.set_xlabel('k')
    ax.legend()
    save_path = save_path + file_name
    fig.savefig(save_path)

# arff_path_train = r"training/iris.arff"
# mat = arff.Arff(arff_path_train,label_count=0)
# raw_data = mat.data
# data = raw_data
# data = data[...,:-1]
# scaler = preprocessing.MinMaxScaler()
# scaler.fit(data)
# norm_data = scaler.transform(data)
# ### KMEANS ###
# graphsSSE('k_means',norm_data,'kmeansSSE')
# ## HAC SINGLE LINK ###
# graphsSSE('single',norm_data,'HACsinlgesSSE')
# ## HAC COMPLETE LINK ###
# graphsSSE('complete',norm_data,'HACcompletesSSE')
#
# # including last output column
# arff_path_train = r"training/iris.arff"
# mat = arff.Arff(arff_path_train,label_count=0)
# raw_data = mat.data
# data = raw_data
# data = data[...,:-1]
# scaler = preprocessing.MinMaxScaler()
# scaler.fit(data)
# norm_data = scaler.transform(data)
# ### KMEANS ###
# graphsSSE('k_means',norm_data,'kmeansSSE2')
# ## HAC SINGLE LINK ###
# graphsSSE('single',norm_data,'HACsinlgesSSE2')
# ## HAC COMPLETE LINK ###
# graphsSSE('complete',norm_data,'HACcompletesSSE2')

## final running k-5 4 different times
# arff_path_train = r"training/iris.arff"
# mat = arff.Arff(arff_path_train,label_count=0)
# raw_data = mat.data
# data = raw_data
# data = data[...,:-1]
# scaler = preprocessing.MinMaxScaler()
# scaler.fit(data)
# norm_data = scaler.transform(data)
# # save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/clustering/plots/k4runs"
# k = []
# sse = []
# for i in range(0,5):
#     KMEANS = Kmeans.KMEANSClustering(k=4,debug=False)
#     KMEANS.fit(norm_data)
#     k.append(i)
#     sse.append(KMEANS.totalSSE)
# x = np.arange(len(k))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# sses = ax.bar(x - width/2, sse, width, label='SSE')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('SSE')
# ax.set_title('Total SSE for different k = 4 runs')
# ax.set_xticks(x)
# ax.set_xticklabels(k)
# ax.set_xlabel('Model Run')
# ax.legend()
# fig.savefig(save_path)
print("*******************PART 3************************************")
## kMEANS
# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/clustering/plots/"
# range_n_clusters = [2, 3, 4, 5,7]
# arff_path_train = r"training/iris.arff"
# mat = arff.Arff(arff_path_train,label_count=0)
# raw_data = mat.data
# data = raw_data
# data = data[...,:-1]
# scaler = preprocessing.MinMaxScaler()
# scaler.fit(data)
# norm_data = scaler.transform(data)
# X = norm_data
# for n_clusters in range_n_clusters:
#     save_path = save_path + f'kmeans_{n_clusters}'
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)
#
#     # The 1st subplot is the silhouette plot
#     # The silhouette coefficient can range from -1, 1 but in this example all
#     # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
#
#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(X)
#
#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#
#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)
#
#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]
#
#         ith_cluster_silhouette_values.sort()
#
#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i
#
#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)
#
#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples
#
#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")
#
#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#     # 2nd Plot showing the actual clusters formed
#     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors, edgecolor='k')
#
#     # Labeling the clusters
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                 c="white", alpha=1, s=200, edgecolor='k')
#
#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                     s=50, edgecolor='k')
#
#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")
#
#     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')
#     plt.savefig(save_path)

## HAC Silhouette

# save_path="/home/elkin/university/gradSchool/Fall2020/CS472/CS472/clustering/plots/"
# range_n_clusters = [2, 3, 4, 5,7]
# arff_path_train = r"training/iris.arff"
# mat = arff.Arff(arff_path_train,label_count=0)
# raw_data = mat.data
# data = raw_data
# data = data[...,:-1]
# scaler = preprocessing.MinMaxScaler()
# scaler.fit(data)
# norm_data = scaler.transform(data)
# X = norm_data
# for n_clusters in range_n_clusters:
#     save_path = save_path + f'HAC_{n_clusters}'
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)
#
#     # The 1st subplot is the silhouette plot
#     # The silhouette coefficient can range from -1, 1 but in this example all
#     # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
#
#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = AgglomerativeClustering(n_clusters = n_clusters).fit(X)
#     cluster_labels = clusterer.fit_predict(X)
#
#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#
#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)
#
#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]
#
#         ith_cluster_silhouette_values.sort()
#
#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i
#
#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)
#
#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples
#
#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")
#
#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#     # 2nd Plot showing the actual clusters formed
#     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors, edgecolor='k')
#
#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")
#
#     plt.suptitle(("Silhouette analysis for HAC clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')
#     plt.savefig(save_path)

## METRICS TABLE ##
arff_path_train = r"training/iris.arff"
mat = arff.Arff(arff_path_train,label_count=0)
raw_data = mat.data
data = raw_data
labels = data[:, -1]
data = data[...,:-1]
scaler = preprocessing.MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

n_digits = len(np.unique(labels))
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean'
                                      )))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits),
              name="k-means++", data=norm_data)

## DIFFERENT DATASET
# create dataset
X, y = make_blobs(
   n_samples=150, n_features=2,
   centers=3, cluster_std=0.5,
   shuffle=True, random_state=0
)

km = KMeans(
    n_clusters=3, init='random',
    max_iter=300,
    tol=1e-04
)
y_km = km.fit_predict(X)
print(y_km)
HAC = AgglomerativeClustering(n_clusters=3).fit_predict(X)
print(HAC)
