import sys
import numpy as np
import random
import matplotlib.pyplot as plt
## change Path ##
sys.path.append("/home/elkin/university/gradSchool/Fall2020/CS472/CS472")
from tools import arff, normalization
import itertools
from clustering import HAC, Kmeans
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.cluster import AgglomerativeClustering
import numpy as np

print("*******************PART 1************************************")
print("*******************DEBUG************************************")
arff_path_train = r"training/abalone.arff"
# arff_path_train = r"training/labor-negotiations.arff"
mat = arff.Arff(arff_path_train,label_count=0)
raw_data = mat.data
data = raw_data
# data = data[...,:-1]
# print(data)
##Normalize the data##
scaler = preprocessing.MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

# clustering = AgglomerativeClustering(n_clusters = 5, linkage= 'single').fit(norm_data)
# norm_data = np.array([[0.8,0.7],[-0.1,0.2],[0.9,0.8],[0,0.2],[0.2,0.1]])
### KMEANS ###
# KMEANS = Kmeans.KMEANSClustering(k=5,debug=True)
# KMEANS.fit(norm_data)
# KMEANS.save_clusters("debug_kmeans.txt")

### HAC SINGLE LINK ###
HAC_single = HAC.HACClustering(k=5,link_type='single')
HAC_single.fit(norm_data)
HAC_single.save_clusters("debug_hac_single.txt")
print(clustering.labels_)
