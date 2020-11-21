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

# norm_data = np.array([[185,72],[170,56],[168,60],[179,68],[182,72],[188,77]])
### KMEANS ###
KMEANS = Kmeans.KMEANSClustering(k=5,debug=True)
KMEANS.fit(norm_data)
KMEANS.save_clusters("debug_kmeans.txt")

### HAC SINGLE LINK ###
# HAC_single = HAC.HACClustering(k=5,link_type='single')
# HAC_single.fit(norm_data)
# HAC_single.save_clusters("debug_hac_single.txt")

### HAC COMPLETE LINK ###
# HAC_complete = HAC.HACClustering(k=5,link_type='complete')
# HAC_complete.fit(norm_data)
# HAC_complete.save_clusters("debug_hac_complete.txt")
# clustering = AgglomerativeClustering(n_clusters = 5, linkage= 'single').fit(norm_data)
