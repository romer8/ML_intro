import sys
import numpy as np
import random
import matplotlib.pyplot as plt
## change Path ##
sys.path.append("/home/elkin/university/gradSchool/Fall2020/CS472/CS472")
from tools import arff
import itertools
from decisiontree import DTClassifier
from sklearn.model_selection import train_test_split


data = [[0,0,0],
        [1,1,0],
        [1,2,1],
        [0,2,1],
        [0,1,0],
        [0,1,1],
        [1,0,1],
        [0,1,0],
        [1,0,0]]
labels = [[0],
          [1],
          [2],
          [0],
          [2],
          [0],
          [2],
          [2],
          [1]]
counts = [2,3,2,3] ## this is so you know how many types for each column
DTClass = DTClassifier(counts)
DTClass.fit(data,labels)
