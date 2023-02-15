import pandas as pd  
import numpy as np  
from pprint import pprint 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

# Text file data converted to integer data type
data = np.loadtxt("dummy.txt", dtype=float)

# data[:,0] = np.sort(data[:,0])
# data[:,1] = np.sort(data[:,1])
# data[:,2] = np.sort(data[:,2])


# print(data[:,2])
# print(data.shape[0])
# print(data.)

def entropy(data):

    class1 = np.sum(data[:,2].sum())
    class0 = data.shape[0] - class1
    size = len(data[:,2])

    if(class1 == 0 or class0 == 0):
        return 0

    # print(class1, class2, size)


    entropy =  -((class1/size)*np.log2(class1/size) + (class0/size)*np.log2(class0/size))
    
    
    # print(entropy)
    return entropy




def info_gain(data, data1, data2):
    # if(entropy(data) == 0 or entropy(data1) == 0 or entropy(data2) == 0):
    #     return 0
    
    info_gain = entropy(data) - ((len(data1)/len(data))*entropy(data1) + (len(data2)/len(data))*entropy(data2))
    return info_gain



def gain_ratio(data, data1, data2):
        
    if(len(data1) == 0 or len(data2) == 0):
        split_info = 0
        info_gainVAL = 0
        gain_ratio = 0
    else:
        # calculate info gain and split info
        info_gainVAL = info_gain(data, data1, data2)
        split_info = -((len(data1)/len(data))*np.log2(len(data1)/len(data)) + (len(data2)/len(data))*np.log2(len(data2)/len(data)))

        # skip candidates with zero split information
        if(split_info == 0):
            gain_ratio = 0
        else:
            gain_ratio = info_gainVAL/split_info
    

    # print("Info Gain: ", info_gainVAL)
    # print("Split Info: ", split_info)
    # print("Gain Ratio: ", gain_ratio, "\n")

    return gain_ratio


def FindBestSplit(data_x1, data_x2, C_x1, C_x2):


    gain_ratio_x1 = []
    gain_ratio_x2 = []

    # print()
    for i in C_x1:
        data1 = data_x1[:i]
        data2 = data_x1[i:]
        # print(i)
        # print(data1)
        # print(data2)
        # print("X1 split index: ", i, "")
        if(i == len(data_x1)):
            # print("X1 >=", data_x1[0,0])
            gain_ratio_x1.append((gain_ratio(data_x1, data1, data2),i))
        else:
            # print("X1 >=", data_x1[i,0])
            gain_ratio_x1.append((gain_ratio(data_x1, data1, data2),i))
    
    for i in C_x2:
        data1 = data_x2[:i]
        data2 = data_x2[i:]
        # print(i)
        # print(data1)
        # print(data2)
        if(i == len(data_x2)):
            # print("X2 >=", data_x2[0,1])
            gain_ratio_x2.append((gain_ratio(data_x2, data1, data2),i, 1))
        else: 
            # print("X2 >=", data_x2[i,1])
            gain_ratio_x2.append((gain_ratio(data_x2, data1, data2),i, 2))

    gr1 = max(gain_ratio_x1)
    gr2 = max(gain_ratio_x2)

    return max(gr1, gr2)
        


# Return splits of feature indices. Each index is the after the split
def DetermineCandidateSplits(data, feature):
    C = set()
    for i in range(1, len(data)):
        if(data[i, feature] != data[i-1, feature]):
            C.add(i)
    C.add(len(data))
    return C


def MakeSubtree(data):

    # Sort data based on each feature
    data_x1 = data[data[:, 0].argsort()]
    data_x2 = data[data[:, 1].argsort()]

    # Determine candidate splits for each feature
    C_x1 = DetermineCandidateSplits(data_x1, 0)
    C_x2 = DetermineCandidateSplits(data_x2, 1)

    # Determine if stopping criteria is met
    if(len(data)==0):
        print("Empty data")
        return Leaf(data, 1)
    
    
    # Create internal node
    nodeVal = Node(data)
    

    # Find best split for each feature
    S = FindBestSplit(data_x1, data_x2, C_x1, C_x2)
    # print(S)
    # If stopping criteria is met, return leaf node
    if(S[0] == 0):
        class1 = np.sum(data[:,2].sum())
        class0 = data.shape[0] - class1
        value = 0
        if(class1 == class0):
            value = 1
        else:
            if(class0 == class1):
                value = 1
            elif(class0 > class1):
                value = 0
            else:
                value = 1
        # print("Leaf Node: ", data, value)
        return Leaf(data, value)

    # print(S)
    data1 = data_x2[:S[1]]
    data2 = data_x2[S[1]:]


    # print("*****************************")
    # print("RIGHT")
    nodeVal.right = MakeSubtree(data1)
    # print("LEFT")
    nodeVal.left = MakeSubtree(data2)
    # print(C_x1)
    # print(C_x2)
    # print("*****************************")
    # print("Node: ", nodeVal.dataval[0])


    return nodeVal    

class Node:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.left= None
        self.right = None

class Leaf:
    def __init__(self, dataval=None, prediction=None):
        self.dataval = dataval
        self.prediction = prediction
        


# CODE START 
# Text file data converted to integer data type

dataset = np.loadtxt("Dbig.txt", dtype=float)

# Create scatter plots of data
# class0 = []
# class1 = []

# class0 = dataset[dataset[:,2] == 0]
# class1 = dataset[dataset[:,2] == 1]

# plt.scatter(class0[:,0], class0[:,1], color='red')
# plt.scatter(class1[:,0], class1[:,1], color='blue')
# plt.show()

# print("Class 0: ", class0)
# print("Class 1: ", class1)


X = dataset[:,0:2]
y = dataset[:,2]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8192, random_state=42)

DTest = np.concatenate((X_test, y_test[:,None]), axis=1)
D8192 = np.concatenate((X_train, y_train[:,None]), axis=1)
D2048 = D8192[0:2048]
D512 = D8192[0:512]
D128 = D8192[0:128]
D32 = D8192[0:32]



clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Error", 1 - accuracy_score(y_test, y_pred))
print("Number of nodes: ", clf.tree_.node_count)






