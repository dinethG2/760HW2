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
            gain_ratio_x2.append((gain_ratio(data_x2, data1, data2),i))
        else: 
            # print("X2 >=", data_x2[i,1])
            gain_ratio_x2.append((gain_ratio(data_x2, data1, data2),i))

    gr1 = max(gain_ratio_x1)
    gr2 = max(gain_ratio_x2)

    if(gr1[0] > gr2[0]):
        return (gr1, "x1")
    else:
        return (gr2, "x2")
        


# Return splits of feature indices. Each index is the after the split
def DetermineCandidateSplits(data, feature):
    C = set()
    for i in range(1, len(data)):
        if(data[i, feature] != data[i-1, feature]):
            C.add(i)
    C.add(len(data))
    return C


def predict(data, node):
    if(isinstance(node, Leaf)):
        # print("Prediction: ", node.prediction)
        return node.prediction

    # print(node.feature, node.dataval)
    if(node.feature == "x1"):
        if(data[0] >= node.dataval):
            return predict(data, node.left)
        else:
            return predict(data, node.right)
    else:
        if(data[1] >= node.dataval):
            return predict(data, node.left)
        else:
            return predict(data, node.right)


def countNodes(node):
    if(isinstance(node, Leaf)):
        return 1
    return 1 + countNodes(node.left) + countNodes(node.right)

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
        return Leaf(1)
    
   
    

    # Find best split for each feature
    S = FindBestSplit(data_x1, data_x2, C_x1, C_x2)
    # print(S)
    # If stopping criteria is met, return leaf node
    if(S[0][0] == 0):
        class1 = np.sum(data[:,2].sum())
        class0 = data.shape[0] - class1

       

        value = 0
        if(class1 == class0):
            value = 1
        elif(class0 > class1):
                value = 0
        else:
                value = 1
        # print("Class 1: ", class1)
        # print("Class 0: ", class0)
        # print("Leaf Node: ", value)
        # print(data)
        return Leaf(value)

     
    # Create internal node
    # print(S)
    # print("Internal Node: ", S[0][0], S[1])
    if(S[1] == "x1"):
        nodeVal = Node(data_x1[S[0][1]][0], S[1])
        data1 = data_x1[:S[0][1]]
        data2 = data_x1[S[0][1]:]
    else:
        nodeVal = Node(data_x2[S[0][1]][1], S[1])
        data1 = data_x2[:S[0][1]]
        data2 = data_x2[S[0][1]:]
    

    # print(S[1])
   

    nodeVal.right = MakeSubtree(data1)
    nodeVal.left = MakeSubtree(data2)
   
    # print(nodeVal.dataval, nodeVal.feature)

    return nodeVal    

class Node:
    def __init__(self, dataval=None, feature=None):
        self.dataval = dataval
        self.feature = feature
        self.left= None
        self.right = None

class Leaf:
    def __init__(self, prediction=None):
        self.prediction = prediction
        


# CODE START 
# Text file data converted to integer data type
dataset = np.loadtxt("Dbig.txt", dtype=float)


# D1 & D2 ***************************************************
# print("Visualize")
# Measured = [0] * len(dataset[:,2])
# Tree = MakeSubtree(dataset)
# Tree_0 = []
# Tree_1 = []
# for i in range(0, len(dataset[:,2])):
#     Measured[i] = predict(dataset[i,0:2], Tree)
#     if(Measured[i] == 0):
#         Tree_0.append(dataset[i,0:2].tolist())
#     else:
#         Tree_1.append(dataset[i,0:2].tolist())

# Tree_0 = np.array(Tree_0)
# Tree_1 = np.array(Tree_1)
# plt.scatter(Tree_0[:,0], Tree_0[:,1], color='red')
# plt.scatter(Tree_1[:,0], Tree_1[:,1], color='blue')
# plt.legend(['Class 0', 'Class 1'])
# plt.title('Visualization')
# plt.show()



# Create scatter plots of training data ****************************************
# class0 = []
# class1 = []

# class0 = dataset[dataset[:,2] == 0]
# class1 = dataset[dataset[:,2] == 1]

# plt.scatter(class0[:,0], class0[:,1], color='red')
# plt.scatter(class1[:,0], class1[:,1], color='blue')
# plt.show()

# print("Class 0: ", class0)
# print("Class 1: ", class1)


# X = dataset[:,0:2]
# y = dataset[:,2]
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8192, random_state=42)

# DTest = np.concatenate((X_test, y_test[:,None]), axis=1)
# D8192 = np.concatenate((X_train, y_train[:,None]), axis=1)
# D2048 = D8192[0:2048]
# D512 = D8192[0:512]
# D128 = D8192[0:128]
# D32 = D8192[0:32]


# numNodes = [0] * 5
# error = [0] * 5

# D8192 ****************************************************
# print("D8192")
# D8192_measured = [0] * len(y_test)
# D8192_tree = MakeSubtree(D8192)
# D8192_0 = []
# D8192_1 = []
# for i in range(0, len(y_test)):
#     D8192_measured[i] = predict(X_test[i], D8192_tree)
#     if(D8192_measured[i] == 0):
#         D8192_0.append(X_test[i].tolist())
#     else:
#         D8192_1.append(X_test[i].tolist())

# D8192_0 = np.array(D8192_0)
# D8192_1 = np.array(D8192_1)
# plt.scatter(D8192_0[:,0], D8192_0[:,1], color='red')
# plt.scatter(D8192_1[:,0], D8192_1[:,1], color='blue')
# plt.legend(['Class 0', 'Class 1'])
# plt.title('8192 Training Data')
# plt.show()

# numNodes[4] = countNodes(D8192_tree)
# error[4] = np.sum(D8192_measured != y_test) / len(y_test)
# print("Number of Nodes: ", numNodes[4])
# print("Error: ", error[4])

# D2048 ****************************************************
# print("D2048")
# D2048_measured = [0] * len(y_test)
# D2048_tree = MakeSubtree(D2048)
# D2048_0 = []
# D2048_1 = []
# for i in range(0, len(y_test)):
#     D2048_measured[i] = predict(X_test[i], D2048_tree)
#     if(D2048_measured[i] == 0):
#         D2048_0.append(X_test[i].tolist())
#     else:
#         D2048_1.append(X_test[i].tolist())

# D2048_0 = np.array(D2048_0)
# D2048_1 = np.array(D2048_1)
# plt.scatter(D2048_0[:,0], D2048_0[:,1], color='red')
# plt.scatter(D2048_1[:,0], D2048_1[:,1], color='blue')
# plt.legend(['Class 0', 'Class 1'])
# plt.title('2048 Training Data')
# plt.show()


# numNodes[3] = countNodes(D2048_tree)
# error[3] = np.sum(D2048_measured != y_test) / len(y_test)
# print("Number of Nodes: ", numNodes[3])
# print("Error: ", error[3])


# # # D512 ****************************************************
# print("D512")
# D512_measured = [0] * len(y_test)
# D512_tree = MakeSubtree(D512)
# D512_0 = []
# D512_1 = []
# for i in range(0, len(y_test)):
#     D512_measured[i] = predict(X_test[i], D512_tree)
#     if(D512_measured[i] == 0):
#         D512_0.append(X_test[i].tolist())
#     else:
#         D512_1.append(X_test[i].tolist())

# D512_0 = np.array(D512_0)
# D512_1 = np.array(D512_1)

# plt.scatter(D512_0[:,0], D512_0[:,1], color='red')
# plt.scatter(D512_1[:,0], D512_1[:,1], color='blue')
# plt.legend(['Class 0', 'Class 1'])
# plt.title('512 Training Data')
# plt.show()

# numNodes[2] = countNodes(D512_tree)
# error[2] = np.sum(D512_measured != y_test) / len(y_test)
# print("Number of Nodes: ", numNodes[2])
# print("Error: ", error[2])


# # # D128 ****************************************************
# print("D128")
# D128_measured = [0] * len(y_test)
# D128_0 = []
# D128_1 = []
# D128_tree = MakeSubtree(D128)
# for i in range(0, len(y_test)):
#     D128_measured[i] = predict(X_test[i], D128_tree)
#     if(D128_measured[i] == 0):
#         D128_0.append(X_test[i].tolist())
#     else:
#         D128_1.append(X_test[i].tolist())

# D128_0 = np.array(D128_0)
# D128_1 = np.array(D128_1)

# plt.scatter(D128_0[:,0], D128_0[:,1], color='red')
# plt.scatter(D128_1[:,0], D128_1[:,1], color='blue')
# plt.legend(['Class 0', 'Class 1'])
# plt.title('128 Training Data')
# plt.show()

# numNodes[1] = countNodes(D128_tree)
# error[1] = np.sum(D128_measured != y_test) / len(y_test)
# print("Number of Nodes: ", numNodes[1])
# print("Error: ", error[1])


# # # D32 ****************************************************
# print("D32")
# D32_measured = [0] * len(y_test)
# D32_tree = MakeSubtree(D32)
# D32_0 = []
# D32_1 = []
# for i in range(0, len(y_test)):
#     D32_measured[i] = predict(X_test[i], D32_tree)
#     if(D32_measured[i] == 0):
#         D32_0.append(X_test[i].tolist())
#     else:
#         D32_1.append(X_test[i].tolist())

# D32_0 = np.array(D32_0)
# D32_1 = np.array(D32_1)

# plt.scatter(D32_0[:,0], D32_0[:,1], color='red')
# plt.scatter(D32_1[:,0], D32_1[:,1], color='blue')
# plt.legend(['Class 0', 'Class 1'])
# plt.title('32 Training Data')
# plt.show()

# numNodes[0] = countNodes(D32_tree)
# error[0] = np.sum(D32_measured != y_test) / len(y_test)
# print("Number of Nodes: ", numNodes[0])
# print("Error: ", error[0])

# # Plot the number of nodes vs error ****************************************************
# plt.plot(numNodes, error)
# plt.xlabel('Number of Nodes')
# plt.ylabel('Error')
# plt.title('Number of Nodes vs Error')
# plt.show()

# # Create scatter plots of data
# D8192_0 = []
# D8192_1 = []

# D8192_0 = D8192[D8192[:,2] == 0]
# D8192_1 = D8192[D8192[:,2] == 1]

# plt.scatter(D8192_0[:,0], D8192_0[:,1], color='red')
# plt.scatter(D8192_1[:,0], D8192_1[:,1], color='blue')
# plt.legend(['Class 0', 'Class 1'])
# plt.title('8192 Training Data')
# plt.show()



# Using sklearn to compare results ****************************************************

# numNodes = [0] * 5
# error = [0] * 5

# # 8192
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(D8192[:,0:2], D8192[:,2])
# y_pred = clf.predict(X_test)
# numNodes[4] = clf.tree_.node_count
# error[4] = 1 - accuracy_score(y_test, y_pred)
# print("8192: ", numNodes[4], error[4])

# # 2048
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(D2048[:,0:2], D2048[:,2])
# y_pred = clf.predict(X_test)
# numNodes[3] = clf.tree_.node_count
# error[3] = 1 - accuracy_score(y_test, y_pred)
# print("2048: ", numNodes[3], error[3])

# # 512
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(D512[:,0:2], D512[:,2])
# y_pred = clf.predict(X_test)
# numNodes[2] = clf.tree_.node_count
# error[2] = 1 - accuracy_score(y_test, y_pred)
# print("512: ", numNodes[2], error[2])

# # 128
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(D128[:,0:2], D128[:,2])
# y_pred = clf.predict(X_test)
# numNodes[1] = clf.tree_.node_count
# error[1] = 1 - accuracy_score(y_test, y_pred)
# print("128: ", numNodes[1], error[1])

# # 32
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(D32[:,0:2], D32[:,2])
# y_pred = clf.predict(X_test)
# numNodes[0] = clf.tree_.node_count
# error[0] = 1 - accuracy_score(y_test, y_pred)
# print("32: ", numNodes[0], error[0])

# # Plot number of nodes vs error
# plt.plot(numNodes, error)
# plt.xlabel("Number of Nodes")
# plt.ylabel("Error")
# plt.show()







