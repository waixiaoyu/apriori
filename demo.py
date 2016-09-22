# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:01:43 2016

@author: vladimir
"""

import numpy as np
from scipy.sparse import lil_matrix
from matplotlib import pyplot as plt
import networkx as nx
from fim import apriori

import itertools 

def test_data():
    data = [ [ 1, 2, 3 ], [ 1, 4, 5 ], [ 2, 3, 4 ], [ 1, 2, 3, 4 ],[ 2, 3 ],[ 1, 2, 4 ],[ 4, 5 ],[ 1, 2, 3, 4 ],[ 3, 4, 5 ],[ 1, 2, 3 ] ]
    return data
   
###############################################################################
# Read data  
###############################################################################
data = test_data()

###############################################################################
# Some basic data analysis
###############################################################################

# Find items list
items = np.unique([item for sublist in data for item in sublist])

# Size of data
N_baskets = len(data)
M_items = len(items)


###############################################################################
# Convert data into vector space format.  We will use a sparse boolean matrix
#   to represent data
###############################################################################
H =lil_matrix((N_baskets,M_items), dtype=np.bool)
for i in range(0,len(data)):
    for j in list(map(int,data[i])):
        H[i,j-1] = True

# Plot this matrix
plt.figure(1)
plt.subplot(121)
plt.spy(H)
plt.title('Vector representation')
plt.xlabel('Items')
plt.ylabel('Baskets')
plt.show()


###############################################################################
# Convert data into graph format
###############################################################################
g = nx.Graph()
a=['b_'+str(i) for i in range(N_baskets)]
b=['i_'+str(j) for j in range(M_items)]
g.add_nodes_from(a,bipartite=0)
g.add_nodes_from(b,bipartite=1)

i=0
for basket in data:
    for item in basket:
            g.add_edge(a[i], b[list(items).index(item)])
    i+=1

# Draw this graph
pos_a={}
x=0.100
const=0.100
y=1.0
for i in range(len(a)):
    pos_a[a[i]]=[x,y-i*const]

xb=0.500
pos_b={}
for i in range(len(b)):
    pos_b[b[i]]=[xb,y-i*const]

plt.subplot(121)
nx.draw_networkx_nodes(g,pos_a,nodelist=a,node_color='r',node_size=300,alpha=0.8)
nx.draw_networkx_nodes(g,pos_b,nodelist=b,node_color='b',node_size=300,alpha=0.8)

# edges
pos={}
pos.update(pos_a)
pos.update(pos_b)
nx.draw_networkx_edges(g,pos,edgelist=nx.edges(g),width=1,alpha=0.8,edge_color='g')
nx.draw_networkx_labels(g,pos,font_size=10,font_family='sans-serif')

plt.title('Graph representation')
plt.show()


###############################################################################
# Now do rule finding
###############################################################################

frequent_itemset = apriori(data, supp=-3, zmin=2, target='s', report='a')
rules = apriori(data, supp=-3, zmin=2, target='r', report='rCL')

print(frequent_itemset)
print(rules)



frequent_itemset_1=[]
frequent_itemset_2=[]
frequent_itemset_3=[]
'''
original apriori, generate 1 item
''' 
def apriori(data,supp=3,zmin,target,report):
    frequent_itemset=[]
    items_all = [item for sublist in data for item in sublist]
    items_number=[x for x in range(0, len(items))]
    for i in range(len(items_all)):
        items_number[items_all[i]-1]+=1
    for i in range(len(items_number)):
        if items_number[i] > supp:
            frequent_itemset.append([i+1,items_number[i]])
            frequent_itemset_1=frequent_itemset
'''
generate item pairs
'''       
def apriori_1(data,supp=3,zmin,frequent_itemset):
    frequent_itemset_new=[]
    for i in range(len(frequent_itemset)):
        for j in range(i,len(frequent_itemset_1)):
            if not frequent_itemset_1[j][0] in frequent_itemset[i][:len(frequent_itemset[i])-1]:
                item_new=[]
                item_new.extend(frequent_itemset[i][:len(frequent_itemset[i])-1])
                item_new.extend(frequent_itemset_primary[j])
                frequent= caculate_frequent(data,item_new) 
                if frequent > supp:
                    item_new.append(frequent)
                    frequent_itemset_new.append(item_new)
    frequent_itemset_2=frequent_itemset_new
 
'''
generate 3-tuple itemsets
'''
def apriori_2(data,supp=3,zmin,frequent_itemset):
    frequent_itemset_new=[]
    combinations=getCombinations(frequent_itemset,3)
    for i in range(len(combinations)):
        item_new=checkTuple(combinations[i],frequent_itemset_2,3)
        if len(item_new):
            frequent= caculate_frequent(data,item_new)
            if frequent >= supp:
                item_new.append(frequent)
                frequent_itemset_new.append(item_new)        
    frequent_itemset_3=getUniqueList(frequent_itemset_new)
'''
get the frequent of certain item according to original dataset
'''           
def caculate_frequent(data,item):
    frequent=0;
    for i in range(len(data)):
        bool_result=True
        for j in range(len(item)):
            if item[j] in data[i]:
                bool_result=bool_result and True
            else:
                bool_result=bool_result and False
        if bool_result:
            frequent+=1
    return frequent
'''
When checking k-tuples, we need get combinations of k-1-tuples.
'''
def getCombinations(frequent_itemset,n):
    return list(itertools.combinations([i for i in range(len(frequent_itemset))],n))

'''
According to apriori algorithm, k-tuple itemsets should be generated from k-1-tuple itemsets.
This function is for checking whether choosen several itemsets can create a new itemset. 
'''
def checkTuple(combination,frequent_itemset,k):
    itemset_temp=[]
    for i in range(k):
        for j in range(k-1):
            if not frequent_itemset[combination[i]][j] in itemset_temp:
                itemset_temp.append(frequent_itemset[combination[i]][j])
    
    if len(itemset_temp)==k+1:
        return itemset_temp
    else:
        return []

'''
this function is for sorting results, and filtering the repeating results.
'''
def getUniqueList(old_list):
    for i in range(len(old_list)):
        temp_list=old_list[i][:len(old_list[i])-1]
        temp_list=list(np.sort(temp_list))
        temp_list.append(old_list[i][len(old_list[i])-1])
        old_list[i]=temp_list
        
    newList = []
    for x in old_list:
        if x not in newList :
            newList.append(x)
    return newList