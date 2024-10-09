# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:17:52 2022

@author: tz22f646
"""
import os
import torch
import collections
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import torch.utils.data as Data
from pandas.core.frame import DataFrame


import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


import torch
import torch.nn as nn
import torch.nn.functional as F

score='synergy 0'
n_hop=2
n_memory=128
shuffle=True
validation_split=0.1
test_split=0.2 
num_workers=2
data_dir = 'C:/Users/tz22f646/Desktop/GCN/GraphSynergy-master/data/DrugCombDB'
threshold = 0
emb_dim = 64
protein_num = 15970
cell_num = 16
drug_num = 3
l1_decay = 1e-6

ppi_df = pd.read_excel(os.path.join(data_dir, 'protein-protein_network.xlsx'))
protein_node = list(set(ppi_df['protein_a']) | set(ppi_df['protein_b']))#set()是集合去重函数,|号的含义是取并集
mapping = {protein_node[idx]:idx for idx in range(len(protein_node))}

cpi_df = pd.read_csv(os.path.join(data_dir, 'meso/cell_protein_meso.csv'))
cell_node = list(set(cpi_df['cell']))

dpi_df = pd.read_csv(os.path.join(data_dir, 'meso/drug_protein_meso.csv'))
drug_node = list(set(dpi_df['drug']))

node_num_dict = {'protein': len(protein_node), 'cell': len(cell_node), 'drug': len(drug_node)}
#creating cell/drug/protein - index dict. The index was the corresponding index of the keys(cell/drug/protein) in xxx_node df.       
mapping = {protein_node[idx]:idx for idx in range(len(protein_node))}
mapping.update({cell_node[idx]:idx for idx in range(len(cell_node))})
mapping.update({drug_node[idx]:idx for idx in range(len(drug_node))})


#remap the interaction network
#这里的map不是Python自带的批量计算函数(类似于lapply)，而是pandas中的pandas.map()方法
#pandas.map() is used to map values from two series having one column same. 
# For mapping two series, the last column of the first series should be same as index column of the second series, also the values should be unique.
node_map_dict = mapping#self.node_map_dict是get_node_map_dict()函数的output，get_node_map_dict() returns two object, mapping and node_num_dict. Acoording to the defining order of the two variables, mapping is corresponded to self.node_map_dict
ppi_df['protein_a'] = ppi_df['protein_a'].map(node_map_dict)
ppi_df['protein_b'] = ppi_df['protein_b'].map(node_map_dict)
ppi_df = ppi_df[['protein_a', 'protein_b']]
#上段代码将ppi_df中的protein name转变成cell/drug/protein - index dict中protein name(key)对应的index(value)

cpi_df['cell'] = cpi_df['cell'].map(node_map_dict)
cpi_df['protein'] = cpi_df['protein'].map(node_map_dict)
cpi_df = cpi_df[['cell', 'protein']]

dpi_df['drug'] = dpi_df['drug'].map(node_map_dict)
dpi_df['protein'] = dpi_df['protein'].map(node_map_dict)
dpi_df = dpi_df[['drug', 'protein']]

#drug_combination_process
drug_combination_df = pd.read_csv(os.path.join(data_dir, 'meso/drug_combinations_meso.csv'))
drug_combination_df['drug1_db'] = drug_combination_df['drug1_db'].map(node_map_dict)
drug_combination_df['drug2_db'] = drug_combination_df['drug2_db'].map(node_map_dict)
drug_combination_df['cell'] = drug_combination_df['cell'].map(node_map_dict)
#设置threshold将synergy effect分为0，1
drug_combination_df['synergistic'] = [0] * len(drug_combination_df)
drug_combination_df.loc[drug_combination_df['synergy'] > threshold, 'synergistic'] = 1
drug_combination_df.to_csv(os.path.join(data_dir, 'drug_combination_processed.csv'), index=False)
drug_combination_df = drug_combination_df[['cell', 'drug1_db', 'drug2_db', 'synergistic']]#splicing the dataframe drug_combination_df for useful columns

#build_graph(self):
tuples = [tuple(x) for x in ppi_df.values]#数据框行转tuple,经典代码
graph = nx.Graph()#创建空图
graph.add_edges_from(tuples)
#get_target_dict
cp_dict = collections.defaultdict(list)#collections.defaultdict()用在，要生成一个字典，但是又没有默认值的情况，可以使用它来生成一个默认值，而不发生keyerror报错。括号内填入值的类型(键值对的值)
#Create cell-protein dict.The proteins were substantial drug targets in the corresponding keys(cells).
cell_list = list(set(cpi_df['cell']))
for cell in cell_list:
    cell_df = cpi_df[cpi_df['cell']==cell]
    target = list(set(cell_df['protein']))
    cp_dict[cell] = target

#Create drug-protein dict.The function of these proteins could be inhibited by corresponding keys(drugs).       
dp_dict = collections.defaultdict(list)
drug_list = list(set(dpi_df['drug']))
for drug in drug_list:
    drug_df = dpi_df[dpi_df['drug']==drug]
    target = list(set(drug_df['protein']))
    dp_dict[drug] = target

#create_dataset
# shape [n_data, 3]
feature = torch.from_numpy(drug_combination_df.iloc[:,0:3].to_numpy())
# change tensor type
feature = feature.type(torch.LongTensor)

#get_neighbor_set
#Among the target proteins in each cells, only 32 proteins were chosen into the first element of the list neighbor_set. 
#Among the 32 proteins, only the last protein was chosen. Among all the interacted proteins of the last protein, only 32 proteins were chosen into the second element of the list neighbor_set. 
neighbor_set = collections.defaultdict(list)
cells = list(cp_dict.keys())
for item in cp_dict:
    for hop in range(n_hop):#range(2)输出0，1
        # use the target directly
        if hop == 0:
            replace = len(cp_dict[item]) < n_memory
            target_list = list(np.random.choice(cp_dict[item], size=n_memory, replace=replace)) #np.random.choice从一维数据中随机抽取数字，返回指定大小(size)的数组
        else:
            # use the last one to find k+1 hop neighbors
            origin_nodes = neighbor_set[item][-1]
            neighbors = []
            for node in origin_nodes:
                neighbors += graph.neighbors(node)
                # sample
                replace = len(neighbors) < n_memory
                target_list = list(np.random.choice(neighbors, size=n_memory, replace=replace))                
        neighbor_set[item].append(target_list)
cell_neighbor_set=neighbor_set
                
neighbor_set = collections.defaultdict(list)
drugs = list(dp_dict.keys())
for item in dp_dict:
    for hop in range(n_hop):
        # use the target directly
        if hop == 0:
            replace = len(dp_dict[item]) < n_memory
            target_list = list(np.random.choice(dp_dict[item], size=n_memory, replace=replace)) #np.random.choice从一维数据中随机抽取数字，返回指定大小(size)的数组
        else:
            # use the last one to find k+1 hop neighbors
            origin_nodes = neighbor_set[item][-1]
            neighbors = []
            for node in origin_nodes:
                neighbors += graph.neighbors(node)
                # sample
                replace = len(neighbors) < n_memory
                target_list = list(np.random.choice(neighbors, size=n_memory, replace=replace))
        neighbor_set[item].append(target_list)
drugs_neighbor_set=neighbor_set



###constitute the model###
#####LINE算法求protein_embedding。LINE原理见https://blog.csdn.net/Gamer_gyt/article/details/117278065
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import LINE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
#G=nx.from_pandas_edgelist(ppi_df,source="protein_a",target="protein_b")
#model = LINE(G, embedding_size=64, order='second')
#model.train(batch_size=1024, epochs=50, verbose=2)
#protein_embedding = model.get_embeddings()
#with open(os.path.join(data_dir, 'protein_embedding.pickle'), 'wb') as f:
#    pickle.dump(protein_embedding, f)
#protein_embedding = torch.LongTensor(list(protein_embedding.values()))
def Z_ScoreNormalization(data):
    stdDf = data.std(0)
    meanDf = data.mean(0)
    normSet = (data - meanDf) / stdDf
    return normSet 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,#输入1个特征图
                out_channels=4,#用4个卷积核得到4张特征图
                kernel_size=3,#3*3卷积核
                stride=1,#卷积核滑动步长为1
                padding=1),#边缘填充1个单元
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,#输入4个特征图
                out_channels=8,#用2个卷积核得到8张特征图
                kernel_size=3,#3*3卷积核
                stride=1,#卷积核滑动步长为1
                padding=1),#边缘填充1个单元
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32*32*8,64) #全连接层，32*32*16个神经元输出10个参数

    def forward(self, item_neighbors_emb_list_for_emb):
        i = item_neighbors_emb_list_for_emb
        i = torch.unsqueeze(i,dim=1)#[item_size,128,128]~[item_size,1,128,128]
        i = self.conv1(i)#[item_size,1,128,128]~[item_size,4,64, 64]
        i = self.conv2(i)#[item_size,4,64, 64]~[item_size,8,32,32]
        i = i.view(i.size(0),-1) #将三维矩阵拉长为一维向量,[item_size,8,32,32]~[item_size,1,16384]
        i = torch.squeeze(self.out(i))#[item_size,1,16384]~[item_size,1,64]~[item_size,64]
        i = Z_ScoreNormalization(i)
        return i
CNN_net=CNN()

######get item_emb_pre######
FI=open(os.path.join(data_dir, 'protein_embedding.pickle'), 'rb')
protein_embedding=pickle.load(FI)  
protein_embedding=list(protein_embedding.values())
def list_slice(whole_list,mylist):
    t=[]
    for k in mylist:
        t.append(whole_list[k])
    return t
test=list_slice(protein_embedding,cell_neighbor_set[0][0])

def _get_neighbor_emb(neighbor_set, num): 
    neighbors_emb_list = []
    for n in range(num):
        neighbors_emb_list_each=[]
        neighbors = neighbor_set[n]
        for hop in range(n_hop):
            neighbors_emb_list_each.append(list_slice(protein_embedding,neighbors[hop]))
        neighbors_emb_list.append(np.hstack(neighbors_emb_list_each))#np.hstack横向合并list中的ndarray
    return neighbors_emb_list


cell_emb_pre = np.array(_get_neighbor_emb(cell_neighbor_set, cell_num))    
drug_emb_pre = np.array(_get_neighbor_emb(drugs_neighbor_set, drug_num))   


#def _get_neighbors_emb_tensor_for_emb(item_neighbors_emb_list):
#     t=[]
#     for i in item_neighbors_emb_list:
#         i = torch.concat(i,dim=1)
#         t.append(i)
#     item_neighbors_emb_list_for_embedding=torch.stack(t)
#     return item_neighbors_emb_list_for_embedding

FI=open(os.path.join(data_dir, 'protein_embedding.pickle'), 'rb')
protein_embedding=pickle.load(FI)   
protein_embedding = torch.FloatTensor(list(protein_embedding.values()))           
class GraphSynergy(torch.nn.Module):
    def __init__(self, therapy_method):
        super(GraphSynergy, self).__init__()
        #在构造函数中不要出现用包含循环的函数定义的参数，否则在之后的循环中会出现loss.backward()无法回传的现象
        self.protein_num = protein_num
        self.cell_num = cell_num
        self.drug_num = drug_num
        self.emb_dim = emb_dim
        self.n_hop = n_hop
        self.l1_decay = l1_decay
        self.therapy_method = therapy_method
        self.cell_neighbor_set =cell_neighbor_set
        self.drugs_neighbor_set=drugs_neighbor_set
        self.CNN_net=CNN()
        #self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)
        self.protein_embedding = nn.Embedding.from_pretrained(protein_embedding, freeze=False)#[protein_num，64]
        #self.cell_embedding = nn.Embedding(self.cell_num, self.emb_dim)#[cell_num，64]
        #self.drug_embedding = nn.Embedding(self.drug_num, self.emb_dim)#[drug_num，64]       
        
        self.cell_emb_pre = torch.from_numpy(cell_emb_pre)
        self.drug_emb_pre = torch.from_numpy(drug_emb_pre)

        self.aggregation_function = nn.Linear(self.emb_dim*self.n_hop, self.emb_dim)

        if self.therapy_method == 'transformation_matrix':
            self.combine_function = nn.Linear(self.emb_dim*2, self.emb_dim, bias=False)
        elif self.therapy_method == 'weighted_inner_product':
            self.combine_function = nn.Linear(in_features=2, out_features=1, bias=False)    

    def _get_neighbor_emb(self, neighbor_set, num): 
        neighbors_emb_list = []
        for n in range(num):
            neighbors_emb_list_each=[]
            neighbors = neighbor_set[n]
            for hop in range(self.n_hop):
                neighbors_emb_list_each.append(self.protein_embedding(torch.from_numpy(np.array(neighbors[hop]))))
            neighbors_emb_list.append(neighbors_emb_list_each)
        return neighbors_emb_list
    
    def _get_emb_pre(self, neighbor_set, num): 
        neighbors_emb_list = []
        for n in range(num):
            neighbors_emb_list_each=[]
            neighbors = neighbor_set[n]
            for hop in range(self.n_hop):
                neighbors_emb_list_each.append(self.protein_embedding(torch.from_numpy(np.array(neighbors[hop]))))
            neighbors_emb_list.append(torch.cat(neighbors_emb_list_each,dim=1))#np.hstack横向合并
        item_emb_pre=torch.stack(neighbors_emb_list)
        return  item_emb_pre

    def _interaction_aggregation(self, item_embeddings, num, neighbors_emb_list):    
        #求π(e,p),IS(e,p),即求attention机制下各细胞的embedding
        interact_list = []
        for n in range(num):
            interact_list_each = []
            for hop in range(self.n_hop):
                # [n_memory, dim]
                neighbor_emb = neighbors_emb_list[n][hop]
                # [dim, 1]
                item_embeddings_expanded = torch.unsqueeze(item_embeddings[n], dim=1)#[64]~[64,1]
                # [n_memory]
                contributions = torch.squeeze(torch.matmul(neighbor_emb,item_embeddings_expanded))#[128,64]*[64,1]~[128,1]~[128]    
                # [n_memory]
                contributions_normalized = F.softmax(contributions, dim=0)
                # [n_memory, 1]
                contributions_expaned = torch.unsqueeze(contributions_normalized, dim=1)#[128]~[128,1]    
                # [n_memory, dim]
                i = (neighbor_emb * contributions_expaned).sum(dim=0)#矩阵与一维数组相乘，等于每行矩阵乘同行数的数组元素,[128,64]~[128,64]
                # update item_embeddings
                i = torch.unsqueeze(i,dim=0)
                interact_list_each.append(i)
            interact_list.append(interact_list_each)
        return interact_list
    
    def _aggregation(self, item_i_list):
        # [batch_size, n_hop+1, emb_dim]
        item_i_concat = torch.cat(item_i_list, dim=1)#[b,1,64]~[b,1,128]
        # [batch_size, emb_dim]
        item_embeddings = self.aggregation_function(item_i_concat)#[b,1,128]~[b,1,64]
        return item_embeddings

    def _therapy(self, drug1_embeddings, drug2_embeddings, cell_embeddings):
        if self.therapy_method == 'transformation_matrix':
            combined_drug = self.combine_function(torch.cat([drug1_embeddings, drug2_embeddings], dim=2))#[b,1,64]*2~[b,1,128]~[b,1,64]
            therapy_score = (combined_drug * cell_embeddings).sum(dim=2)#[b,1,64]~[b]
        elif self.therapy_method == 'weighted_inner_product':
            drug1_score = torch.unsqueeze((drug1_embeddings * cell_embeddings).sum(dim=2), dim=2)
            drug2_score = torch.unsqueeze((drug2_embeddings * cell_embeddings).sum(dim=2), dim=2)
            therapy_score = torch.squeeze(self.combine_function(torch.cat([drug1_score, drug2_score], dim=2)))
        elif self.therapy_method == 'max_pooling':
            combine_drug = torch.max(drug1_embeddings, drug2_embeddings)
            therapy_score = (combine_drug * cell_embeddings).sum(dim=2)
        return therapy_score
    
    def _toxic(self, drug1_embeddings, drug2_embeddings):
        return (drug1_embeddings * drug2_embeddings).sum(dim=2)
    
    
    def _emb_loss(self, cell_embeddings, drug1_embeddings, drug2_embeddings, cell_neighbors_emb_list, drug1_neighbors_emb_list, drug2_neighbors_emb_list):#self不可省略，否则报错:_emb_loss() takes 6 positional arguments but 7 were given
        item_regularizer = (torch.norm(cell_embeddings) ** 2
                          + torch.norm(drug1_embeddings) ** 2
                          + torch.norm(drug2_embeddings) ** 2) / 2
        node_regularizer = 0
        for hop in range(n_hop):
            node_regularizer += (torch.norm(cell_neighbors_emb_list[hop]) ** 2
                              +  torch.norm(drug1_neighbors_emb_list[hop]) ** 2
                              +  torch.norm(drug2_neighbors_emb_list[hop]) ** 2) / 2    
        emb_loss = l1_decay * (item_regularizer + node_regularizer) / cell_embeddings.shape[0]
        return emb_loss
       
    def forward(self,data):
        cells=data[:,0]
        drug1=data[:,1]
        drug2=data[:,2]
    
        cell_neighbors_emb_list = self._get_neighbor_emb(self.cell_neighbor_set, self.cell_num)
        drug_neighbors_emb_list = self._get_neighbor_emb(self.drugs_neighbor_set, self.drug_num)
        
        #cell_emb_pre = self._get_emb_pre(self.cell_neighbor_set, self.cell_num)
        #drug_emb_pre = self._get_emb_pre(self.drugs_neighbor_set, self.drug_num)
        
        cell_emb_pre = self.cell_emb_pre
        drug_emb_pre = self.drug_emb_pre
        
        cell_embeddings = self.CNN_net(cell_emb_pre)#Tensor格式用list切片，Embedding格式用torch.LongTensor调取为Tensor格式
        drug_embeddings = self.CNN_net(drug_emb_pre)
        
        #cell_embeddings = nn.Embedding.from_pretrained(cell_embeddings, freeze=False)
        #drug_embeddings = nn.Embedding.from_pretrained(drug_embeddings, freeze=False)
        
        #cell_embeddings = cell_embeddings(torch.from_numpy(np.array(cell_list)))
        #drug_embeddings = drug_embeddings(torch.from_numpy(np.array(drug_list)))
        
        cell_i_list = self._interaction_aggregation(cell_embeddings, self.cell_num, cell_neighbors_emb_list)
        drug_i_list = self._interaction_aggregation(drug_embeddings, self.drug_num, drug_neighbors_emb_list)

        cell_embeddings=[]
        for n in range(self.cell_num):
            i=self._aggregation(cell_i_list[n])
            cell_embeddings.append(i)
        
        drug_embeddings=[]
        for n in range(self.drug_num):
            i=self._aggregation(drug_i_list[n])
            drug_embeddings.append(i)        

        cell_embeddings = torch.stack(cell_embeddings)#将以tensor为元素的list转变为tensor用torch.stack
        drug_embeddings = torch.stack(drug_embeddings)
        
        t=[]
        for i in cell_neighbors_emb_list:
            i=torch.stack(i)
            t.append(i)
        cell_neighbors_emb_list=torch.stack(t)
        
        t=[]
        for i in drug_neighbors_emb_list:
            i=torch.stack(i)
            t.append(i)
        drug_neighbors_emb_list=torch.stack(t)
        
        cell_embeddings = cell_embeddings[cells]
        drug1_embeddings = drug_embeddings[drug1]
        drug2_embeddings = drug_embeddings[drug2]
       
        cell_neighbors_emb_list = cell_neighbors_emb_list[cells]
        drug1_neighbors_emb_list = drug_neighbors_emb_list[drug1]
        drug2_neighbors_emb_list = drug_neighbors_emb_list[drug2]
        

        score = self._therapy(drug1_embeddings, drug2_embeddings, cell_embeddings) - \
                    self._toxic(drug1_embeddings, drug2_embeddings)
    
        # embedding loss
        emb_loss = self._emb_loss(cell_embeddings, drug1_embeddings, drug2_embeddings, cell_neighbors_emb_list, drug1_neighbors_emb_list, drug2_neighbors_emb_list)
        return score, emb_loss    

model = GraphSynergy('transformation_matrix')

#加载最佳模型
filename = "C:/Users/tz22f646/Desktop/GCN/checkpoint.pth"
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model.load_state_dict(checkpoint['state_dict'])
output, emb_loss= model(feature)
y_pred = torch.sigmoid(output)
