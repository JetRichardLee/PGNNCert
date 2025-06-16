# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import sys
sys.path.append("models/")
#from mlp import MLP
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import copy
from utils import evaluate, store_checkpoint, load_best_model, train_model,store_vote_checkpoint
from sklearn.model_selection import train_test_split

from gnn import NodeGCN,NodeGAT,NodeGSAGE
from gnn import GraphGCN,GraphGAT,GraphGSAGE

device =  "cuda" if torch.cuda.is_available() else "cpu"
class HashAgent():
    def __init__(self,h="md5",T=30):
        '''
            h: the hash function in "md5","sha1","sha256"
            T: the subset amount
        '''

        super(HashAgent, self).__init__()
        self.T = T
        self.h= h 
        
            
    def hash_node(self,u):
        #"""
        hexstring = hex(u)
        hexstring= hexstring.encode()
        if self.h == "md5":
            hash_device = hashlib.md5()
        elif self.h == "sha1":
            hash_device = hashlib.sha1()
        elif self.h == "sha256":
            hash_device = hashlib.sha256()
        hash_device.update(hexstring)
        I = int(hash_device.hexdigest(),16)%self.T
        
        return I
    
    def generate_node_subgraphs(self, edge_index, x, y):
        
        subgraphs = []
        
        original = edge_index
        nodes = range(x.shape[0])

        V= x.shape[0]
                    
        for i in range(self.T):
            subgraphs.append(Data(
                        x = x,
                        y = y,
                        edge_index = []
                    ))
        for i in range(len(original[0])):
            u=original[0,i]
            v=original[1,i]
            I = self.hash_node(u)
            subgraphs[I].edge_index.append([u,v])
            
        new_subgraphs = []
        for i in range(self.T):
            if len(subgraphs[i].edge_index)==0:
                continue
            subgraphs[i].edge_index = torch.tensor(subgraphs[i].edge_index,dtype=torch.int64).transpose(1,0)
            new_subgraphs.append(subgraphs[i])
            
        return new_subgraphs

    def generate_graph_subgraphs(self, edge_index, x, y):
        
        subgraphs = []
        
        original = edge_index
        nodes = range(x.shape[0])
        zerox = torch.zeros(x[0].size()).reshape(1,-1)
        V= x.shape[0]
        xs = [[] for i in range(self.T)]
        mappings = []
        for i in range(self.T):
            subgraphs.append(Data(
                        x = zerox,
                        y = y,
                        edge_index = [[0,0]]
                    ))
            
        for i in range(x.shape[0]):
            I = self.hash_node(i)
            mappings.append(subgraphs[I].x.shape[0])
            subgraphs[I].x = torch.cat((subgraphs[I].x,x[i].reshape(1,-1)),dim=0)
            subgraphs[I].edge_index.append([mappings[i],0])
            
        for i in range(len(original[0])):
            
            u=original[0,i]
            v=original[1,i]
            I = self.hash_node(u)
            if self.hash_node(v)==I:
                subgraphs[I].edge_index.append([mappings[u],mappings[v]])
            
        for i in range(self.T):
            subgraphs[i].edge_index = torch.tensor(subgraphs[i].edge_index,dtype=torch.int64).transpose(1,0)

            
        return subgraphs
    
    
class RobustNodeClassifier(torch.nn.Module):
    def __init__(self,Hasher,edge_index, x, y, train_mask, val_mask, test_mask,num_x,num_labels,GNN="GCN"):

        super(RobustNodeClassifier, self).__init__()
        self.Hasher = Hasher
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.edge_index = edge_index
        self.x = x
        self.y = y.to(self.device)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_labels= num_labels
        self.T = self.Hasher.T
        self.classifiers = []
        if GNN =="GCN":
            for i in range(self.T):
                self.classifiers.append(NodeGCN(num_x,num_labels).to(self.device))        
                self.register_module("Classifier_{}".format(i), self.classifiers[-1])
        elif GNN =="GAT":
            for i in range(self.T):
                self.classifiers.append(NodeGAT(num_x,num_labels).to(self.device))        
                self.register_module("Classifier_{}".format(i), self.classifiers[-1])
        elif GNN =="GSAGE":
            for i in range(self.T):
                self.classifiers.append(NodeGSAGE(num_x,num_labels).to(self.device))        
                self.register_module("Classifier_{}".format(i), self.classifiers[-1])
              
                
    def load_model(self,path):
        for i in range(self.T):
            checkpoint = torch.load(path+f"_{i}")
            self.classifiers[i].load_state_dict(checkpoint['model_state_dict'])
    
    def train(self, train_args ):
        subgraphs = self.Hasher.generate_node_subgraphs(self.edge_index, self.x, self.y)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=train_args["lr"])

        criterion = torch.nn.CrossEntropyLoss()
    
        best_val_acc = 0.0
        best_epoch = 0
        for i in range(self.T):
            self.classifiers[i].train()
        for epoch in range(0, train_args["epochs"]):
            loss = torch.zeros(1).to(self.device)
            optimizer.zero_grad()
            
            for i in range(len(subgraphs)):
                out_sub = self.classifiers[i](subgraphs[i].x.to(self.device),subgraphs[i].edge_index.to(self.device))
                loss+=criterion(out_sub[self.train_mask], self.y[self.train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=train_args["clip_max"])
            optimizer.step()
            train_acc = 0
            test_acc = 0
            val_acc = 0
            with torch.no_grad():
                out_train,_ = self.vote(self.train_mask)
                out_val,_ = self.vote(self.val_mask)
                out_test,_ = self.vote(self.test_mask)
                train_acc = evaluate(out_train.to(self.device), self.y[self.train_mask])
                val_acc = evaluate(out_val.to(self.device), self.y[self.val_mask])
                test_acc = evaluate(out_test.to(self.device), self.y[self.test_mask])
                
            print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss.item():.4f}")
            if val_acc > best_val_acc: # New best results
                print("Val improved")
                best_val_acc = val_acc
                best_epoch = epoch
                store_vote_checkpoint("robust_n/"+train_args["paper"], train_args["dataset"]+"/{}".format(self.T), self.classifiers, train_acc, val_acc, test_acc)

            if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
                break
        
     
    def test(self ):   
        
        out_test,M = self.vote(self.test_mask)
        test_acc = evaluate(out_test, self.y[self.test_mask])
        return test_acc, M
        
    def vote(self, mask):
        subgraphs = self.Hasher.generate_node_subgraphs(self.edge_index, self.x, self.y)
        V_test = self.x[mask].shape[0]
        votes = torch.zeros((V_test,self.num_labels))
        for i in range(len(subgraphs)):
            self.classifiers[i].eval()
            out_sub = self.classifiers[i](subgraphs[i].x.to(self.device),subgraphs[i].edge_index.to(self.device))
            preds = out_sub[mask].argmax(dim=1)
            for j in range(V_test):
                votes[j,preds[j]]+=1
 
        vote_label = votes.argmax(dim=1)
        M =torch.zeros(V_test)
        for i in range(V_test):
            votes[i,vote_label[i]]=-votes[i,vote_label[i]]
        second_label = votes.argmax(dim=1)
        for i in range(V_test):
            votes[i,vote_label[i]]=-votes[i,vote_label[i]]
        for i in range(V_test):
            if vote_label[i]>second_label[i]:
                M[i] = (votes[i,vote_label[i]]-votes[i,second_label[i]]-1)//2
            else:
                M[i] = (votes[i,vote_label[i]]-votes[i,second_label[i]])//2
        return votes, M
    
class RobustGraphClassifier(torch.nn.Module):
    def __init__(self,Hasher,graphs,labels,train_mask, val_mask, test_mask,num_x,num_labels,GNN="GCN"):

        super(RobustGraphClassifier, self).__init__()
        self.Hasher = Hasher
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_labels= num_labels
        self.graphs=graphs
        self.labels = torch.tensor(labels)
        self.subgraphs=[]
        self.T = self.Hasher.T
        self.classifiers = []
        
        if GNN =="GCN":
            for i in range(self.T):
                self.classifiers.append(GraphGCN(num_x,num_labels).to(self.device))        
                self.register_module("Classifier_{}".format(i), self.classifiers[-1])
        elif GNN =="GAT":
            for i in range(self.T):
                self.classifiers.append(GraphGAT(num_x,num_labels).to(self.device))        
                self.register_module("Classifier_{}".format(i), self.classifiers[-1])
        elif GNN =="GSAGE":
            for i in range(self.T):
                self.classifiers.append(GraphGSAGE(num_x,num_labels).to(self.device))        
                self.register_module("Classifier_{}".format(i), self.classifiers[-1])
                
        self.subgraphsX  = [[] for _ in range(self.T)]
        self.subgraphsE  = [[] for _ in range(self.T)]
        
        for i in range(len(graphs)):
            subgraphs = self.Hasher.generate_graph_subgraphs(graphs[i].edge_index,graphs[i].x,graphs[i].y)
            for j in range(self.T):
                self.subgraphsX[j].append(subgraphs[j].x.to(self.device))
                self.subgraphsE[j].append(subgraphs[j].edge_index.to(self.device))
            
    def load_model(self,path):
        for i in range(self.T):
            checkpoint = torch.load(path+f"_{i}")
            self.classifiers[i].load_state_dict(checkpoint['model_state_dict'])
    
    def enlarge_dataset(self, graphs):
        new_graphs = []
        ys = []
        for i in range(len(graphs)):
            subgraphs = self.Hasher.generate_graph_subgraphs(graphs[i].edge_index,graphs[i].x,graphs[i].y)
            new_graphs.append([])
            ys.append([])
            for j in range(self.T):
                new_graphs[-1].append(subgraphs[j].to(self.device))
                ys[-1].append(subgraphs[j].y)
                
            ys[-1]= torch.tensor(ys[-1],dtype=subgraphs[0].y.dtype).to(self.device)
        return new_graphs, ys
    
    
    def train(self, train_args ):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=train_args["lr"])
        criterion = torch.nn.CrossEntropyLoss()
    
        best_val_acc = 0.0
        best_train_acc = 0.0
        best_epoch = 0
            
        train_graphs = self.graphs[self.train_mask]
        
        entrain_graphs,ys = self.enlarge_dataset(train_graphs)
        for epoch in range(0, train_args["epochs"]):
            optimizer.zero_grad()
            loss = torch.zeros(1).to(self.device)
            for j in range(self.T):
                self.classifiers[j].train()
                
            for i in range(len(entrain_graphs)):
                out = torch.zeros((self.T,self.num_labels)).to(self.device)
                for j in range(self.T):
                    out[j] = self.classifiers[j](entrain_graphs[i][j].x,entrain_graphs[i][j].edge_index)

                loss+=criterion(out, ys[i].to(torch.long))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=train_args["clip_max"])
            optimizer.step()
            
            train_acc = 0
            test_acc = 0
            val_acc = 0
            with torch.no_grad():
                out_train,_ = self.vote(self.train_mask)
                out_val,_ = self.vote(self.val_mask)
                
                train_acc = evaluate(out_train, self.labels[self.train_mask])
                val_acc = evaluate(out_val, self.labels[self.val_mask])


            print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss.item():.4f}")
            if val_acc == best_val_acc and train_acc>best_train_acc: # New best results
                print("Train improved")
                best_train_acc = train_acc
                best_epoch = epoch
                store_vote_checkpoint("robust_n/"+train_args["paper"], train_args["dataset"]+"/{}".format(self.Hasher.T), self.classifiers, train_acc, val_acc, test_acc)

            if val_acc > best_val_acc: # New best results
                print("Val improved")
                best_val_acc = val_acc
                best_train_acc = train_acc
                best_epoch = epoch
                store_vote_checkpoint("robust_n/"+train_args["paper"], train_args["dataset"]+"/{}".format(self.Hasher.T), self.classifiers, train_acc, val_acc, test_acc)

            if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
                break
        
     
    def test(self ):   
        out_test,M = self.vote(self.test_mask)
        test_acc = evaluate(out_test, self.labels[self.test_mask])
        return test_acc, M
        
    def vote(self, mask):
        G_test = len(self.graphs[mask])
        idxs = np.array([i for i in range(len(self.graphs))])
        test_id = idxs[mask]
        
        votes = torch.zeros((G_test,self.num_labels))
        M =torch.zeros(G_test)
        
        
        for j in range(self.T):
            self.classifiers[j].eval()
            
            out = torch.zeros((test_id.shape[0],self.num_labels)).to(self.device)
            for i in range(test_id.shape[0]):
                out[i] = self.classifiers[j](self.subgraphsX[j][test_id[i]],self.subgraphsE[j][test_id[i]])
            
            preds = out.argmax(dim=1)
            
            for i in range(preds.shape[0]):
                votes[i,preds[i]]+=1
        
        vote_label = votes.argmax(dim=1)
        for i in range(G_test):
            votes[i,vote_label[i]]=-votes[i,vote_label[i]]
        second_label = votes.argmax(dim=1)
        for i in range(G_test):
            votes[i,vote_label[i]]=-votes[i,vote_label[i]]
        for i in range(G_test):
            if vote_label[i]>second_label[i]:
                M[i] = (votes[i,vote_label[i]]-votes[i,second_label[i]]-1)//2
            else:
                M[i] = (votes[i,vote_label[i]]-votes[i,second_label[i]])//2
                
        return votes, M
       