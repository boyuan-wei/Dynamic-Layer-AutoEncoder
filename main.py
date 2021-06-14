import torch
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.nn import Module, Parameter, Linear
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import math

class Cat_Layer(Module):
    def __init__(self,num):
        super(Cat_Layer, self).__init__()
        self.weight=Parameter(torch.FloatTensor(num,1))
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self,input=[],func=lambda x:torch.sum(x,dim=0)):
        ls=[]
        for i in range(len(input)):
            ls.append(input[i].reshape(1,-1))
        return func(torch.cat(ls,dim=0)).reshape(input[0].size())
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro
class AE(Module):
    def __init__(self,n_input,n_cluster,layer=3,dense=False):
        super(AE, self).__init__()
        self.num=math.ceil((n_input-n_cluster)/layer)
        self.dense=dense
        self.outdim=[]
        self.layer=layer
        n_enc_in = n_input
        n_dec_in = 0
        for i in range(layer-1):
            if(dense):
                n_enc_in+=n_input-i*self.num
                if (i == 0):
                    self.add_module("ENC{}".format(i + 1), Linear(n_input, n_input - (i + 1) * self.num))
                    n_enc_in = n_input
                else:
                    self.add_module("ENC{}".format(i + 1), Linear(n_enc_in - n_input, n_input - (i + 1) * self.num))
            else:
                n_enc_in -=i*self.num
                self.add_module("ENC{}".format(i + 1),Linear(n_input - i * self.num, n_input - (i + 1) * self.num))
            self.outdim.append(n_input - (i + 1) * self.num)
        # construct z layer
        if(self.dense):
            self.add_module("Z LAYER",Linear(n_enc_in-(layer-1)*self.num,n_cluster))
        else:
            self.add_module("Z LAYER", Linear(n_input-(layer-1)*self.num,n_cluster))
        self.outdim.append(n_cluster)
        # construct decoder
        for i in range(layer - 1):
            if (dense):
                n_dec_in += n_cluster + i * self.num
                self.add_module("DEC{}".format(i + 1), Linear(n_dec_in, n_cluster + (i + 1) * self.num))
            else:
                n_dec_in+=i * self.num
                self.add_module("DEC{}".format(i + 1),Linear(n_cluster + i * self.num, n_cluster + (i + 1) * self.num))
            self.outdim.append(n_cluster + (i + 1) * self.num)
        # construct predict layer
        if(self.dense):
            self.add_module("PRED LAYER",Linear(n_dec_in+n_cluster+(layer-1)*self.num,n_input))
        else:
            self.add_module("PRED LAYER", Linear(n_cluster+(layer-1)*self.num,n_input))
        self.outdim.append(n_input)

    def forward(self, X):
        h=X
        hz=None
        if(self.dense):
            i=0
            for module in self.children():
                if(i<self.layer):
                    if(i==0):
                        h=F.relu(module(h))
                    else:
                        if(i!=self.layer-1):
                            h=torch.cat([h,F.relu(module(h))],dim=1)
                        else:
                            h = F.relu(module(h))
                            hz=h
                else:
                    if(i==self.layer*2-1):
                        h=F.relu(module(h))
                    else:
                        h = torch.cat([h, F.relu(module(h))], dim=1)
                i+=1
        else:
            i=0
            for module in self.children():
                h = F.relu(module(h))
                i+=1
                if (i == self.layer):
                    hz = h
        return hz,h

class Data_Loader:
    directory={}
    def __init__(self,mode=0):
        self.mode=mode
    def load_direct(self,datasetname,directory=".",xsuffix=".txt",ysuffix="_label.txt"):
            self.directory[datasetname]=["{}/{}{}".format(directory,datasetname,xsuffix),"{}/{}{}".format(directory,datasetname,ysuffix)]
    def load_data(self,datasetname):
        xdirectory=self.directory[datasetname][0]
        X=np.loadtxt(xdirectory,dtype=float)
        ydirectory=self.directory[datasetname][1]
        Y=np.loadtxt(ydirectory,dtype=int)
        return torch.from_numpy(np.array(X)).to(torch.float32),torch.from_numpy(np.array(Y))
def train_AE(model,x,y,epoch=100):
    ae=model
    optimizer=Adam(ae.parameters(),lr=1e-3)
    for i in range(epoch):
        Z,PRED=ae(x)
        loss=F.mse_loss(PRED,x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch:{} loss={} ".format(i,loss))
        if(i%5==6):
            with torch.no_grad():
                Z,xpred=ae(x)
                kmeans = KMeans(n_clusters=3, n_init=5)
                pred = kmeans.fit_predict(Z.data.numpy())
                acc,f1=cluster_acc(y.numpy(),pred)
                print("ACC:{} F1:{}".format(acc,f1))
    torch.save(ae.state_dict(), "AE_weights.pkl")
if(__name__=="__main__"):
    data_loader = Data_Loader()
    data_loader.load_direct(datasetname="acm", directory="./data")
    x, y = data_loader.load_data("acm")
    model=AE(n_input=x.shape[1],n_cluster=50,layer=5,dense=True)
    print(model)
    train_AE(model=model,x=x,y=y)
    with torch.no_grad():
        Z, xpred = model(x)
        kmeans = KMeans(n_clusters=3, n_init=5)
        pred = kmeans.fit_predict(Z.data.numpy())
        acc, f1 = cluster_acc(y.numpy(), pred)
        print("ACC:{} F1:{}".format(acc, f1))