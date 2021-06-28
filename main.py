import torch
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.nn import Module, Parameter, Linear
import torch.nn.functional as F
from torch.optim import Adam
import math
import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
import Data_Preprossing
topk = 10
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def construct_graph(features,fname,  method='heat'):
    name = 'graph/'+fname+'_KNN_graph_AE.txt'
    num = features.size(0)
    dist = None
    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind)

    f = open(name, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if vv != i:
                    counter += 1
                    f.write('{} {}\n'.format(i, vv))
    f.close()
def load_graph(dataset, k=False):
    if k:
        path = 'graph/{}_graph_AE.txt'.format(dataset)
    else:
        path = 'graph/{}_KNN_graph_AE.txt'.format(dataset)

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj
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
def train_AE(model,x,y,epoch=10,name="default",n_clusters=2,n_init=5):
    ae=model
    '''kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    pred = kmeans.fit_predict(x)
    acc, f1 = cluster_acc(y.numpy(), pred)
    print("k-means origin Acc={}".format(acc))'''
    optimizer=Adam(ae.parameters(),lr=1e-3)
    for i in range(epoch):
        Z,PRED=ae(x)
        '''kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        pred = kmeans.fit_predict(Z.data.numpy())
        acc, f1 = cluster_acc(y.numpy(), pred)
        # calculate degree of membership P'''
        '''q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - torch.tensor(kmeans.cluster_centers_), 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        weight = q ** 2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()'''
        aeloss = F.mse_loss(PRED, x)
        #pqloss=F.mse_loss(q,p)
        loss=aeloss/50
        #print("epoch{} K-Means ACC:{} F1:{} loss={} AEloss={} PQloss={}".format(i, acc, f1, loss,aeloss,pqloss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(i%10==0 and False):
            with torch.no_grad():
                Z,xpred=ae(x)
                kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
                pred = kmeans.fit_predict(Z.data.numpy())
                acc,f1=cluster_acc(y.numpy(),pred)
                print("ACC:{} F1:{}".format(acc,f1))
    torch.save(ae.state_dict(), "weight/{}_AE_weights.pkl".format(name))
    return loss
def pretrain_test():
    data_loader = Data_Loader()
    data_loader.load_direct(datasetname="acm", directory="./data")
    x, y = data_loader.load_data("acm")
    model=AE(n_input=x.shape[1],n_cluster=3,layer=4,dense=False)
    print(model)
    train_AE(model=model,x=x,y=y,epoch=100,n_clusters=3,n_init=5)
    with torch.no_grad():
        Z, xpred = model(x)
        kmeans = KMeans(n_clusters=3, n_init=5)
        pred = kmeans.fit_predict(Z.data.numpy())
        acc, f1 = cluster_acc(y.numpy(), pred)
        print("K-Means ACC:{} F1:{}".format(acc, f1))
        # calculate degree of membership P
        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - torch.tensor(kmeans.cluster_centers_), 2),2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        weight = q ** 2 / q.sum(0)
        p=(weight.t() / weight.sum(1)).t()
        construct_graph(p,"acm",method="heat")
        adj=load_graph("acm")
        spectral=SpectralClustering(n_clusters=3)
        result=spectral.fit_predict(adj.to_dense())
        acc, f1 = cluster_acc(y.numpy(), pred)
        print("Spectral Clustering ACC:{} F1:{}".format(acc, f1))
def train():
    data=np.array(Data_Preprossing.read_excel_xls("DATA/{}_{}.xls".format("000938","紫光股份")))
    x=[]
    y=[]
    for i in data[2:]:
        ls = []
        for j in i[:5]:
            try:
                ls.append(float(j))
            except:
                ls.append(0.0)
        x.append(ls)
        y.append(int(float(i[6])))
    model = AE(n_input=len(x[0]), n_cluster=4, layer=6, dense=True)
    print(model)
    batch_size=50
    for i in range(100):
        start = 0
        for k in range(len(x) // batch_size):
            X = x[start:start + batch_size]
            Y = y[start:start + batch_size]
            start += batch_size
            X = torch.from_numpy(np.array(X)).to(torch.float32)
            Y = torch.from_numpy(np.array(Y))
            loss=train_AE(model=model, x=X, y=Y, epoch=10, name="000938_紫光股份")
        print("epoch{} loss={}".format(i,loss))
    with torch.no_grad():
        x = torch.from_numpy(np.array(x)).to(torch.float32)
        y = torch.from_numpy(np.array(y))
        Z, xpred = model(x)
        kmeans = KMeans(n_clusters=2, n_init=4)
        pred = kmeans.fit_predict(Z.data.numpy())
        acc, f1 = cluster_acc(y.numpy(), pred)
        print("K-Means ACC:{} F1:{}".format(acc, f1))
        # calculate degree of membership P
        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - torch.tensor(kmeans.cluster_centers_), 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        weight = q ** 2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        construct_graph(p, "acm", method="heat")
        adj = load_graph("acm")
        spectral = SpectralClustering(n_clusters=2)
        result = spectral.fit_predict(adj.to_dense())
        acc, f1 = cluster_acc(y.numpy(), pred)
        print("Spectral Clustering ACC:{} F1:{}".format(acc, f1))
if(__name__=="__main__"):
    #pretrain_test()
    train()