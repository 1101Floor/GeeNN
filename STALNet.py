import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch.nn.functional as F    
from math import sqrt
import networkx as nx
import math
from tqdm import tqdm
import random
from joblib import Parallel, delayed
from tqdm import trange
import warnings
import itertools
from math import sqrt
import os
from torch.utils.data import Dataset,DataLoader
import random  
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
import torch.nn.functional as F

warnings.filterwarnings('ignore')
data_1 =df = pd.read_csv('data.csv').iloc[:,1:5]
data_in = data_1.iloc[:,[0,1,2,3]]
data_out = data_1.iloc[:,[0,1]]
print('Before Processing\n', data_in.head(6))
print('Before Processing\n', data_out.head(6))

def process_dataframes(data_in, data_out, win, output_filename):
    L = []
    for i in range(len(data_in)):
        L.append(i)
        
    x1 = []
    pbar = tqdm(total=len(L)) 
    for i in range(len(L)):
        pbar.update(1)
        for j in win:
            x1.append([c for c in range(i, i + j + 1, 1)])
    pbar.close()

    y = []
    pbar = tqdm(total=len(x1)) 
    for i in x1:
        pbar.update(1)
        if len(i) >= 2:
            y.append((x1.index(i), i[0], i[-1]))
        elif len(i) == 1:
            y.append((x1.index(i), i[0], i[0]))
    pbar.close()
    
    z = []
    pbar = tqdm(total=len(y)) 
    for i in y:
        pbar.update(1)
        for j in y:
            if i[2] + 1 == j[1]:
                z.append((i[0], j[0]))
    pbar.close()
    
    data_input = []
    data_output = []
    pbar = tqdm(total=len(y)) 
    for i in y:
        pbar.update(1)
        data_input.append([
            np.mean(data_in.iloc[i[1]:i[2] + 1, 0:1].values),
            np.mean(data_in.iloc[i[1]:i[2] + 1, 1:2].values),
            np.mean(data_in.iloc[i[1]:i[2] + 1, 2:3].values),
            np.mean(data_in.iloc[i[1]:i[2] + 1, 3:4].values)
        ])
    pbar.close()
    
    for i in y:
        data_output.append([np.mean(data_out.iloc[i[1]:i[2] + 1, 0:1].values),
                            np.mean(data_out.iloc[i[1]:i[2] + 1, 1:2].values),
                           ])
        
    data_input = pd.DataFrame(data_input)
    data_output = pd.DataFrame(data_output)
    
    np.savetxt(output_filename, z, fmt='%d')
    
    G = nx.read_edgelist(output_filename, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    return data_input, data_output, G
def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
class RandomWalker:
    def __init__(self, G):

        self.G = G
            
    def deepwalk_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]  
            cur_nbrs = list(self.G.neighbors(cur)) 
            if len(cur_nbrs) > 0:     
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk
 

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))
        
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deepwalk_walk(
                    walk_length=walk_length, start_node=v))
        
        return walks
    
    
def simulate_random_walks(G, data_input, data_output, num_walks=1, walk_length=30, Len_his = 20,Len_fut = 10):
    # Simulate random walks
    rw = RandomWalker(G)
    sentences = rw.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=1, verbose=1)
    
    L = [i for i in sentences if len(i) == walk_length]

    x_in, y_out = [], []
    ppp1, ppp2 = [], []

    for i in L:
        print(i)
        x = []
        y = []
        for j in i:
            x.append(data_input[eval(j)])
            y.append(data_output[eval(j)])
        x_in.append(x)
        y_out.append(y)


    lstm_input, lstm_output = [], []
    x_in = np.array(x_in)
    y_out = np.array(y_out)

    for i in range(len(x_in)):
        lstm_input.append(x_in[i][0:Len_his])  
        lstm_output.append(y_out[i][Len_his:walk_length])  
        
    lstm_input = np.array(lstm_input)
    lstm_output = np.array(lstm_output)
    return lstm_input, lstm_output


cut = int(len(np.array(data_in))*0.7)
data_in_1,data_out_1 = data_in.iloc[:cut,:],data_out.iloc[:cut,:]

data_input, data_output, G = process_dataframes(data_in, data_out, [0, 1, 2, 3, 4, 5], 'graph.txt')
mm_x = MinMaxScaler()
mm_y = MinMaxScaler()

data_input = mm_x.fit_transform(data_input)
data_output = mm_y.fit_transform(data_output)

lstm_input, lstm_output= simulate_random_walks(G, data_input, data_output, num_walks=1, walk_length=30, Len_his = 20,Len_fut = 10)

val_radio = 0.2
test_radio = 0.7

val_cut = int(len(lstm_input)*val_radio)

x_train,x_val, y_train, y_val= \
lstm_input[:-val_cut,:],lstm_input[-val_cut:,:],lstm_output[:-val_cut,:],lstm_output[-val_cut:,:]

#y_train = y_train.reshape(-1,5)
#y_val=y_val.reshape(-1,5)
test_cut = int(len(np.array(data_in))*test_radio)
test_input = np.array(data_in)[test_cut:]
test_output = np.array(data_out)[test_cut:]

test_input = mm_x.fit_transform(test_input)
test_output = mm_y.fit_transform(test_output)


def convert_data(test_input,test_output,Len_his = 20,Len_fut = 10):
    test_input = np.array(test_input)
    test_output = np.array(test_output)
   
    model_input = []
    model_output = []

    for i in range(len(test_input)-Len_his-Len_fut+1):
        model_input.append(test_input[i:i+Len_his,:])
        model_output.append(test_output[i+Len_his:i+Len_his+Len_fut,:])

    model_input=np.array(model_input)
    model_output=np.array(model_output)  
    
    #model_output = model_output.reshape(-1,5)

    return model_input, model_output
x_test, y_test = convert_data(test_input,test_output,Len_his = 20,Len_fut = 10)
print('x_train.shape',x_train.shape)
print('y_train.shape',y_train.shape)
print('x_val.shape',x_val.shape)
print('y_val.shape',y_val.shape)

print('x_test.shape',x_test.shape)
print('y_test.shape',y_test.shape)


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, index):

        output = [self.x[index], self.y[index]]
        return output
    def __len__(self):
        return self.x.shape[0]
device = torch.device("cuda")

dset_train = MyDataset(lstm_input,lstm_output)
dset_val = MyDataset(x_val,y_val)
dset_test = MyDataset(x_test,y_test)

loader_train = DataLoader(
        dset_train,
        batch_size=64, 
        shuffle =True,
        num_workers=0)

loader_val = DataLoader(
        dset_val,
        batch_size=64, 
        shuffle =True,
        num_workers=0)
loader_test = DataLoader(
        dset_test,
        batch_size=64, 
        shuffle =False,
        num_workers=0)

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return x
class Lon_TAU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Lon_TAU, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))  
        self.relu=torch.nn.ReLU()
    def forward(self, x):
        # x [b, 10, 3]
        #print(x.shape)
        scores = self.linear(x)  # [b, 10, hidden_dim]

        # 将 scores 应用于注意力机制
        attention_scores = self.relu(scores) 
        attention_weights = torch.softmax(attention_scores, dim=1)  

        weighted_sum = torch.bmm(attention_weights.permute(0, 2, 1), x)  # [b, hidden_dim, 3]
        
        return weighted_sum, attention_weights

class Lat_TAU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Lat_TAU, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim)) 
        self.relu=torch.nn.ReLU()
    def forward(self, x):
        # x [b, 10, 3]
        scores = self.linear(x) [b, 10, hidden_dim]

        attention_scores = self.relu(scores)  
        attention_weights = torch.softmax(attention_scores, dim=1)  

        weighted_sum = torch.bmm(attention_weights.permute(0, 2, 1), x)  # [b, hidden_dim, 3]
        
        return weighted_sum, attention_weights

class Lon_SAU(nn.Module):
    def __init__(self, input_dim):
        super(Lon_SAU, self).__init__()
        self.input_dim = input_dim
        self.attention_weights = nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, x):
        x_reshaped = x.view(-1, self.input_dim)  # [b * 10, 3]
        attn_scores = self.attention_weights(x_reshaped)  # [b * 10, 1]
        attn_scores = attn_scores.view(x.size(0), x.size(1), -1)  # [b, 10, 1]
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [b, 10, 1]

        output = x * attn_weights  # [b, 10, 3]
        
        return output

class Lat_SAU(nn.Module):
    def __init__(self, input_dim):
        super(Lat_SAU, self).__init__()
        self.input_dim = input_dim
        self.attention_weights = nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, x):
        x_reshaped = x.view(-1, self.input_dim)  # [b * 10, 3]
        attn_scores = self.attention_weights(x_reshaped)  # [b * 10, 1]
        attn_scores = attn_scores.view(x.size(0), x.size(1), -1)  # [b, 10, 1]

        attn_weights = F.softmax(attn_scores, dim=-1)  # [b, 10, 1]

        output = x * attn_weights  # [b, 10, 3]
        
        return output

    
class STALNet(nn.Module):
    def __init__(self):
        super(STATLSTM,self).__init__()
        
        # basic setting
        self.HST_LEN,self.FUT_LEN=20,10       
        self.enc_size=32       
        
        self.TAU_lon = Lon_TAU(input_dim=self.HST_LEN, hidden_dim=self.enc_size+3)
        self.TAU_lat = Lat_TAU(input_dim=self.HST_LEN, hidden_dim=self.enc_size+3)
        self.SAU_lon = Lon_SAU(input_dim=3)
        self.SAU_lat = Lat_SAU(input_dim=3)
        self.encoder = Encoder(self.HST_LEN, 3, self.enc_size).to(device)
        
        self.MLP_lon=nn.Linear(((self.enc_size+3)*2+3)*self.HST_LEN,self.FUT_LEN)
        self.MLP_lat=nn.Linear(((self.enc_size+3)*2+3)*self.HST_LEN,self.FUT_LEN)
        self.leaky_relu=torch.nn.LeakyReLU(0.1)
        self.relu=torch.nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self,x):
        #print(x.shape)  
        lon = x[:,:,[0,2,3]]
        lat = x[:,:,[1,2,3]]
        
        lon_SAU = self.SAU_lon(lon)
        lat_SAU = self.SAU_lat(lat)
        
        lon_SAU = lon_SAU + lon
        lat_SAU = lat_SAU + lat
        
        lon_enc = self.encoder(lon_SAU)
        lat_enc = self.encoder(lat_SAU)
        lon_enc = self.leaky_relu(lon_enc)
        lat_enc = self.leaky_relu(lat_enc)
        lon_enc = torch.cat([lon_enc,lon_SAU],axis = -1)
        lat_enc = torch.cat([lat_enc,lat_SAU],axis = -1)
        
        b,t,f = lon_enc.shape
        lon_enc_1 = lon_enc.reshape(b,f,t)
        lat_enc_1 = lat_enc.reshape(b,f,t)
                
        lon_TAU, _ = self.TAU_lon(lon_enc_1)
        lat_TAU, _ = self.TAU_lat(lat_enc_1)
        lon_TAU_1 = lon_TAU.reshape(b,self.HST_LEN,self.enc_size+3)
        lat_TAU_1 = lat_TAU.reshape(b,self.HST_LEN,self.enc_size+3)
        
        
        lon_c = torch.cat([lon,lon_enc,lon_TAU_1],axis = -1).reshape(b,-1)
        lat_c = torch.cat([lat,lat_enc,lat_TAU_1],axis = -1).reshape(b,-1)
        #print(lon_c.shape)
        lon_out = self.MLP_lon(lon_c).reshape(b,self.FUT_LEN,1)
        lat_out = self.MLP_lat(lat_c).reshape(b,self.FUT_LEN,1)
        
        x_out = torch.cat([lon_out,lat_out],axis = -1)
        #print(x_out.shape)
        return x_out
model = STALNet().to(device)

def adjust_learning_rate(epoch):

    lr = 0.001

    if epoch > 180:
        lr = lr / 2
    elif epoch > 150:
        lr = lr / 2
    elif epoch > 120:
        lr = lr / 2
    elif epoch > 90:
        lr = lr / 2
    elif epoch > 60:
        lr = lr / 2
    elif epoch > 30:
        lr = lr / 2

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
def test(model,dataloader):
    model.eval()
    total_loss = 0
    #loss_function = ExponentialL2Loss()
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for i,batch in enumerate(dataloader):
            batch = [tensor.cuda() for tensor in batch]
            x,y = batch  
            result = model(x.float())
            loss = loss_fn(result, y.float())
            total_loss += loss
    return (total_loss/len(dataloader))
def train(model,train_dataloader,test_dataloader,num_epoch):
    train_loss_coll = []
    val_loss_coll = []
    min_loss = 1000
    loss_fn = torch.nn.MSELoss()
    #loss_function = ExponentialL2Loss()
    for epoch in range(num_epoch):
        model = model.to(torch.float32)
        model.train()
        train_loss = 0

        for i,batch in enumerate(train_dataloader):
            batch = [tensor.cuda() for tensor in batch]
            x,y = batch 
            optimizer.zero_grad()
            
            result = model(x.float())
            loss = loss_fn(result, y.float())
            #loss = loss_fn(result, out.float())
            loss.backward()
            optimizer.step()
            train_loss+=loss

        adjust_learning_rate(epoch)
        train_loss = train_loss/len(train_dataloader)
        test_loss = test(model,test_dataloader)
        torch.save(model,'STALNet.pt')

        train_loss_coll.append(train_loss)
        val_loss_coll.append(test_loss)

        print("Epoch {}, Train loss {}, val loss {}".format(epoch,train_loss,test_loss))

    return train_loss_coll, val_loss_coll

optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.00001)
train_loss, val_loss = train(model, loader_train, loader_val,1)
epochs_range = range(len(train_loss))
plt.plot(epochs_range,[i.cpu().detach().numpy() for i in train_loss],label= "Train_loss")
plt.plot(epochs_range, [i.cpu().detach().numpy() for i in val_loss], label="Val_loss")
plt.legend(loc='upper right')
plt.title('Train and Val Loss')
plt.show()

def predict(model, test_dataloader):
    model.eval() 
    device = next(model.parameters()).device  
    tgts = []
    results = []
    
    with torch.no_grad():  
        for batch in test_dataloader:
            
            x,y= [x.to(device).float() for x in batch]
            
            result = model(x.float())
            tgts.extend(y.cpu().numpy())  
            results.extend(result.cpu().numpy())  
            
    tgts = torch.tensor(tgts) 
    results = torch.tensor(results)
    
    return tgts, results
tgts , predicts= predict(model,loader_test)

tgts1 = tgts.reshape(-1,2)
predicts1= predicts.reshape(-1,2)
tgts_1 = mm_y.inverse_transform(tgts1)
predicts_1 = mm_y.inverse_transform(predicts1)
tgts_1 =tgts_1.reshape(-1,10,2)
predicts_1 = predicts_1.reshape(-1,10,2)