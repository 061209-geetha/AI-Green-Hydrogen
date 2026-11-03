"""Expanded forecasting comparator suite.
All comparators use IDENTICAL chrono splits and training-only fitted scalers.
Small neural models use PyTorch. Hyperparameter search is left configurable;
these reference comparators are not claims of precise original-paper settings.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from .forecast import metrics


def standard_models(data,masks,include_slow=False):
    x=np.hstack([data['x'].reshape(len(data['x']),-1),data['s']]);yt=data['y'];a=masks['train'];b=masks['test']
    pool={
       'Ridge':Ridge(alpha=10),
       'Lasso':Lasso(alpha=.005,max_iter=1500),
       'ElasticNet':ElasticNet(alpha=.01,l1_ratio=.5,max_iter=1500),
       'RandomForest':RandomForestRegressor(n_estimators=120,max_depth=15,n_jobs=-1,random_state=42),
       'ExtraTrees':ExtraTreesRegressor(n_estimators=120,max_depth=15,n_jobs=-1,random_state=42),
       'GBDT':HistGradientBoostingRegressor(max_iter=250,max_depth=6,learning_rate=.05,random_state=42),
       'KNN':KNeighborsRegressor(n_neighbors=8,weights='distance'),
       'SVR':SVR(C=5.,epsilon=.12),
       'MLP':MLPRegressor(hidden_layer_sizes=(64,32),alpha=.005,max_iter=120,early_stopping=True,random_state=42)
    }
    try:
        import xgboost as xgb
        pool['XGBoost']=xgb.XGBRegressor(n_estimators=250,learning_rate=.05,max_depth=6,subsample=.8,
                                        colsample_bytree=.9,random_state=42,n_jobs=4)
    except ImportError:pass
    if not include_slow:
        pool={k:v for k,v in pool.items() if k in ('Ridge','Lasso','ElasticNet','RandomForest','ExtraTrees','GBDT','KNN')}
    results={}
    for name,model in pool.items():
        model.fit(x[a],yt[a]);p=model.predict(x[b])
        results[name]=data['scaler_y'].inverse_transform(np.asarray(p).reshape(-1,1)).reshape(-1)
    # Temporal naive baselines are expressed in physical units, not target-scaled units.
    last=(data['x'][b,-1,0]*data['scaler_x'].scale_[0]+data['scaler_x'].mean_[0])
    seasonal=(data['x'][b,-7,0]*data['scaler_x'].scale_[0]+data['scaler_x'].mean_[0])
    results['Persistence']=last;results['SeasonalNaive7']=seasonal
    return results


def neural_models(data,masks,epochs=15,batch_size=64,seed=42):
    """GRU, TCN, CNN-LSTM and Transformer next-day comparators (optional)."""
    import copy
    import torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    torch.manual_seed(seed);torch.set_num_threads(min(4,torch.get_num_threads()))
    nseq=data['x'].shape[-1];ns=data['s'].shape[-1]
    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__();self.rnn=nn.GRU(nseq,48,batch_first=True)
            self.out=nn.Sequential(nn.Linear(48+ns,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,x,s):a,_=self.rnn(x);return self.out(torch.cat((a[:,-1],s),1)).flatten()
    class TCNNet(nn.Module):
        def __init__(self):
            super().__init__();self.cnn=nn.Sequential(nn.Conv1d(nseq,48,3,padding=2,dilation=2),nn.ReLU(),
                  nn.Conv1d(48,48,3,padding=4,dilation=4),nn.ReLU())
            self.out=nn.Sequential(nn.Linear(48+ns,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,x,s):
            a=self.cnn(x.transpose(1,2))[:,:,-1]
            return self.out(torch.cat((a,s),1)).flatten()
    class CNNLSTMNet(nn.Module):
        def __init__(self):
            super().__init__();self.conv=nn.Conv1d(nseq,32,3,padding=1);self.rnn=nn.LSTM(32,48,batch_first=True)
            self.out=nn.Sequential(nn.Linear(48+ns,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,x,s):
            a=torch.relu(self.conv(x.transpose(1,2))).transpose(1,2);b,_=self.rnn(a)
            return self.out(torch.cat((b[:,-1],s),1)).flatten()
    class TransformerNet(nn.Module):
        def __init__(self):
            super().__init__();self.proj=nn.Linear(nseq,48)
            layer=nn.TransformerEncoderLayer(d_model=48,nhead=4,dim_feedforward=96,dropout=.1,batch_first=True)
            self.enc=nn.TransformerEncoder(layer,num_layers=2)
            self.pos=nn.Parameter(torch.zeros(1,data['x'].shape[1],48))
            self.out=nn.Sequential(nn.Linear(48+ns,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,x,s):
            a=self.enc(self.proj(x)+self.pos)
            return self.out(torch.cat((a[:,-1],s),1)).flatten()
    result={}
    for name,make in [('GRU',GRUNet),('TCN',TCNNet),('CNNLSTM',CNNLSTMNet),('Transformer',TransformerNet)]:
        model=make();opt=torch.optim.Adam(model.parameters(),lr=.001);loss_fn=nn.MSELoss()
        a=masks['train'];v=masks['val'];b=masks['test']
        dl=DataLoader(TensorDataset(torch.from_numpy(data['x'][a]),torch.from_numpy(data['s'][a]),torch.from_numpy(data['y'][a])),batch_size=batch_size,shuffle=True)
        xval=torch.from_numpy(data['x'][v]);sval=torch.from_numpy(data['s'][v]);yval=torch.from_numpy(data['y'][v]);
        best=float('inf');weights=None;stagnation=0
        for epoch in range(epochs):
            model.train()
            for x,s,y in dl:
                opt.zero_grad();l=loss_fn(model(x,s),y);l.backward();opt.step()
            model.eval()
            with torch.no_grad():score=loss_fn(model(xval,sval),yval).item()
            if score<best-1e-6:best=score;weights=copy.deepcopy(model.state_dict());stagnation=0
            else:stagnation+=1
            if stagnation>=5:break
        if weights:model.load_state_dict(weights)
        model.eval()
        with torch.no_grad():
            x=torch.from_numpy(data['x'][b]);s=torch.from_numpy(data['s'][b]);raw=np.concatenate([model(x[k:k+1024],s[k:k+1024]).numpy() for k in range(0,len(x),1024)])
        result[name]=data['scaler_y'].inverse_transform(raw.reshape(-1,1)).reshape(-1)
    return result
