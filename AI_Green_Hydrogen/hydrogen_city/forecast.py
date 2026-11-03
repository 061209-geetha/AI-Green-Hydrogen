"""Chronological, leakage-safe one-day-ahead hydrogen forecasting.
Torch implementation with locked 128/64 LSTM; optional TF/Keras faithful variant.
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import Ridge
from .common import safe_mape,seed_all

SEQ_COLS=['h2_kwh','hdd','temperature_c','day_sin','day_cos','holiday']
STATIC_COLS=['relative_compactness','surface_area','wall_area','roof_area','overall_height','orientation','glazing_area','glazing_distribution']

def make_windows(data,lookback=24):
    """Sort per-household before constructing lag-only inputs and target dates."""
    seq=[]; stat=[]; cats=[]; ys=[]; dates=[]; ids=[]
    for h,g in data.sort_values(['household','date']).groupby('household',sort=True):
        a=g[SEQ_COLS].to_numpy(float)
        b=g[STATIC_COLS].to_numpy(float)
        y=g.h2_kwh.to_numpy(float)
        dt=pd.to_datetime(g.date).to_numpy()
        ac=g.acorn.astype(str).to_numpy()
        for t in range(lookback,len(g)):
            # Only genuine contiguous preceding dates are eligible.
            if (dt[t]-dt[t-lookback])/np.timedelta64(1,'D')!=lookback: continue
            seq.append(a[t-lookback:t]); stat.append(b[t]); cats.append(ac[t]); ys.append(y[t]); dates.append(dt[t]);ids.append(h)
    if not ys: raise ValueError('No continuous sequences; insufficient data or gaps.')
    return {'Xseq':np.asarray(seq,dtype=np.float32),'Xstatic':np.asarray(stat,dtype=np.float32),
            'acorn':np.asarray(cats),'y':np.asarray(ys,dtype=np.float32),
            'date':pd.to_datetime(dates),'household':np.asarray(ids)}

def partitions(w,val_fraction=.15,test_fraction=.15):
    """Global chronological boundaries; validation and test labels never train."""
    dates=np.sort(pd.DatetimeIndex(w['date']).unique())
    a=int(len(dates)*(1-val_fraction-test_fraction));b=int(len(dates)*(1-test_fraction))
    if a<1 or b<=a or b>=len(dates):raise ValueError('Not enough unique days for 70/15/15 partition.')
    mask={'train':np.asarray(w['date']<dates[a]),'val':np.asarray((w['date']>=dates[a])&(w['date']<dates[b])),
          'test':np.asarray(w['date']>=dates[b])}
    return mask,{'train_end':str(pd.Timestamp(dates[a-1]).date()),'val_start':str(pd.Timestamp(dates[a]).date()),'val_end':str(pd.Timestamp(dates[b-1]).date()),'test_start':str(pd.Timestamp(dates[b]).date()),'test_end':str(pd.Timestamp(dates[-1]).date())}

def standardize(w,masks):
    x=w['Xseq'].copy();s=w['Xstatic'].copy(); y=w['y'].copy()
    train=masks['train']
    sx=StandardScaler().fit(x[train].reshape(-1,x.shape[-1]))
    ss=StandardScaler().fit(s[train]);sy=StandardScaler().fit(y[train,None])
    x=sx.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape).astype(np.float32)
    s=ss.transform(s).astype(np.float32)
    onehot=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    onehot.fit(w['acorn'][train,None])
    cats=onehot.transform(w['acorn'][:,None]).astype(np.float32)
    s=np.column_stack([s,cats]).astype(np.float32)
    y=sy.transform(y[:,None]).reshape(-1).astype(np.float32)
    return {'x':x,'s':s,'y':y,'raw_y':w['y'],'dates':w['date'],'households':w['household'],
            'scaler_y':sy,'scaler_x':sx,'scaler_static':ss,'encoder_acorn':onehot}

def metrics(y,yhat):
    y=np.asarray(y,dtype=float);p=np.asarray(yhat,dtype=float)
    mae=float(mean_absolute_error(y,p));rmse=float(np.sqrt(mean_squared_error(y,p)))
    mape=safe_mape(y,p)
    return {'mae_kwh_day':mae,'rmse_kwh_day':rmse,'r2':float(r2_score(y,p)) if len(y)>1 else None,
            'mape_percent':mape, 'mape_accuracy_percent':100-mape if np.isfinite(mape) else None,'n':len(y)}

def _get_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError('PyTorch is required. pip install -r requirements.txt') from e

class TorchForecaster:
    """Two-stage stacked LSTM. Recurrent dropout for exact Keras parity requires --backend tensorflow."""
    def __init__(self,n_seq,n_static,cfg):
        import torch
        from torch import nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1=nn.LSTM(n_seq,int(cfg.get('hidden1',128)),batch_first=True)
                self.l2=nn.LSTM(int(cfg.get('hidden1',128)),int(cfg.get('hidden2',64)),batch_first=True)
                self.drop=nn.Dropout(float(cfg.get('dropout',.2)))
                self.head=nn.Sequential(nn.Linear(int(cfg.get('hidden2',64))+n_static,32),nn.ReLU(),nn.Linear(32,1))
            def forward(self,x,s):
                a,_=self.l1(x);b,_=self.l2(a)
                return self.head(self.drop(torch.cat([b[:,-1,:],s],dim=1))).view(-1)
        self.model=Net()
        self.cfg=cfg
    def fit(self,data,masks,epochs=None,device='auto',verbose=False):
        import torch
        from torch.utils.data import TensorDataset,DataLoader
        torch.set_num_threads(min(4,torch.get_num_threads()))
        dev='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
        self.model.to(dev)
        cfg=self.cfg;lr=float(cfg.get('learning_rate',.001))
        train=masks['train'];val=masks['val']
        def ds(m): return TensorDataset(torch.as_tensor(data['x'][m]),torch.as_tensor(data['s'][m]),torch.as_tensor(data['y'][m]))
        dl=DataLoader(ds(train),batch_size=int(cfg.get('batch_size',32)),shuffle=True)
        vl=DataLoader(ds(val),batch_size=min(512,max(1,int(val.sum()))),shuffle=False)
        opt=torch.optim.Adam(self.model.parameters(),lr=lr)
        sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=int(cfg.get('lr_patience',5)))
        loss_fn=torch.nn.MSELoss()
        best=float('inf');best_ep=0;weights=None;hist=[]
        for epoch in range(1,int(epochs or cfg.get('epochs',100))+1):
            self.model.train();tr=[]
            for x,s,y in dl:
                x=x.to(dev);s=s.to(dev);y=y.to(dev)
                opt.zero_grad();pred=self.model(x,s);loss=loss_fn(pred,y);loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0);opt.step();tr.append(loss.item())
            self.model.eval();vs=[]
            with torch.no_grad():
                for x,s,y in vl: vs.append((loss_fn(self.model(x.to(dev),s.to(dev)),y.to(dev)).item()))
            v=float(np.mean(vs));t=float(np.mean(tr));hist.append({'epoch':epoch,'train_mse':t,'val_mse':v})
            sch.step(v)
            if v<best-1e-6:
                best=v;best_ep=epoch;weights=copy.deepcopy(self.model.state_dict())
            if verbose:print(f'epoch {epoch:03d}: train={t:.5f} val={v:.5f}')
            if epoch-best_ep>=int(cfg.get('patience',10)):break
        if weights is not None:self.model.load_state_dict(weights)
        self.device=dev;self.history=hist;self.best_epoch=best_ep
        return hist
    def predict(self,x,s,batch=1024):
        import torch
        self.model.eval();out=[]
        with torch.no_grad():
            for k in range(0,len(x),batch):
                a=torch.as_tensor(x[k:k+batch],device=self.device)
                b=torch.as_tensor(s[k:k+batch],device=self.device)
                out.extend(self.model(a,b).cpu().numpy())
        return np.asarray(out,dtype=float)
    def save(self,path):
        import torch
        torch.save({'state_dict':self.model.state_dict(),'n_seq':self.model.l1.input_size,
                    'n_static':self.model.head[0].in_features-self.model.l2.hidden_size,
                    'config':self.cfg,'best_epoch':self.best_epoch},path)


def train_forecaster(data,masks,cfg,epochs=None,device='auto',backend='torch',verbose=False):
    if backend=='tensorflow':
        try:
            import tensorflow as tf
        except ImportError as e: raise ImportError('Install TensorFlow to use --backend tensorflow') from e
        i1=tf.keras.Input((data['x'].shape[1],data['x'].shape[2]));i2=tf.keras.Input((data['s'].shape[1],))
        a=tf.keras.layers.LSTM(cfg['hidden1'],return_sequences=True,recurrent_dropout=cfg.get('recurrent_dropout',.1))(i1)
        a=tf.keras.layers.LSTM(cfg['hidden2'],recurrent_dropout=cfg.get('recurrent_dropout',.1))(a)
        a=tf.keras.layers.Concatenate()([a,i2]);a=tf.keras.layers.Dropout(cfg.get('dropout',.2))(a)
        a=tf.keras.layers.Dense(32,activation='relu')(a);out=tf.keras.layers.Dense(1)(a)
        model=tf.keras.Model([i1,i2],out)
        model.compile(optimizer=tf.keras.optimizers.Adam(cfg['learning_rate']),loss='mse')
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=cfg['patience'],restore_best_weights=True),
                   tf.keras.callbacks.ReduceLROnPlateau(patience=cfg['lr_patience'],factor=.5)]
        h=model.fit([data['x'][masks['train']],data['s'][masks['train']]],data['y'][masks['train']],
                    validation_data=([data['x'][masks['val']],data['s'][masks['val']]],data['y'][masks['val']]),
                    epochs=epochs or cfg['epochs'],batch_size=cfg['batch_size'],callbacks=callbacks,verbose=int(verbose))
        return model,[{'epoch':i+1,'train_mse':a,'val_mse':b} for i,(a,b) in enumerate(zip(h.history['loss'],h.history['val_loss']))]
    model=TorchForecaster(data['x'].shape[-1],data['s'].shape[-1],cfg)
    hist=model.fit(data,masks,epochs,device,verbose)
    return model,hist

def predict_forecaster(model,data,mask,backend='torch'):
    if backend=='tensorflow':pred=model.predict([data['x'][mask],data['s'][mask]],verbose=0).reshape(-1)
    else:pred=model.predict(data['x'][mask],data['s'][mask])
    return data['scaler_y'].inverse_transform(pred[:,None]).reshape(-1)

def benchmark_predictions(data,masks):
    """Three strong, executable tabular comparisons; others are optional in research.py."""
    x=np.hstack([data['x'].reshape(len(data['x']),-1),data['s']]);yt=data['y'];a=masks['train'];b=masks['test']
    models={'Ridge':Ridge(alpha=10),
            'RandomForest':RandomForestRegressor(n_estimators=80,max_depth=12,n_jobs=-1,random_state=42),
            'GBDT':HistGradientBoostingRegressor(max_iter=250,learning_rate=.05,max_depth=6,random_state=42)}
    out={}
    for name,model in models.items():
        model.fit(x[a],yt[a]);p=model.predict(x[b]);out[name]=data['scaler_y'].inverse_transform(p[:,None]).reshape(-1)
    naive=data['x'][b,-1,0] # standardized using training-only statistics
    out['Persistence']=data['scaler_y'].inverse_transform(naive[:,None]).reshape(-1) if False else (
        naive * data['scaler_x'].scale_[0] + data['scaler_x'].mean_[0])
    return out
