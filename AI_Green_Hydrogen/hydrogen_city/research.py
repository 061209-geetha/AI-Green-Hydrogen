"""Statistical tests, plotting, optional benchmark families and ablations."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from .forecast import metrics

def diebold_mariano(y,forecast_a,forecast_b,max_lag=None):
    """DM test with Newey-West long-run variance and normal approximation.
    Both forecasts must refer to matched daily observations; output labels explicit.
    """
    y=np.asarray(y,float);a=np.asarray(forecast_a,float);b=np.asarray(forecast_b,float)
    d=(y-a)**2-(y-b)**2;n=len(d)
    if n<5:raise ValueError('DM requires at least five matched observations.')
    lag=max_lag if max_lag is not None else int(np.floor(n**(1/3)))
    centered=d-d.mean();variance=np.dot(centered,centered)/n
    for k in range(1,min(lag,n-1)+1):
        gamma=np.dot(centered[:-k],centered[k:])/n
        variance+=2*(1-k/(lag+1))*gamma
    if variance<=0:return {'statistic':0.,'p_value':1.,'n':n}
    statistic=float(d.mean()/np.sqrt(variance/n))
    return {'statistic':statistic,'p_value':float(2*stats.norm.sf(abs(statistic))),'n':n}

def holm_adjust(pvalues):
    p=np.asarray(pvalues,float);idx=np.argsort(p);out=np.zeros(len(p));last=0
    for rank,i in enumerate(idx):
        last=max(last,min(1.,(len(p)-rank)*p[i]));out[i]=last
    return out.tolist()

def bootstrap_ci(values,seed=42,n_boot=500,confidence=.95):
    v=np.asarray(values,float);rng=np.random.default_rng(seed)
    if len(v)<2:return [float(v.mean()),float(v.mean())]
    sampled=rng.choice(v,size=(n_boot,len(v)),replace=True).mean(axis=1)
    return np.quantile(sampled,[(1-confidence)/2,(1+confidence)/2]).tolist()

def daily_aggregate(pred_df):
    return pred_df.groupby('date',as_index=False)[['observed_kwh','predicted_kwh']].sum()

def plotting(predictions,history,scenario,output):
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def save(fig,name):fig.tight_layout();fig.savefig(output/name,dpi=600);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.scatter(predictions.observed_kwh,predictions.predicted_kwh,s=5,alpha=.25)
    lo=min(predictions.observed_kwh.min(),predictions.predicted_kwh.min());hi=max(predictions.observed_kwh.max(),predictions.predicted_kwh.max())
    ax.plot([lo,hi],[lo,hi],linestyle='--',color='black');ax.set(xlabel='Observed H₂-equivalent demand (kWh/day)',ylabel='Predicted demand (kWh/day)');save(fig,'forecast_parity.png')
    df=daily_aggregate(predictions)
    fig,ax=plt.subplots(figsize=(9,4));ax.plot(df.date,df.observed_kwh,label='Observed');ax.plot(df.date,df.predicted_kwh,label='Predicted')
    ax.set(ylabel='Aggregate H₂-equivalent demand (kWh/day)',xlabel='Date');ax.legend();save(fig,'forecast_timeseries.png')
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(predictions.predicted_kwh-predictions.observed_kwh,bins=35)
    ax.set(xlabel='Forecast residual (kWh/day)',ylabel='Number of predictions');save(fig,'forecast_residuals.png')
    if history is not None and len(history):
        df=pd.DataFrame(history);fig,ax=plt.subplots(figsize=(7,4));ax.plot(df.epoch,df.train_mse,label='Train MSE');ax.plot(df.epoch,df.val_mse,label='Validation MSE')
        ax.set(xlabel='Epoch',ylabel='Standardised MSE');ax.legend();save(fig,'training_history.png')
    sf=scenario[(scenario.h2_intensity_kgco2e_kg==.75)].sort_values('fraction')
    if len(sf):
        fig,ax=plt.subplots(figsize=(7,4));ax.bar([f'{x:.0%}' for x in sf.fraction],sf.avoided_tco2e_yr)
        ax.set(xlabel='Hydrogen substitution',ylabel='Avoided emissions (t CO₂e/year)');save(fig,'carbon_scenarios.png')
    sf=scenario[scenario.fraction==1].sort_values('h2_intensity_kgco2e_kg')
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(sf.h2_intensity_kgco2e_kg,sf.avoided_tco2e_yr,marker='o');ax.axhline(0,color='black',linestyle='--')
    ax.set(xlabel='Hydrogen carbon intensity (kg CO₂e/kg H₂)',ylabel='Avoided emissions (t CO₂e/year)');save(fig,'carbon_intensity_sensitivity.png')
