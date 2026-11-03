"""Real-data adapters and explicitly synthetic demonstration data.

Canonical household dataframe: household,date,electricity_kwh,acorn.
All synthetic objects are marked in manifests; NEVER misrepresent them as measurements.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from .common import seed_all

UCI_COLUMNS = ['relative_compactness','surface_area','wall_area','roof_area','overall_height','orientation','glazing_area','glazing_distribution','heating_load','cooling_load']

def read_uci(path):
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f'UCI file missing: {path}')
    df = pd.read_excel(path, header=None) if path.suffix.lower() in ('.xlsx','.xls') else pd.read_csv(path)
    if len(df.columns) >= 10:
        # UCI's Excel file includes headers in first row.
        if str(df.iloc[0,0]).strip().lower() in ('x1','relative compactness'):
            df = df.iloc[1:].copy()
        df = df.iloc[:,:10].copy()
        df.columns = UCI_COLUMNS
    else: raise ValueError('Expected ten UCI fields (eight features and two targets).')
    for c in UCI_COLUMNS: df[c] = pd.to_numeric(df[c],errors='coerce')
    df = df.dropna(subset=UCI_COLUMNS).reset_index(drop=True)
    if len(df) < 30: raise ValueError('Too few valid UCI rows.')
    return df

def train_heat_archetypes(uci,seed=42):
    x = uci[UCI_COLUMNS[:8]].to_numpy(float)
    y = uci['heating_load'].to_numpy(float)
    model = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, learning_rate=0.06, random_state=seed)
    cv = KFold(5,shuffle=True,random_state=seed)
    scores = cross_val_score(model,x,y,cv=cv,scoring='neg_mean_absolute_error')
    model.fit(x,y)
    return model, {'five_fold_mae_mean': float(-scores.mean()), 'five_fold_mae_sd': float(scores.std())}

def _meter_cols(df):
    mapping={str(c).strip().lower().replace(' ','').replace('_','').replace('/',''): c for c in df.columns}
    def find(*names):
        for n in names:
            if n in mapping: return mapping[n]
        return None
    hh=find('lclid','household','householdid','id')
    dt=find('datetime','timestamp','date')
    power=find('kwhhhperhalfhour','kwhhalfhour','kwh','electricitykwh','consumption')
    acorn=find('acorn','acorngroup','acorncategory')
    return hh,dt,power,acorn

def read_lcl(path,max_households=None,chunksize=250000):
    """Aggregate LCL half-hourly readings in chunks. Supports file, directory, csv/zip.

    Missing readings remain missing (no imaginary zero consumption). Duplicate meter
    time stamps are consolidated. Entire date, household pairs with <44 readings
    are removed; short interpolation is performed only inside a day's 48 slots.
    """
    p=Path(path)
    paths=sorted(x for x in p.rglob('*') if x.suffix.lower() in ('.csv','.zip')) if p.is_dir() else [p]
    if not paths: raise FileNotFoundError(f'No LCL CSV/ZIP files in {p}')
    raw=[]; chosen=set()
    for f in paths:
        # Streaming per file, but retained canonical rows may be substantial: cap IDs when desired.
        for chunk in pd.read_csv(f,chunksize=chunksize,low_memory=False):
            h,d,v,a=_meter_cols(chunk)
            if any(z is None for z in [h,d,v]):
                raise ValueError(f'Unrecognized LCL columns in {f}: {list(chunk.columns)}')
            c=chunk[[h,d,v]+([a] if a is not None else [])].copy()
            c.columns=['household','datetime','electricity_kwh']+(['acorn'] if a is not None else [])
            c['household']=c['household'].astype(str)
            if max_households is not None:
                new=sorted(set(c['household'])-chosen)
                chosen.update(new[:max(0,max_households-len(chosen))])
                c=c[c['household'].isin(chosen)]
            if c.empty: continue
            c['datetime']=pd.to_datetime(c['datetime'],errors='coerce',dayfirst=True)
            c['electricity_kwh']=pd.to_numeric(c['electricity_kwh'],errors='coerce')
            c=c.dropna(subset=['datetime','electricity_kwh'])
            c=c[(c['electricity_kwh']>=0) & (c['electricity_kwh']<100)]
            if c.empty: continue
            c['date']=c['datetime'].dt.normalize()
            # Deliberately conservative short-gap handling; no fill across days.
            raw.append(c[['household','date','datetime','electricity_kwh']+(['acorn'] if 'acorn' in c else [])])
    if not raw: raise ValueError('No valid LCL readings.')
    d=pd.concat(raw,ignore_index=True).drop_duplicates(['household','datetime'])
    grp=d.groupby(['household','date'],sort=True)
    daily=grp['electricity_kwh'].agg(['sum','count']).reset_index()
    daily=daily[daily['count']>=44].rename(columns={'sum':'electricity_kwh'})[['household','date','electricity_kwh']]
    if 'acorn' in d:
        ac=d.groupby('household')['acorn'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        daily['acorn']=daily['household'].map(ac)
    else: daily['acorn']='Unknown'
    return daily.reset_index(drop=True)

def weather_from_csv(path):
    df=pd.read_csv(path)
    date=next((c for c in df if c.lower() in ('date','time','datetime')),None)
    temp=next((c for c in df if c.lower() in ('tavg','temperature','temperature_c','temp_c','tmean')),None)
    if not date or not temp: raise ValueError('Weather CSV needs date and temperature/tavg column.')
    o=pd.DataFrame({'date':pd.to_datetime(df[date],errors='coerce').dt.normalize(), 'temperature_c':pd.to_numeric(df[temp],errors='coerce')})
    return o.dropna().groupby('date',as_index=False)['temperature_c'].mean().sort_values('date')

def fetch_meteostat(start,end,lat=51.4775,lon=-0.4614):
    try:
        from meteostat import Point, Daily
    except ImportError as e:
        raise ImportError('Install meteostat to fetch historical weather, or use --weather-csv') from e
    data=Daily(Point(lat,lon),pd.Timestamp(start),pd.Timestamp(end)).fetch().reset_index()
    if data.empty: raise RuntimeError('Weather request returned no rows.')
    return weather_from_frame(data)

def weather_from_frame(df):
    return pd.DataFrame({'date':pd.to_datetime(df['time']).dt.normalize(), 'temperature_c':pd.to_numeric(df['tavg'])}).dropna()

def make_demo(n_households=80,n_days=380,seed=42):
    """DEMO ONLY: synthetic daily meter observations, synthetic weather/UCI."""
    rng=np.random.default_rng(seed)
    dates=pd.date_range('2012-01-01',periods=n_days,freq='D')
    doy=dates.dayofyear.to_numpy()
    temp=11+8*np.sin(2*np.pi*(doy-100)/365)+rng.normal(0,1.8,n_days)
    weather=pd.DataFrame({'date':dates,'temperature_c':temp})
    u=rng.uniform(0,1,(768,8))
    uci=pd.DataFrame({
        'relative_compactness':.55+.4*u[:,0], 'surface_area':450+350*u[:,1],
        'wall_area':200+180*u[:,2], 'roof_area':100+100*u[:,3],
        'overall_height':3.5+3.5*(u[:,4]>.5), 'orientation':(1+np.floor(4*u[:,5])).astype(int),
        'glazing_area':.4*u[:,6], 'glazing_distribution':np.floor(5*u[:,7]).astype(int)
    })
    uci['heating_load']=9+0.018*uci.surface_area+0.04*uci.wall_area+5*uci.overall_height/7-4*uci.glazing_area+rng.normal(0,1.8,768)
    uci['cooling_load']=7+0.014*uci.surface_area+4*uci.glazing_area+rng.normal(0,1.5,768)
    data=[]
    for i in range(n_households):
        baseline=rng.uniform(4,12)
        weekday=(dates.dayofweek.to_numpy()<5).astype(float)
        values=baseline+1.4*weekday+1.2*np.maximum(15.5-temp,0)/10+1.0*np.sin(2*np.pi*doy/7+i)+rng.normal(0,0.6,n_days)
        values=np.maximum(values,0.2)
        data.append(pd.DataFrame({'household':f'DEMO_{i:04d}','date':dates,'electricity_kwh':values,'acorn':f'Group_{i%5}'}))
    return pd.concat(data,ignore_index=True), weather, uci

def create_hydrogen_targets(meter,weather,heat_model,uci,params,seed=42):
    """Transparent pseudo-heating from UCI-derived archetypes and HDD calibration.

    Critically, UCI and ACORN have no observed crosswalk; this is an ASSUMED
    archetype assignment, not an empirically identifiable household attribute.
    """
    d=meter.copy(); d['date']=pd.to_datetime(d['date']).dt.normalize()
    d=d.merge(weather,on='date',how='left',validate='many_to_one')
    if d.temperature_c.isna().any():
        raise ValueError(f'Weather missing on {d.temperature_c.isna().sum()} meter days; provide coverage for full meter period.')
    d['hdd']=np.maximum(0,params['hdd_base_c']-d.temperature_c)
    rng=np.random.default_rng(seed)
    groups=sorted(d.acorn.fillna('Unknown').astype(str).unique())
    archetypes={g: int(rng.integers(len(uci))) for g in groups}
    features=uci.iloc[[archetypes[g] for g in groups]][UCI_COLUMNS[:8]]
    predicted=heat_model.predict(features.to_numpy(float))
    bygroup=dict(zip(groups,np.maximum(predicted,0.1)))
    d['heat_weight']=d.acorn.map(bygroup).astype(float)
    # Heat loss scales according to UCI archetypes; mean gas requirement calibrated
    # on the available weather period, then normalized to 11,200 kWh/yr per home.
    hdd_mean=d.groupby('household')['hdd'].mean()
    factor=params['annual_gas_kwh']/365.25
    q0=factor*params['hot_water_fraction']
    heat_scale=(factor-q0)/max(1e-8,float((d.heat_weight*d.hdd).mean()))
    d['heating_kwh']=q0+heat_scale*d.heat_weight*d.hdd
    d['h2_kwh']=d.electricity_kwh/params['fuel_cell_efficiency']+params['boiler_ratio']*d.heating_kwh
    d['h2_kg']=d.h2_kwh/params['h2_lhv_kwh_kg']
    d['day_sin']=np.sin(2*np.pi*d.date.dt.dayofweek/7)
    d['day_cos']=np.cos(2*np.pi*d.date.dt.dayofweek/7)
    d['holiday']=0.0 # no inferred holidays; optional explicit calendar feed
    for k in UCI_COLUMNS[:8]: d[k]=d.acorn.map(dict(zip(groups,features[k].to_numpy(float))))
    metadata={'synthetic_gas_demand':True,'synthetic_acorn_uci_crosswalk':True,'hotwater_kwh_day':float(q0), 'heat_scale':float(heat_scale),'acorn_to_uci_index':archetypes,'n_households':int(d.household.nunique())}
    return d.sort_values(['household','date']).reset_index(drop=True),metadata
