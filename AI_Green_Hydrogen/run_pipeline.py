#!/usr/bin/env python3
"""End-to-end hydrogen-city forecasting -> infrastructure -> carbon simulation.
Demo mode is synthetic and is NEVER evidence of manuscript results.
"""
from __future__ import annotations
import argparse,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

from hydrogen_city.common import config,seed_all,write_json,save_table,checksum
from hydrogen_city.data import make_demo,read_uci,read_lcl,train_heat_archetypes,create_hydrogen_targets,weather_from_csv,fetch_meteostat
from hydrogen_city.forecast import make_windows,partitions,standardize,train_forecaster,predict_forecaster,benchmark_predictions,metrics
from hydrogen_city.spatial import demo_world,osm_world,run_spatial,save_world_plot
from hydrogen_city.carbon import baseline,simulate,break_even_intensity
from hydrogen_city.research import plotting,diebold_mariano,holm_adjust,daily_aggregate

def cli():
    ap=argparse.ArgumentParser(description='Hydrogen urban planning: executable research pipeline')
    ap.add_argument('--mode',choices=['demo','real'],default='demo')
    ap.add_argument('--config',default=None)
    ap.add_argument('--output',default='outputs/demo')
    ap.add_argument('--epochs',type=int,default=None)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--n-households',type=int,default=60,help='Synthetic demo count; real processing cap via --max-real-households')
    ap.add_argument('--days',type=int,default=360)
    ap.add_argument('--lcl',help='Real LCL CSV, ZIP, or folder')
    ap.add_argument('--uci',help='Real UCI Energy Efficiency CSV or XLSX')
    ap.add_argument('--weather-csv',help='Daily CSV: date,temperature_c (or tavg)')
    ap.add_argument('--fetch-weather',action='store_true',help='Fetch Meteostat Heathrow history (requires network + meteostat)')
    ap.add_argument('--max-real-households',type=int,default=250)
    ap.add_argument('--bbox',nargs=4,type=float,metavar=('NORTH','SOUTH','EAST','WEST'),help='Bounding box for real OSM map')
    ap.add_argument('--backend',choices=['torch','tensorflow'],default='torch')
    ap.add_argument('--skip-benchmarks',action='store_true')
    ap.add_argument('--shap',action='store_true',help='Compute optional GradientExplainer SHAP on a bounded test subset')
    ap.add_argument('--full-benchmarks',action='store_true',help='Additional statistical/ML + four neural benchmarks (slow)')
    ap.add_argument('--benchmark-epochs',type=int,default=8)
    ap.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    ap.add_argument('--verbose',action='store_true')
    return ap.parse_args()

def main(args=None):
    if args is None:args=cli()
    start=time.monotonic();cfg=config(args.config);seed_all(args.seed)
    out=Path(args.output);(out/'tables').mkdir(parents=True,exist_ok=True)
    (out/'figures').mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    raw_files={}
    if args.mode=='demo':
        meter,weather,uci=make_demo(args.n_households,args.days,args.seed)
        world=demo_world(cfg['spatial']['n_zones'],cfg['spatial']['n_sites'],args.seed)
        input_type='SYNTHETIC DEMONSTRATION ONLY'
    else:
        if not args.lcl or not args.uci or not args.bbox:raise ValueError('Real mode requires --lcl, --uci and --bbox NORTH SOUTH EAST WEST')
        meter=read_lcl(args.lcl,max_households=args.max_real_households)
        uci=read_uci(args.uci)
        if args.weather_csv:weather=weather_from_csv(args.weather_csv)
        elif args.fetch_weather:weather=fetch_meteostat(meter.date.min(),meter.date.max())
        else:raise ValueError('Real mode needs --weather-csv or --fetch-weather.')
        world=osm_world(*args.bbox,n_zones=cfg['spatial']['n_zones'],n_sites=cfg['spatial']['n_sites'],seed=args.seed)
        input_type='MEASURED LCL ELECTRICITY, MODELLED GAS, PROVISIONAL OSM SITES'
        for name,path in [('uci',args.uci),('lcl',args.lcl),('weather',args.weather_csv)]:
            if path and Path(path).is_file():raw_files[name]={'path':str(path),'sha256':checksum(path)}
    heat,cv=train_heat_archetypes(uci,args.seed)
    target,conv=create_hydrogen_targets(meter,weather,heat,uci,cfg['hydrogen'],args.seed)
    if not target.empty:save_table(target.head(100000),out/'tables'/'hydrogen_targets_sample')
    w=make_windows(target,cfg['lookback']);masks,dates=partitions(w,cfg['forecast']['val_fraction'],cfg['forecast']['test_fraction'])
    scaled=standardize(w,masks)
    model,hist=train_forecaster(scaled,masks,cfg['forecast'],epochs=args.epochs,device=args.device,backend=args.backend,verbose=args.verbose)
    mask=masks['test'];pred=predict_forecaster(model,scaled,mask,args.backend)
    obs=w['y'][mask].astype(float)
    scores={'LSTM':metrics(obs,pred)}
    pred_df=pd.DataFrame({'date':w['date'][mask],'household':w['household'][mask],
                          'observed_kwh':obs,'predicted_kwh':pred})
    save_table(pred_df,out/'tables'/'test_predictions')
    save_table(pd.DataFrame(hist),out/'tables'/'training_history')
    if args.backend=='torch':
        model.save(out/'checkpoints'/'lstm_best.pt')
        from hydrogen_city.explain import gradient_saliency
        gradient_saliency(model,scaled,masks,out)
        if args.shap:
            from hydrogen_city.explain import shap_attribution
            shap_attribution(model,scaled,masks,out)
    else:model.save(str(out/'checkpoints'/'lstm_best.keras'))
    if not args.skip_benchmarks:
        bpred=benchmark_predictions(scaled,masks)
        if args.full_benchmarks:
            from hydrogen_city.benchmark_suite import standard_models,neural_models
            bpred.update(standard_models(scaled,masks,include_slow=True))
            bpred.update(neural_models(scaled,masks,epochs=args.benchmark_epochs,seed=args.seed))
        tests=[];daily=daily_aggregate(pred_df)
        for name,p in bpred.items():
            scores[name]=metrics(obs,p)
            d=pd.DataFrame({'date':w['date'][mask],'observed_kwh':obs,'predicted_kwh':p})
            aligned=daily.merge(daily_aggregate(d),on='date',suffixes=('_lstm','_baseline'))
            test=diebold_mariano(aligned.observed_kwh_lstm,aligned.predicted_kwh_lstm,aligned.predicted_kwh_baseline)
            tests.append({'comparator':name,**test})
        adjusted=holm_adjust([t['p_value'] for t in tests])
        for t,p in zip(tests,adjusted):t['holm_p_value']=p
        save_table(pd.DataFrame(tests),out/'tables'/'diebold_mariano_tests')
    save_table(pd.DataFrame([{'model':k,**v} for k,v in scores.items()]),out/'tables'/'forecast_metrics')
    # Out-of-sample mean forecast per observed household -> 25,400 dwelling-scale proxy.
    # Ten zone weights are explicitly synthetic pending authentic spatial meter locations.
    forecast_mean=max(0,float(pred_df.predicted_kwh.mean()))
    h2mass_tpd=forecast_mean*cfg['spatial']['n_dwellings']/cfg['hydrogen']['h2_lhv_kwh_kg']/1000
    # For prototype spatial siting use policy-consistent national reference annual demand;
    # forecast estimate is still reported and can be enabled by user config.
    b=baseline(cfg['spatial']['n_dwellings'],cfg['carbon'],cfg['hydrogen'])
    demand_source=cfg['spatial'].get('demand_source','forecast')
    if demand_source not in ('reference_inventory','forecast'):raise ValueError('spatial.demand_source must be reference_inventory or forecast')
    planning_demand=(h2mass_tpd if demand_source=='forecast' else b['annual_h2_tonnes']/365.25)
    rng=np.random.default_rng(args.seed+100)
    weights=rng.dirichlet(np.ones(len(world.zones))*8)
    zone_demand=planning_demand*weights
    dwellings=rng.multinomial(cfg['spatial']['n_dwellings'],weights)
    zones=world.zones.copy()
    zones['dwellings']=dwellings
    zones['demand_tpd']=zone_demand
    zones['allocation_type']='synthetic_weighted'
    save_table(zones.assign(node=zones.node.astype(str)),out/'tables'/'planning_zones')
    save_table(world.sites.assign(node=world.sites.node.astype(str)),out/'tables'/'candidate_sites')
    results=run_spatial(world,zone_demand,cfg['spatial'])
    write_json(out/'spatial_results.json', {key:val for key,val in results.items() if not key.endswith('_plan')})
    save_world_plot(world,results,out/'figures'/'station_pipeline_map.png')
    # Capacity feasibility is guaranteed for served whole zones, not for arbitrary substitution rates.
    coverage=results['optimized']['zone_coverage']
    scenarios,zonals=simulate(zones[['zone_id','dwellings']],coverage,cfg['carbon'],cfg['hydrogen'])
    save_table(scenarios,out/'tables'/'carbon_scenarios')
    save_table(zonals,out/'tables'/'zonal_carbon')
    plotting(pred_df,hist,scenarios,out/'figures')
    summary={'type':input_type,'forecast_metrics':scores,'forecast_household_mean_h2_kwh_day':forecast_mean,
             'forecast_scaled_h2_tonnes_day':h2mass_tpd,'spatial_planning_demand_source':demand_source,
             'spatial_planning_h2_tonnes_day':planning_demand,'baseline_inventory':b,
             'spatial_optimized':results['optimized'], 'spatial_distance_only_baseline':results['distance_only_baseline'],
             'carbon_break_even_kgco2e_per_kg_h2':break_even_intensity(cfg['carbon'],cfg['hydrogen']),
             'heat_regressor_cv':cv,'conversion_metadata':conv,'date_split':dates,
             'runtime_seconds':time.monotonic()-start}
    write_json(out/'summary.json',summary)
    write_json(out/'run_manifest.json',{'input_data_type':input_type,'source_files':raw_files,
        'notes':['Demo inputs are synthetic','LCL gas is estimated using degree days; not measured',
                 'ACORN-to-UCI mapping is synthetic','Spatial allocation is synthetic','Real OSM station locations are provisional',
                 'Exact manuscript results are NOT built into the implementation',
                 'Default planning uses documented annual baseline inventory; select spatial.demand_source=forecast to couple demand forecast into planning',
                 'Torch LSTM omits Keras recurrent dropout; use tensorflow backend for exact recurrent-dropout setting'],
        'python':sys.version,'platform':platform.platform(),'config':cfg,'seed':args.seed,'backend':args.backend,
        'command':' '.join(sys.argv),'train_size':int(masks['train'].sum()),'val_size':int(masks['val'].sum()),
        'test_size':int(masks['test'].sum()),'case_study_type':world.kind})
    print(f'Completed {args.mode} pipeline. Outputs: {out.resolve()}')
    print(f'LSTM test MAE={scores["LSTM"]["mae_kwh_day"]:.3f} kWh/day, R2={scores["LSTM"]["r2"]:.3f}')
    print(f'Spatial: {results["optimized"]["n_opened"]} stations, cost={results["optimized"]["deployment_cost_musd"]:.2f} USD M')
    print('Important: synthetic/modeled fields and manuscript comparison limitations are documented in run_manifest.json')
    return summary

if __name__=='__main__':main()
