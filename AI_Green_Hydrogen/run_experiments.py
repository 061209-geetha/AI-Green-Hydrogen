#!/usr/bin/env python3
"""Five-seed research runner; every experiment is a complete independent run."""
import argparse,subprocess,sys,json
from pathlib import Path
import pandas as pd
import numpy as np
from hydrogen_city.common import write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--mode',choices=['demo','real'],default='demo')
    p.add_argument('--output',default='outputs/repeated')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--n-households',type=int,default=60)
    p.add_argument('--days',type=int,default=380)
    args,extra=p.parse_known_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in args.seeds:
        destination=root/f'seed_{seed}'
        cmd=[sys.executable,'run_pipeline.py','--mode',args.mode,'--seed',str(seed),
             '--output',str(destination),'--epochs',str(args.epochs),
             '--n-households',str(args.n_households),'--days',str(args.days)]+extra
        subprocess.run(cmd,check=True)
        obj=json.loads((destination/'summary.json').read_text(encoding='utf-8'))
        results.append({'seed':seed,**obj['forecast_metrics']['LSTM'],
                        'stations':obj['spatial_optimized']['n_opened'],
                        'pipeline_km':obj['spatial_optimized']['pipeline_unique_km'],
                        'cost_musd':obj['spatial_optimized']['deployment_cost_musd'],
                        'coverage_fraction':obj['spatial_optimized']['coverage_fraction']})
    df=pd.DataFrame(results);df.to_csv(root/'seed_level_results.csv',index=False)
    summary={c:{'mean':float(df[c].mean()),'sd':float(df[c].std(ddof=1)) if len(df)>1 else 0.0,
                'n':len(df)} for c in df.select_dtypes(include=np.number).columns if c!='seed'}
    write_json(root/'five_seed_summary.json',summary)
    print(f'Completed {len(results)} independent runs. Summary: {root/"five_seed_summary.json"}')
if __name__=='__main__':main()
