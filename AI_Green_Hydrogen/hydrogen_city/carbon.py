"""Carbon accounting is component consistent and coverage/capacity constrained."""
import numpy as np
import pandas as pd

def baseline(n_dwellings,cfg,h2cfg):
    elec=n_dwellings*h2cfg['annual_electric_kwh'];gas=n_dwellings*h2cfg['annual_gas_kwh']
    co2=elec*cfg['electricity_kgco2e_kwh']+gas*cfg['gas_kgco2e_kwh']
    hydrogen_energy=elec/h2cfg['fuel_cell_efficiency']+h2cfg['boiler_ratio']*gas
    return {'dwellings':int(n_dwellings),'annual_electricity_kwh':elec,'annual_gas_kwh':gas,
            'baseline_tco2e_yr':co2/1000,'annual_h2_kwh':hydrogen_energy,
            'annual_h2_tonnes':hydrogen_energy/h2cfg['h2_lhv_kwh_kg']/1000}

def simulate(zones,coverage_fractions,cfg,h2cfg,scenario_levels=None,intensities=None):
    """zones columns zone_id,dwellings; service_fraction can vary by zone.

    The emitted hydrogen mass is directly calculated from displaced electricity/
    heat energy, so kg and kWh emissions factors are never accidentally mixed.
    """
    zone=zones.copy();zone['coverage']=np.asarray(coverage_fractions,float)
    if ((zone.coverage<0)|(zone.coverage>1)).any():raise ValueError('Coverage must be a fraction in [0,1].')
    fractions=scenario_levels or cfg['substitution_fractions']
    factors=intensities or cfg['intensities_kgco2e_kg']
    rows=[];zonals=[]
    for fraction in fractions:
        for h2_intensity in factors:
            for z in zone.itertuples():
                elec=z.dwellings*h2cfg['annual_electric_kwh'];gas=z.dwellings*h2cfg['annual_gas_kwh']
                base=(elec*cfg['electricity_kgco2e_kwh']+gas*cfg['gas_kgco2e_kwh'])/1000
                mass=(elec/h2cfg['fuel_cell_efficiency']+gas*h2cfg['boiler_ratio'])/h2cfg['h2_lhv_kwh_kg']/1000
                actual=float(fraction)*float(z.coverage)
                saved=actual*(base-h2_intensity*mass)
                zonals.append({'zone_id':z.zone_id,'fraction':fraction,'h2_intensity':h2_intensity,
                               'baseline_tco2e':base,'avoided_tco2e':saved,'scenario_tco2e':base-saved,
                               'served_h2_tonnes':mass*actual,'effective_substitution':actual})
            subset=zonals[-len(zone):]
            saved=sum(r['avoided_tco2e'] for r in subset);original=sum(r['baseline_tco2e'] for r in subset)
            rows.append({'fraction':float(fraction),'h2_intensity_kgco2e_kg':float(h2_intensity),
                         'avoided_tco2e_yr':float(saved),'baseline_tco2e_yr':float(original),
                         'reduction_percent':float(saved/original*100) if original else 0,
                         'scenario_emissions_tco2e_yr':float(original-saved)})
    return pd.DataFrame(rows),pd.DataFrame(zonals)

def break_even_intensity(cfg,h2cfg):
    b=baseline(1,cfg,h2cfg)
    return b['baseline_tco2e_yr']/b['annual_h2_tonnes']
