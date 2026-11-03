import unittest
import numpy as np
from hydrogen_city.common import config
from hydrogen_city.data import make_demo,train_heat_archetypes,create_hydrogen_targets
from hydrogen_city.forecast import make_windows,partitions,standardize
from hydrogen_city.spatial import demo_world,run_spatial
from hydrogen_city.carbon import baseline,simulate,break_even_intensity
from hydrogen_city.research import holm_adjust

class TestHydrogenCity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=config()
    def test_baseline_inventory(self):
        b=baseline(25400,self.c['carbon'],self.c['hydrogen'])
        self.assertAlmostEqual(b['baseline_tco2e_yr']/1000,69.9,delta=.25)
        self.assertAlmostEqual(b['annual_h2_tonnes']/1000,13.7,delta=.1)
    def test_carbon_mass_and_intensity(self):
        import pandas as pd
        zones=pd.DataFrame({'zone_id':['a','b'],'dwellings':[100,200]})
        rows,_=simulate(zones,[1,0],self.c['carbon'],self.c['hydrogen'],[1],[0.75,11.4])
        self.assertGreater(rows.iloc[0].avoided_tco2e_yr,0)
        self.assertLess(rows.iloc[1].avoided_tco2e_yr,0)
        self.assertTrue(0<break_even_intensity(self.c['carbon'],self.c['hydrogen'])<10)
    def test_no_time_leakage(self):
        m,w,u=make_demo(5,110,7)
        heat,_=train_heat_archetypes(u,7)
        d,_=create_hydrogen_targets(m,w,heat,u,self.c['hydrogen'],7)
        win=make_windows(d,24)
        masks,dates=partitions(win)
        a=win['date'][masks['train']];b=win['date'][masks['val']];c=win['date'][masks['test']]
        self.assertLess(a.max(),b.min());self.assertLess(b.max(),c.min())
        scaled=standardize(win,masks)
        self.assertEqual(scaled['x'].shape[1],24)
    def test_spatial_capacity_and_cover(self):
        world=demo_world(10,38,42)
        dem=np.full(10,3.75)
        r=run_spatial(world,dem,self.c['spatial'])
        self.assertEqual(r['optimized']['coverage_fraction'],1.)
        self.assertLessEqual(r['optimized']['n_opened'],self.c['spatial']['k_max'])
        self.assertTrue(all(v<=12.001 for v in r['optimized']['station_load_tpd'].values()))
    def test_holm(self):
        p=holm_adjust([0.01,0.03,.2]);self.assertAlmostEqual(p[0],.03)
if __name__=='__main__':unittest.main()
