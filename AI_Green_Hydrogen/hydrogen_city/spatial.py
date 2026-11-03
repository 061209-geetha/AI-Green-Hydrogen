"""Spatial graph planning with reproducible demo graph and optional real OSM data.
Distances are ROAD-network km; pipeline footprint is UNION of shared route edges.
Synthetic coordinates are explicitly non-geographic local km coordinates.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import milp,Bounds,LinearConstraint

@dataclass
class SpatialWorld:
    graph: nx.Graph
    zones: pd.DataFrame
    sites: pd.DataFrame
    hub: object
    kind: str


def demo_world(n_zones=10,n_sites=38,seed=42):
    rng=np.random.default_rng(seed)
    G=nx.grid_2d_graph(13,13)
    for a,b in G.edges:
        ax,ay=a;bx,by=b
        # grid each block .4167 km; deterministic road variability
        km=.4166667
        G[a][b].update(length_km=km,accessibility=float(rng.uniform(.7,1)),allowed=True,
                       cost_musd=.24*km+.5*(1-float(rng.uniform(.75,1))))
    for n in G.nodes:
        G.nodes[n]['x_km']=n[0]*5/12;G.nodes[n]['y_km']=n[1]*5/12
    vertices=[n for n in G if 1<n[0]<11 and 1<n[1]<11]
    rng.shuffle(vertices)
    zones=vertices[:n_zones];remaining=[n for n in vertices if n not in zones]
    sites=remaining[:n_sites]
    # guaranteed feasible capacity requires enough sites near demand
    z=pd.DataFrame({'zone_id':[f'Z{i+1:02d}' for i in range(n_zones)],'node':zones})
    s=pd.DataFrame({'site_id':[f'S{i+1:02d}' for i in range(n_sites)],'node':sites,'eligible':True})
    return SpatialWorld(G,z,s,(0,6),'synthetic_demo')

def osm_world(north,south,east,west,n_zones=10,n_sites=38,seed=42,network_type='drive'):
    """Download real OSM roads using OSMnx; candidate points are sampled from graph.

    Candidate coordinates are algorithmically sampled—not verified brownfield/
    fuel-station parcels. Planning results are provisional unless supplied
    with independently validated parcel data.
    """
    try:import osmnx as ox
    except ImportError as e:raise ImportError('Install requirements-geo.txt for OSMnx') from e
    # Support OSMnx 1.x and 2.x bbox conventions.
    try:g=ox.graph_from_bbox(bbox=(west,south,east,north),network_type=network_type,simplify=True)
    except TypeError:g=ox.graph_from_bbox(north,south,east,west,network_type=network_type,simplify=True)
    # projected graph gives lengths in metres; simplify directed to undirected.
    try: g=ox.project_graph(g)
    except Exception:pass
    G=nx.Graph()
    for n,att in g.nodes(data=True):
        G.add_node(n,x_km=float(att.get('x',0))/1000,y_km=float(att.get('y',0))/1000)
    for a,b,att in g.edges(data=True):
        if a==b:continue
        d=float(att.get('length',0))/1000
        if not 0<d<100:continue
        cost=.24*d
        if (not G.has_edge(a,b)) or cost<G[a][b]['cost_musd']:
            G.add_edge(a,b,length_km=d,accessibility=1.,allowed=True,cost_musd=cost)
    if not len(G):raise ValueError('No accessible OSM roads in study area.')
    G=G.subgraph(max(nx.connected_components(G),key=len)).copy()
    rng=np.random.default_rng(seed)
    nodes=list(G.nodes);rng.shuffle(nodes)
    if len(nodes)<n_zones+n_sites+1:raise ValueError('OSM road graph too small.')
    z=pd.DataFrame({'zone_id':[f'Z{i+1:02d}' for i in range(n_zones)],'node':nodes[:n_zones]})
    s=pd.DataFrame({'site_id':[f'S{i+1:02d}' for i in range(n_sites)],'node':nodes[n_zones:n_zones+n_sites], 'eligible':True})
    hub=nodes[n_zones+n_sites]
    return SpatialWorld(G,z,s,hub,'real_osm_provisional_candidates')

def load_points(world,zones_csv=None,sites_csv=None,hub_node=None):
    """Replace provisional points with validated node IDs from user CSV files."""
    if zones_csv:
        z=pd.read_csv(zones_csv)
        if not {'zone_id','node'}.issubset(z.columns):raise ValueError('zones CSV must contain zone_id,node')
        world.zones=z
    if sites_csv:
        s=pd.read_csv(sites_csv)
        if not {'site_id','node'}.issubset(s.columns):raise ValueError('sites CSV must contain site_id,node')
        if 'eligible' not in s:s['eligible']=True
        world.sites=s
    if hub_node is not None:world.hub=hub_node
    for n in list(world.zones.node)+list(world.sites.node)+[world.hub]:
        if n not in world.graph:raise ValueError(f'Point node {n} absent from graph.')
    return world

def shortest_matrices(world):
    G=world.graph;zones=list(world.zones.node);sites=list(world.sites.node)
    D=np.full((len(zones),len(sites)),np.inf);C=D.copy();A=np.zeros_like(D)
    for j,s in enumerate(sites):
        lengths=nx.single_source_dijkstra_path_length(G,s,weight='length_km')
        costs=nx.single_source_dijkstra_path_length(G,s,weight='cost_musd')
        for i,z in enumerate(zones):
            D[i,j]=lengths.get(z,np.inf);C[i,j]=costs.get(z,np.inf)
            if np.isfinite(D[i,j]):
                A[i,j]=float(np.exp(-D[i,j]/3.5))
    return D,C,A

def solve_siting(world,demand_tpd,cfg,method='milp',k_fixed=None,dist_only=False):
    """Capacitated facility location MILP (zonal connection costs + opening costs).

    Global shared-corridor tree cost is assessed after MILP; solving integrated
    Steiner design would require additional flow variables and is NOT claimed.
    """
    dem=np.asarray(demand_tpd,float)
    nz=len(world.zones);ns=len(world.sites)
    if len(dem)!=nz or np.any(dem<0):raise ValueError('Demand must be nonnegative per zone.')
    D,C,A=shortest_matrices(world)
    radius=float(cfg['radius_km']);cap=float(cfg['station_capacity_tpd']);K=int(cfg['k_max'])
    eligible=world.sites.eligible.astype(bool).to_numpy()
    valid=np.isfinite(D)&eligible[None,:]
    if not dist_only:valid &= D<=radius+1e-9
    if not np.all(valid.any(axis=1)) and not dist_only:raise ValueError('At least one zone has no eligible station within radius.')
    if not dist_only and np.sum(dem)>min(K,eligible.sum())*cap+1e-8:
        raise ValueError('Insufficient total station capacity: increase K or station_capacity_tpd.')
    # Variables y_s (open) + u_zs (assignment). Binary, with sum_s u_zs == 1.
    n=ns+nz*ns;obj=np.zeros(n)
    obj[:ns]=float(cfg['station_cost_musd']) if not dist_only else 0
    if dist_only:
        # distance-only p-median, Euclidean on x/y of graph node positions
        def xy(n):return np.array([world.graph.nodes[n]['x_km'],world.graph.nodes[n]['y_km']])
        eu=np.array([[np.linalg.norm(xy(z)-xy(s)) for s in world.sites.node] for z in world.zones.node])
        obj[ns:]=eu.ravel()
    else:
        obj[ns:]=np.where(np.isfinite(C),C,1e8).ravel()
    lo=np.zeros(n);hi=np.ones(n)
    for i in range(nz):
        for j in range(ns):
            if not valid[i,j]:hi[ns+i*ns+j]=0
    for j in range(ns):
        if not eligible[j]:hi[j]=0
    rows=[];lbs=[];ubs=[]
    def constrain(coefs,lb,ub):rows.append(coefs);lbs.append(lb);ubs.append(ub)
    for i in range(nz):
        r=np.zeros(n);r[ns+i*ns:ns+(i+1)*ns]=1;constrain(r,1,1)
    for i in range(nz):
        for j in range(ns):
            r=np.zeros(n);r[ns+i*ns+j]=1;r[j]=-1;constrain(r,-np.inf,0)
    if not dist_only:
        for j in range(ns):
            r=np.zeros(n);r[j]=-cap
            for i in range(nz):r[ns+i*ns+j]=dem[i]
            constrain(r,-np.inf,0)
    r=np.zeros(n);r[:ns]=1
    if k_fixed is not None:constrain(r,k_fixed,k_fixed)
    else:constrain(r,0,K)
    from scipy.sparse import csr_matrix
    res=milp(obj,integrality=np.ones(n),bounds=Bounds(lo,hi),constraints=LinearConstraint(csr_matrix(np.asarray(rows)),lbs,ubs),
             options={'time_limit':120,'mip_rel_gap':.005})
    if res.x is None:raise ValueError(f'Siting MILP infeasible/failed: {res.message}')
    opened=np.flatnonzero(res.x[:ns]>.5).tolist()
    ass=np.asarray(res.x[ns:]).reshape(nz,ns).argmax(axis=1).tolist()
    return {'opened':opened,'assignments':ass,'milp_objective':float(res.fun), 'milp_status':str(res.message),
            'distances':D, 'accessibility':A}

def evaluate_plan(world,plan,demand_tpd,cfg,radial=False,coverage_only=False):
    G=world.graph;zn=list(world.zones.node);sn=list(world.sites.node);hub=world.hub
    D=plan['distances'];opened=plan['opened'];ass=plan['assignments']
    demand=np.asarray(demand_tpd,float)
    permitted=[bool(D[i,j]<=cfg['radius_km']+1e-9) for i,j in enumerate(ass)]
    served=np.where(permitted,demand,0.)
    cov=float(served.sum()/demand.sum()) if demand.sum()>0 else 1.
    if radial:
        # Conventional baseline: duplicate routes retained in total installed length.
        paths=[]
        for j in opened:
            paths.append(nx.shortest_path(G,hub,sn[j],weight='cost_musd'))
        for i,j in enumerate(ass):
            paths.append(nx.shortest_path(G,zn[i],sn[j],weight='cost_musd'))
        total=0.;unique=set()
        for path in paths:
            for a,b in zip(path[:-1],path[1:]):
                edge=frozenset([a,b]);total+=G[a][b]['length_km'];unique.add(edge)
        unique_dist=sum(G[tuple(e)[0]][tuple(e)[1]]['length_km'] for e in unique)
        redundancy=(total-unique_dist)/total if total else 0
    else:
        from networkx.algorithms.approximation.steinertree import steiner_tree
        terminals=list(dict.fromkeys([hub]+[sn[j] for j in opened]+[zn[i] for i,j in enumerate(ass) if permitted[i]]))
        sub=steiner_tree(G,terminals,weight='cost_musd')
        unique_dist=float(sum(d['length_km'] for _,_,d in sub.edges(data=True)))
        total=unique_dist;redundancy=0.
    cost=float(total*cfg['pipeline_cost_musd_km']+len(opened)*cfg['station_cost_musd'])
    response=float(np.average([D[i,j] for i,j in enumerate(ass)],weights=np.maximum(demand,1e-8)))
    # manuscript routed supply-path mean is different from unique installed corridor length
    supplied=[]
    for i,j in enumerate(ass):
        if permitted[i]:
            supplied.append(nx.shortest_path_length(G,hub,zn[i],weight='length_km'))
    mean_routed=float(np.mean(supplied)) if supplied else None
    capacities={str(world.sites.iloc[j].site_id):float(demand[np.asarray(ass)==j].sum()) for j in opened}
    return {'n_opened':len(opened),'opened_ids':[str(world.sites.iloc[j].site_id) for j in opened],
            'assignments':{str(world.zones.iloc[i].zone_id):str(world.sites.iloc[j].site_id) for i,j in enumerate(ass)},
            'coverage_fraction':cov,'served_demand_tpd':float(served.sum()),'total_demand_tpd':float(demand.sum()),
            'pipeline_unique_km':float(unique_dist),'pipeline_installed_km':float(total),'mean_routed_hub_zone_km':mean_routed,
            'deployment_cost_musd':cost,'mean_station_response_km':response,'redundancy_ratio':float(redundancy),
            'station_load_tpd':capacities,'zone_coverage':permitted}

def run_spatial(world,demand_tpd,cfg):
    optimized=solve_siting(world,demand_tpd,cfg)
    baseline=solve_siting(world,demand_tpd,cfg,dist_only=True,k_fixed=min(4,len(world.sites)))
    return {'optimized':evaluate_plan(world,optimized,demand_tpd,cfg),
            'distance_only_baseline':evaluate_plan(world,baseline,demand_tpd,cfg,radial=True),
            'optimized_plan':optimized,'baseline_plan':baseline}

def save_world_plot(world,results,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,7))
    G=world.graph
    pos={n:(v['x_km'],v['y_km']) for n,v in G.nodes(data=True)}
    nx.draw_networkx_edges(G,pos,ax=ax,width=.35,edge_color='#cccccc',alpha=.65)
    nx.draw_networkx_nodes(G,pos,nodelist=world.zones.node.tolist(),ax=ax,node_size=95,node_color='#2468a0',label='Demand zones')
    nx.draw_networkx_nodes(G,pos,nodelist=world.sites.node.tolist(),ax=ax,node_size=16,node_color='#999999',label='Candidate stations')
    nx.draw_networkx_nodes(G,pos,nodelist=[world.hub],ax=ax,node_size=180,node_color='#b24431',node_shape='s',label='Supply hub')
    opened=[world.sites.iloc[j].node for j in results['optimized_plan']['opened']]
    nx.draw_networkx_nodes(G,pos,nodelist=opened,ax=ax,node_size=150,node_color='#28885a',node_shape='D',label='Selected stations')
    try:
        from networkx.algorithms.approximation.steinertree import steiner_tree
        connected=list(dict.fromkeys([world.hub]+opened+world.zones.node.tolist()))
        tree=steiner_tree(G,connected,weight='cost_musd')
        nx.draw_networkx_edges(tree,pos,ax=ax,edge_color='#15714c',width=2.1)
    except Exception:pass
    ax.set_xlabel('Local x (km)' if world.kind=='synthetic_demo' else 'Projected x (km)')
    ax.set_ylabel('Local y (km)' if world.kind=='synthetic_demo' else 'Projected y (km)')
    ax.set_aspect('equal');ax.legend(fontsize=8,loc='best');fig.tight_layout()
    fig.savefig(path,dpi=600);plt.close(fig)
