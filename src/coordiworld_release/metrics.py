"""Score arithmetic and drive-cluster resampling used by the revised study."""
import numpy as np
import pandas as pd

SHORT=['NC','DAC','DDC','TLC','P','TTC','LK','HC']
LONG=['no_at_fault_collisions','drivable_area_compliance',
      'driving_direction_compliance','traffic_light_compliance','ego_progress',
      'time_to_collision_within_bound','lane_keeping','history_comfort']

def utility(components,ec):
    c=np.asarray(components,dtype=float);ec=np.asarray(ec,dtype=float)
    return c[...,:4].prod(axis=-1)*(5*c[...,4]+5*c[...,5]+2*c[...,6]+2*c[...,7]+2*ec)/16

def macro(values,drive):
    return float(pd.DataFrame({'value':values,'drive':drive}).groupby('drive').value.mean().mean())

def balanced_weights(scene,drive):
    scene=np.asarray(scene);drive=np.asarray(drive)
    count={s:int((scene==s).sum()) for s in np.unique(scene)}
    scenes={d:len(np.unique(scene[drive==d])) for d in np.unique(drive)}
    w=np.array([1/(count[s]*scenes[d]) for s,d in zip(scene,drive)],float)
    return w/w.mean()

def clustered(delta,drive,seed=102819):
    """Resample whole drives; retain mapping weights in the mapping-equal target."""
    d=pd.DataFrame({'delta':np.asarray(delta,float),'drive':np.asarray(drive)})
    q=d.groupby('drive').delta.agg(['sum','size','mean']).sort_index()
    s=q['sum'].to_numpy();n=q['size'].to_numpy();m=q['mean'].to_numpy()
    ix=np.random.default_rng(seed).integers(len(q),size=(10000,len(q)))
    return {'point':float(s.sum()/n.sum()),
            'ci95_mapping_equal_drive_cluster':np.quantile(s[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),
            'drive_equal_point':float(m.mean()),'ci95_drive_equal':np.quantile(m[ix].mean(1),[.025,.975]).tolist(),
            'drives':len(q),'bootstrap_samples':10000,'seed':seed}

def aggregate(table,groups):
    """Two-stage, two-frame aggregation of saved native scores and endpoint weights."""
    t=table.set_index('token');rows=[]
    for g in groups:
        pair=[g['initial_token'],g['followup_token']]
        first=[];second=[]
        for frame in [0,1]:
            a=t.loc[pair[frame]]
            b=t.loc[[p[frame] for p in g['synthetic_initial_followup_pairs']]]
            first.append(float(a.score));second.append(float(np.average(b.score,weights=b.weight)))
        rows.append({'group':g['registered_group_index'],'drive':g['drive'],
                     'combined':float(np.mean(np.array(first)*np.array(second))),
                     'stage_one':float(np.mean(first)),'stage_two':float(np.mean(second))})
    return pd.DataFrame(rows).set_index('group')

def moment_projection(y):
    """Euclidean projection onto 0 <= each joint moment <= Q <= 1."""
    y=np.asarray(y,dtype=float);q=y[:,0].copy()
    children=np.sort(np.maximum(y[:,1:],0),axis=1)[:,::-1]
    total=q.copy();count=np.ones(len(q))
    for k in range(3):
        active=children[:,k]>q
        total+=np.where(active,children[:,k],0);count+=active;q=total/count
    q=np.clip(q,0,1)
    return np.column_stack((q,np.minimum(np.maximum(y[:,1:],0),q[:,None])))
