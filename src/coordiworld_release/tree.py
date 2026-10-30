"""Evaluate exported XGBoost trees using NumPy, without importing XGBoost."""
import ctypes
import ctypes.util
import json
from pathlib import Path
import numpy as np

def predict(folder, inputs):
    """Predict in float32, preserving strict split, missing-value and tree order."""
    folder=Path(folder)
    config=json.loads((folder/'booster_config.json').read_text())['learner']
    params=config['learner_model_param']
    dimension=int(params['num_feature'])
    x=np.asarray(inputs,dtype=np.float32)
    if x.shape[-1]!=dimension:
        raise ValueError(f'Expected {dimension} features; received {x.shape[-1]}.')
    flat=x.reshape(-1,dimension)
    logistic=config['objective']['name']=='binary:logistic'
    base=float(params['base_score'])
    if logistic:base=np.log(base/(1-base))
    values=np.full(len(flat),base,np.float32)
    for tree in json.loads((folder/'tree_dump.json').read_text()):
        def walk(node,indices):
            if not len(indices):return
            if 'leaf' in node:
                values[indices]+=np.float32(node['leaf'])
                return
            v=flat[indices,int(node['split'][1:])]
            destination=np.where(np.isnan(v),node['missing'],
                                 np.where(v<np.float32(node['split_condition']),node['yes'],node['no']))
            for child in node['children']:
                walk(child,indices[destination==child['nodeid']])
        walk(tree,np.arange(len(flat)))
    if logistic:
        # Linux libm expf matches the pinned CPU transform. Other platforms may
        # differ by a few float32 ulps; numerical replay uses a 2e-7 tolerance.
        name=ctypes.util.find_library('m')
        if name:
            expf=ctypes.CDLL(name).expf
            expf.argtypes=[ctypes.c_float];expf.restype=ctypes.c_float
            expo=np.array([expf(float(v)) for v in np.minimum(-values,np.float32(88.7))],np.float32)
        else:expo=np.exp(np.minimum(-values,np.float32(88.7)))
        values=np.float32(1)/(expo+np.float32(1)+np.float32(1e-16))
    return values.reshape(x.shape[:-1])
