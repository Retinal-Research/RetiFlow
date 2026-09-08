"""Original RBAD fast_keypoints, endpoint repair, then the same RBAD again."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import utils as original_rbad
from .endpoint_bridge import bridge_endpoints, EndpointBridgeConfig

S8 = np.ones((3,3),bool)


def run_original(skeleton, *, root_yx=None, tail=15, child_min_dist=5,
                 angle_min=20., angle_max=120.):
    skel = np.asarray(skeleton, dtype=bool)
    if skel.ndim != 2 or not skel.size:
        raise ValueError('skeleton must be nonempty and 2D')
    labels, count = ndi.label(skel,S8)
    if not count:
        return {'angles':[],'keypoints':[],'roots_yx':[],'root_yx':None,'components':0,
                'primary_component_pixels':0}
    points = np.argwhere(skel)
    if root_yx is None:
        density = cv2.GaussianBlur(skel.astype(np.float32),(0,0),21)
        center = np.asarray(np.unravel_index(np.argmax(density),skel.shape))
    else:
        center = np.asarray(root_yx,float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError('root must be a finite (y,x) pair')
    root = points[np.argmin(np.abs(points-center).sum(axis=1))]
    primary = int(labels[tuple(root)])
    all_keys, angles, roots = [], [], []
    # The same policy in both rounds: main root first, then each remaining
    # island from its closest pixel to that root. No component is hidden.
    for component in [primary]+[c for c in range(1,count+1) if c != primary]:
        mask = labels == component
        local_points = np.argwhere(mask)
        local_root = local_points[np.argmin(np.abs(local_points-root).sum(axis=1))]
        roots.append(local_root.tolist())
        # Guard legacy 3x3 slicing at image boundaries without changing its
        # traversal, skeleton geometry, or image resolution.
        padded = np.pad(mask.astype(np.uint8),1)
        *_, keys = original_rbad.fast_keypoints(padded,[tuple(local_root+1)],tail=tail,save_gif=False)
        records = []
        for node,node_type,level,step,parent in keys:
            records.append({'point_xy':[int(node[0])-1,int(node[1])-1],
                            'type':int(node_type),'level':int(level),'step':int(step),
                            'parent_xy':[int(parent[0])-1,int(parent[1])-1],
                            'component':int(component)})
        all_keys.extend(records)
        for record in records:
            if record['type'] != 1:
                continue
            node = record['point_xy']
            children = [r['point_xy'] for r in records if r['parent_xy']==node
                        and original_rbad.manhattan_distance(r['point_xy'],node)>=child_min_dist]
            if len(children)<2:
                continue
            a,c = children[:2]  # Exactly the original wrapper's child selection.
            try:
                angle,_ = original_rbad.calculate_angle(a,node,c)
            except (ZeroDivisionError,ValueError):
                continue
            if angle_min <= angle <= angle_max:
                angles.append({'points':node,'angle':float(angle),
                               'children':children[:2],'parent_xy':record['parent_xy'],
                               'component':int(component)})
    return {'angles':angles,'keypoints':all_keys,'roots_yx':roots,'root_yx':root.tolist(),
            'components':int(count),'primary_component_pixels':int(np.sum(labels==primary))}


def detect_two_pass(skeleton, *, bv_probability=None, bridge_config=None,
                    root_yx=None, tail=15, child_min_dist=5, angle_min=20., angle_max=120.):
    parameters = dict(tail=tail,child_min_dist=child_min_dist,angle_min=angle_min,angle_max=angle_max)
    before = run_original(skeleton,root_yx=root_yx,**parameters)
    repaired, repair = bridge_endpoints(skeleton,bv_probability=bv_probability,config=bridge_config)
    original_end_parents = {tuple(record['point_xy']):record['parent_xy']
                            for record in before['keypoints'] if record['type']==2}
    for bridge in repair['bridges']:
        bridge['source_rbad_parent_xy'] = original_end_parents.get(tuple(bridge['source_xy']))
        bridge['target_rbad_parent_xy'] = original_end_parents.get(tuple(bridge['target_xy']))
    after = run_original(repaired,root_yx=before['root_yx'],**parameters)
    unmatched = set(range(len(before['angles'])))
    additions, matches = [], []
    for current in after['angles']:
        point = np.asarray(current['points'])
        nearby = [(float(np.linalg.norm(point-np.asarray(before['angles'][i]['points']))),i)
                  for i in unmatched]
        if nearby and min(nearby)[0] <= 3:
            _,index = min(nearby)
            unmatched.remove(index)
            matches.append({'before':before['angles'][index],'after':current,
                            'angle_change':current['angle']-before['angles'][index]['angle']})
        else:
            additions.append(current)
    return repaired, {'backend':'RBAD.utils.fast_keypoints (original)',
                      'traversal':'main component then every island, identical before/after',
                      'before':before,'after':after,'repair':repair,
                      'new_candidates':additions,'matched_candidates':matches,
                      'removed_candidates':[before['angles'][i] for i in sorted(unmatched)]}
