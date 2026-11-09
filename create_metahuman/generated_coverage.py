"""Bounded chart-local extension of a generated atlas's exterior background."""
import cv2
import numpy as np
from scipy import ndimage


def fill_exterior(color,valid,eligible,*,limit=None):
    height,width=valid.shape
    if color.shape!=(height,width,3) or eligible.shape!=valid.shape:
        raise ValueError('Generated UV coverage requires matching atlas inputs')
    dark=(color.max(axis=2)<=.01).astype(np.uint8)
    count,labels=cv2.connectedComponents(dark,8)
    border=np.unique(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))
    border=border[border!=0]
    exterior=np.isin(labels,border);del labels,dark
    missing=exterior&(valid>.5)&(eligible>0)
    report={'method':'bounded_chart_local_exterior_extension','missing_editable_texels':int(missing.sum()),
            'filled_texels':0,'maximum_distance_texels':0.,'physical_detail_recovered':False}
    if not np.any(missing):return report
    limit=float(limit if limit is not None else max(width,height)/128)
    count,charts,stats,_=cv2.connectedComponentsWithStats((valid>.5).astype(np.uint8),8)
    for index in np.unique(charts[missing]):
        if not index:continue
        x,y,w,h,_=stats[index];chart=charts[y:y+h,x:x+w]==index
        holes=missing[y:y+h,x:x+w]&chart
        known=chart&~exterior[y:y+h,x:x+w]
        if not np.any(known):raise ValueError('Generated albedo does not cover a selected body UV island')
        distance,nearest=ndimage.distance_transform_edt(~known,return_indices=True)
        maximum=float(distance[holes].max())
        if maximum>limit:raise ValueError(f'Generated body UV island is displaced by {maximum:.1f} texels; align its layout before material derivation')
        patch=color[y:y+h,x:x+w];yy,xx=np.nonzero(holes)
        patch[yy,xx]=patch[nearest[0,yy,xx],nearest[1,yy,xx]]
        report['filled_texels']+=len(yy)
        report['maximum_distance_texels']=max(report['maximum_distance_texels'],maximum)
    report['distance_limit_texels']=limit
    return report
