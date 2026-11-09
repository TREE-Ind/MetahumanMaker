"""Refine coarse eyelid targets on the actual external skin topology."""
import numpy as np
from .ocular_motion import collar_vertices,fitted_blink_pose
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


def physical_collar(faces,positions,margin,distance):
    """Use a physical repair radius rather than a density-dependent ring count."""
    edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0);length=np.linalg.norm(positions[edges[:,0]]-positions[edges[:,1]],axis=1)
    graph=sparse.coo_matrix((np.r_[length,length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(positions),len(positions))).tocsr()
    reached=dijkstra(graph,directed=False,indices=margin,min_only=True,limit=distance)
    support=np.union1d(np.flatnonzero(np.isfinite(reached)),collar_vertices(faces,margin,rings=3))
    return np.unique(faces[np.all(np.isin(faces,support),axis=1)])


def refine_blink(current,open_pose,anatomy,faces,structure,globes,volumes,meeting,closures):
    """Close the native boundary and clear the real globe triangles.

    Positions and fitted volumes share X-right/Z-up/negative-Y-forward units.
    The caller supplies canonical native identities; globes remain unchanged.
    This produces calibration targets, not an accepted rig revision.
    """
    target=np.asarray(current,dtype=float).copy();sheet=np.asarray(structure['outer_triangles'],dtype=int);reports=[]
    for role,closure in closures.items():
        if closure<=0:continue
        lid=structure['eyes'][role];margin=np.asarray(lid['margin']);eye=np.asarray(globes[role]['positions']);tri=np.asarray(globes[role]['triangles']);center=np.asarray(volumes[role]['center']);cross=np.cross(eye[tri[:,1]]-eye[tri[:,0]],eye[tri[:,2]]-eye[tri[:,0]])
        if np.median(np.einsum('ij,ij->i',cross,eye[tri].mean(1)-center))<0:tri=tri[:,[0,2,1]]
        region=dict(collar=physical_collar(sheet,np.asarray(open_pose),margin,np.ptp(eye[:,0])*.65),margin=margin,interior=lid['interior'],sheet=sheet,volume=volumes[role],globe=eye,globe_triangles=tri,meeting_fraction=meeting[role],fair_contact=False,shared_seam=True,convex_contact=True)
        target,proof=fitted_blink_pose(target,open_pose,anatomy,faces,[region],closure=float(closure));reports.append(dict(role=role,**proof))
    return target,dict(method='NATIVE_LOD_BOUNDARY_AND_TRIANGLE_CONTACT',repairs=reports,globe_geometry_changed=False,native_motion_accepted=False)
