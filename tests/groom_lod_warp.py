"""Whole-field invariants for private native groom LOD deformation."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'create_metahuman'))
from groom_lod_warp import warp


class GroomLODField(unittest.TestCase):
    def setUp(self):
        self.controls=np.array([(x,y,z) for x in range(4) for y in range(4) for z in range(4)],dtype=np.float64)
        self.vertices=np.array([[1.1,1.2,1.3],[1.5,1.2,1.3],[1.1,1.6,1.3],[1.1,1.2,1.7]],dtype=np.float32).astype(np.float64)
        self.triangles=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])

    def test_unchanged_field_retains_native_positions_exactly(self):
        positions,rotations,report=warp(self.controls,self.controls,self.vertices,self.triangles,
            root_positions=self.controls)
        self.assertTrue(report['accepted'],report)
        np.testing.assert_array_equal(positions,self.vertices)
        np.testing.assert_array_equal(rotations,np.broadcast_to(np.eye(3),rotations.shape))

    def test_rigid_motion_includes_card_root_offsets_and_normal_frames(self):
        angle=.42
        rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
        offset=np.array([.2,-.3,.1])
        positions,frames,report=warp(self.controls,self.controls@rotation.T+offset,self.vertices,self.triangles,
            source_uv=self.controls[:,:2]/3,vertex_uv=self.vertices[:,:2]/3,root_positions=self.controls)
        self.assertTrue(report['accepted'],report)
        np.testing.assert_allclose(positions,self.vertices@rotation.T+offset,atol=2e-6,rtol=0)
        np.testing.assert_allclose(frames,np.broadcast_to(rotation,frames.shape),atol=2e-6,rtol=0)
        np.testing.assert_allclose(np.linalg.det(frames),1,atol=2e-6,rtol=0)


if __name__=='__main__':unittest.main(verbosity=2)
