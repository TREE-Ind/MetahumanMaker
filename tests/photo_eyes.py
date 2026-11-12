"""Observed iris sampling must reject lids/reflections and preserve distinct eyes."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'create_metahuman'))
from photo_eye_core import iris_guidance


class IrisGuidance(unittest.TestCase):
    def reference(self):
        image=np.full((100,180,3),[190,130,110],dtype=np.uint8)
        points=np.zeros((478,2));y,x=np.mgrid[:100,:180]
        for center,ids,lids,color in [((50,50),(468,469,470,471,472),(33,160,158,133,153,144),(90,60,35)),
                                      ((130,50),(473,474,475,476,477),(362,385,387,263,373,380),(40,90,65))]:
            center=np.array(center);points[ids[0]]=center
            points[list(ids[1:])]=center+[[20,0],[0,-20],[-20,0],[0,20]]
            points[list(lids)]=center+[[-24,0],[-12,-13],[12,-13],[24,0],[12,13],[-12,13]]
            radius=np.hypot(x+.5-center[0],y+.5-center[1]);image[radius<20]=color;image[radius<8]=0
            image[(x-center[0]-12)**2+(y-center[1]+1)**2<16]=255
        return image,points

    def test_distinct_pigment_survives_pupils_skin_and_glints(self):
        image,points=self.reference();result=iris_guidance(image,points)
        self.assertEqual(result['eyes']['image_left']['hex'],'#5A3C23')
        self.assertEqual(result['eyes']['image_right']['hex'],'#285A41')
        for eye in result['eyes'].values():self.assertGreater(eye['pixels'],24)
        self.assertEqual(iris_guidance(np.zeros_like(image),points)['status'],'unavailable')
        self.assertEqual(iris_guidance(image,points[:468])['status'],'unavailable')
        for ids in ((33,160,158,133,153,144),(362,385,387,263,373,380)):
            points[list(ids),1]=50
        self.assertEqual(iris_guidance(image,points)['status'],'unavailable')


if __name__=='__main__':unittest.main(verbosity=2)
