"""Known-shape multi-view regression; NumPy/SciPy, no licensed model data."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'create_metahuman'))
from photo_fit_core import fit,fit_multiview, project, primary_corridor,eye_shape_operator


class MultiViewShape(unittest.TestCase):
    def model(self):
        points = np.zeros((68, 3))
        angle = np.linspace(np.pi, 2 * np.pi, 17)
        points[:17, :2] = np.c_[.3 * np.cos(angle), .35 * np.sin(angle)]
        points[17:22, :2] = np.c_[np.linspace(-.22, -.06, 5), [.16, .18, .19, .18, .16]]
        points[22:27, :2] = points[17:22, :2] * [-1, 1]
        points[27:31, 1] = np.linspace(.13, -.09, 4)
        points[31:36, :2] = np.c_[np.linspace(-.07, .07, 5), [-.08, -.10, -.11, -.10, -.08]]
        for start, center in ((36, -.14), (42, .14)):
            a = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            points[start:start + 6, :2] = np.c_[center + .06 * np.cos(a), .09 + .025 * np.sin(a)]
        # Standard outer-eye ordering is important for camera/profile detection.
        points[36:42] = points[36:42][[3, 2, 1, 0, 5, 4]]
        for start, count, scale in ((48, 12, 1.), (60, 8, .6)):
            a = np.linspace(0, 2 * np.pi, count, endpoint=False)
            points[start:start + count, :2] = np.c_[.11 * scale * np.cos(a), -.2 + .035 * scale * np.sin(a)]
        points[:, 2] = .03 * (1 - (points[:, 0] / .3) ** 2)
        modes = np.zeros((170, 68, 3))
        modes[0, 27:36, 2] = [.01, .02, .03, .04, .02, .03, .04, .03, .02]
        definition = np.zeros((68, 6))
        definition[:, 0], definition[:, 1] = np.arange(68), 1
        return points, modes, definition

    def test_depth_recovery_keeps_front_and_non_head_identity(self):
        points, modes, definition = self.model()
        truth = points + modes[0] * .5
        initial = np.zeros(253)
        initial[170:] = np.linspace(-.2, .2, 83)
        views = []
        for yaw in (0., -.7, .7, -1.2, 1.2):
            camera = np.array([0., yaw, 0., 0., 0., 0.])
            views.append(dict(name=str(yaw), target=project(truth, camera), generated=False))
        result = fit_multiview(points, modes, definition, views, initial, np.zeros(6))
        fitted = points + np.tensordot(np.array(result['coefficients'])[:170], modes, axes=(0, 0))
        self.assertLess(np.linalg.norm(fitted - truth), np.linalg.norm(points - truth) * .5)
        np.testing.assert_array_equal(result['coefficients'][170:], initial[170:])
        self.assertLess(result['primary_drift_normalized'], .01)
        self.assertLess(result['weighted_error_after'], result['weighted_error_before'])
        self.assertTrue(result['camera_only_baseline'])

    def test_primary_corrections_are_not_frozen_with_the_old_mismatch(self):
        points, modes, definition = self.model()
        modes[:] = 0
        modes[0, 17:27, 1] = .12
        truth = points + modes[0] * .5
        views = [dict(name=str(yaw), target=project(truth, np.array([0., yaw, 0., 0., 0., 0.])),
                      generated=index != 0) for index, yaw in enumerate((0., -.7, .7))]
        result = fit_multiview(points, modes, definition, views, np.zeros(253), np.zeros(6))
        self.assertGreater(result['primary_drift_normalized'], .01)
        self.assertLess(result['primary_outward_drift_normalized'], .01)
        self.assertLess(result['primary_regions']['brows']['after'],
                        result['primary_regions']['brows']['before'] * .5)
        for region in result['primary_regions'].values():
            self.assertLessEqual(region['after'], region['before'] * 1.05 + .002)
        # A target-directed move is free; overshoot and drift away still cost.
        baseline = np.array([[0., 1.]])
        target = np.array([[1., 0.]])
        np.testing.assert_array_equal(primary_corridor(np.array([[.5, .5]]), baseline, target)[0], 0)
        error, slope = primary_corridor(np.array([[1.1, 1.2]]), baseline, target)
        np.testing.assert_allclose(error, [[.1, .2]])
        np.testing.assert_array_equal(slope, 1)

    def test_photo_eye_contours_recover_aperture_and_slope(self):
        points,modes,definition=self.model();modes[:]=0
        for start in (36,42):
            sign=1 if start==36 else -1
            eye=points[start:start+6]
            modes[0,start:start+6,1]=(eye[:,0]-eye[:,0].mean())*sign*.5
            modes[1,start+1:start+3,1]=.03
            modes[1,start+4:start+6,1]=-.03
        truth=points+modes[0]*.4+modes[1]*.4
        target=project(truth,np.zeros(6));operator=eye_shape_operator(target)
        # Eye placement has its own observations; this term measures contour.
        translated=target.copy();translated[36:42]+=[.2,-.1];translated[42:48]+=[-.1,.3]
        np.testing.assert_allclose(operator@(translated-target).ravel(),0,atol=1e-14)
        initial=np.zeros(253);initial[170:]=np.linspace(-.1,.1,83)
        views=[dict(target=project(truth,np.array([0.,yaw,0.,0.,0.,0.])),generated=i>0)
               for i,yaw in enumerate((0.,-.6,.6))]
        before=np.linalg.norm(operator@(points[:,:2]-target).ravel())
        for result in (fit(points,modes,definition,target,initial),
                       fit_multiview(points,modes,definition,views,initial,np.zeros(6))):
            after=np.linalg.norm(operator@(np.array(result['projected'])-target).ravel())
            self.assertLess(after,before*.5)
            np.testing.assert_array_equal(result['coefficients'][170:],initial[170:])


if __name__ == '__main__':
    unittest.main(verbosity=2)
