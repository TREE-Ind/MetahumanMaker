"""Component overrides follow group identity through blends and new sources."""
from pathlib import Path
import runpy
import unittest

api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'create_metahuman/unreal_groom.py'))
plan = api['_group_override_plan']


def origin(asset, group):
    return dict(project='C:/Demo/Demo.uproject', asset=asset, group=group)


class GroomGroupMapping(unittest.TestCase):
    def test_asset_lod_policy_survives_and_conflicting_blends_stop(self):
        legacy=dict(enable_global_interpolation=False,enable_simulation_cache=False,hair_interpolation_type='RIGID_TRANSFORM')
        manual=dict(legacy,lod_mode='MANUAL',auto_lod_bias=.25)
        self.assertEqual(api['asset_settings']([dict(asset_settings=manual)]),manual)
        self.assertEqual(api['asset_settings']([dict(asset_settings=legacy)])['lod_mode'],'DEFAULT')
        with self.assertRaisesRegex(ValueError,'different asset-wide'):
            api['asset_settings']([dict(asset_settings=manual),dict(asset_settings=legacy)])
        for invalid in (dict(manual,auto_lod_bias=float('nan')),dict(manual,lod_mode='UNKNOWN')):
            with self.assertRaises(ValueError):api['asset_settings']([dict(asset_settings=invalid)])

    def test_reordered_and_added_groups(self):
        source = [[origin('/Game/Old', 0)], [origin('/Game/Old', 1)]]
        target = [[origin('/Game/New', 0), *source[1]],
                  [origin('/Game/New', 1), *source[0]], [origin('/Game/Added', 0)]]
        result = plan(source, target, [True] * 3, ['thin', 'thick'])
        self.assertEqual([r['source_groups'] for r in result], [[1], [0], []])
        self.assertEqual(result[2]['policy'], 'INCOMING_SETTINGS')

    def test_ambiguous_controls_require_a_choice_but_equal_controls_are_safe(self):
        ancestor = origin('/Game/Shared', 0)
        source = [[origin('/Game/Old', i), ancestor] for i in range(2)]
        target = [[origin('/Game/New', 0), ancestor]]
        with self.assertRaisesRegex(ValueError, 'Use Incoming Group Settings'):
            plan(source, target, [True], ['thin', 'thick'])
        self.assertEqual(plan(source, target, [True], ['same', 'same'])[0]['source_groups'], [0, 1])
        self.assertEqual(plan(source, target, [True], ['thin', 'thick'], 'TARGET_DEFAULTS')[0]['source_groups'], [])

    def test_nearest_origin_wins_and_legacy_order_is_bounded(self):
        base = origin('/Game/Base', 0)
        source = [[origin('/Game/Old', 0), base], [origin('/Game/Old', 1), base]]
        target = [[origin('/Game/New', 0), origin('/Game/Old', 1), base]]
        self.assertEqual(plan(source, target, [True], ['thin', 'thick'])[0]['source_groups'], [1])
        target[0].append(dict(origin('/Game/Old', 1), project='C:\\Demo\\Demo.uproject'))
        self.assertEqual(plan(source, target, [True], ['thin', 'thick'])[0]['source_groups'], [1])
        legacy = [[origin('/Game/Legacy', i)] for i in range(3)]
        self.assertEqual([r['source_groups'] for r in plan(source, legacy, [False] * 3, ['thin', 'thick'])], [[0], [1], []])
        self.assertEqual([r['source_groups'] for r in plan(source, legacy, [True] * 3, ['thin', 'thick'])], [[], [], []])


if __name__ == '__main__':
    unittest.main(verbosity=2)
