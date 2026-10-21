"""Deterministic horizon counterexample; no trained checkpoints required.

The fixture supplies actions/futures and uses the native analytic rollout and
cost. Production still uses its configured learned robot model for the recheck.
Run: python tests/test_composed_plan.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nwm.planners.planners import RiskAwareNeuralMPC, SAFE, _physics_rollout, _traj_cost


class FixedPlans(RiskAwareNeuralMPC):
    def __init__(self, actions, futures, **kwargs):
        super().__init__(None, None, horizon=actions.shape[1],
                         samples=len(actions), **kwargs)
        self.actions = actions.copy()
        self.futures = futures.copy()

    def _sample_actions(self):
        return self.actions.copy()

    def _predict_obstacles_all(self, obs):
        return self.futures.copy()

    def _rollout(self, robot, actions):
        return _physics_rollout(robot, actions)


def fixture():
    actions = np.empty((2, 18, 2))
    actions[:, :, 0] = 1.2
    actions[0, :, 1] = 1.5
    actions[1, :, 1] = -1.5
    futures = np.tile([6.4, 10.0], (1, 18, 1, 1))
    obs = {"robot": np.array([5., 10., 0.]), "goal": np.array([10., 10.])}
    return actions, futures, obs


def clearance(robot, actions, future):
    traj = _physics_rollout(robot, actions)
    return np.linalg.norm(traj[:, :, None, :2] - future[:, None], axis=-1).min(axis=(0, 2))


class ComposedPlanTests(unittest.TestCase):
    def test_native_counterexample(self):
        actions, futures, obs = fixture()
        np.testing.assert_allclose(clearance(obs["robot"], actions, futures[0]),
                                   [0.7619, 0.7619], atol=5e-5)
        self.assertAlmostEqual(clearance(obs["robot"], actions.mean(axis=0)[None], futures[0])[0], .04)
        costs = _traj_cost(_physics_rollout(obs["robot"], actions), futures[0], obs["goal"], actions)
        np.testing.assert_allclose(costs[0], costs[1])

    def test_default_is_baseline_weighted_mean(self):
        actions, futures, obs = fixture()
        default = FixedPlans(actions, futures)
        disabled = FixedPlans(actions, futures, recheck_composed_plan=False)
        np.testing.assert_array_equal(default.act(obs), disabled.act(obs))
        np.testing.assert_allclose(default.mean, disabled.mean)
        np.testing.assert_allclose(default.mean, np.tile([1.2, 0.], (18, 1)), atol=1e-12)

    def test_enabled_replaces_colliding_mean(self):
        actions, futures, obs = fixture()
        planner = FixedPlans(actions, futures, recheck_composed_plan=True)
        action = planner.act(obs)
        self.assertTrue(any(np.allclose(action, candidate[0]) for candidate in actions))
        selected = np.concatenate([action[None], planner.mean[:-1]], axis=0)
        self.assertGreaterEqual(clearance(obs["robot"], selected[None], futures[0])[0], SAFE)

    def test_checks_every_future(self):
        actions, futures, obs = fixture()
        upper = _physics_rollout(obs["robot"], actions[:1])[-1, 0, :2]
        futures = np.concatenate([futures, np.tile(upper, (1, 18, 1, 1))])
        planner = FixedPlans(actions, futures, lam=1e8, recheck_composed_plan=True)
        np.testing.assert_array_equal(planner.act(obs), actions[1, 0])

    def test_chooses_lowest_cvar_clear_sample(self):
        actions, futures, obs = fixture()
        actions = np.concatenate([actions, np.zeros((1, 18, 2))])
        traj = _physics_rollout(obs["robot"], actions)
        costs = _traj_cost(traj, futures[0], obs["goal"], actions)
        expected = int(np.argmin(costs))
        planner = FixedPlans(actions, futures, lam=1e8, recheck_composed_plan=True)
        np.testing.assert_array_equal(planner.act(obs), actions[expected, 0])

    def test_clear_composed_plan_is_preserved(self):
        actions, futures, obs = fixture()
        futures[:] = [15., 15.]
        enabled = FixedPlans(actions, futures, recheck_composed_plan=True)
        disabled = FixedPlans(actions, futures)
        np.testing.assert_array_equal(enabled.act(obs), disabled.act(obs))
        np.testing.assert_array_equal(enabled.mean, disabled.mean)

    def test_no_clear_sample_retains_baseline(self):
        actions, futures, obs = fixture()
        futures[:] = [5., 10.]
        enabled = FixedPlans(actions, futures, recheck_composed_plan=True)
        disabled = FixedPlans(actions, futures)
        np.testing.assert_array_equal(enabled.act(obs), disabled.act(obs))
        np.testing.assert_array_equal(enabled.mean, disabled.mean)

    def test_selected_plan_is_shifted_for_next_step(self):
        actions, futures, obs = fixture()
        actions[:, -1, 0] = .5
        planner = FixedPlans(actions, futures, recheck_composed_plan=True)
        first = planner.act(obs)
        selected = actions[np.argmin(np.linalg.norm(actions[:, 0] - first, axis=1))]
        expected = np.roll(selected, -1, axis=0)
        expected[-1] = expected[-2]
        np.testing.assert_array_equal(planner.mean, expected)
        np.testing.assert_array_equal(actions[:, -1, 0], [.5, .5])

    def test_nonfinite_prediction_is_not_clear(self):
        actions, futures, obs = fixture()
        traj = _physics_rollout(obs["robot"], actions)
        traj[0, 0, 0] = np.nan
        clear = RiskAwareNeuralMPC._collision_free_in_all_futures(traj, futures)
        np.testing.assert_array_equal(clear, [False, True])

    def test_boundary_matches_native_collision_threshold(self):
        traj = np.zeros((1, 2, 3))
        traj[0, :, 0] = [SAFE, SAFE - 1e-6]
        futures = np.zeros((1, 1, 1, 2))
        clear = RiskAwareNeuralMPC._collision_free_in_all_futures(traj, futures)
        np.testing.assert_array_equal(clear, [True, False])


if __name__ == "__main__":
    unittest.main(verbosity=2)
