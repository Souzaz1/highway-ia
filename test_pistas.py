"""Validação do MVP: python -m unittest test_pistas -v (Python 3.9)."""
from __future__ import annotations

import math
import random
import unittest
from unittest.mock import patch
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from pista_aleatoria import ENV_ID, RandomRacetrackEnv
from teste_visual import follower_action

PROCEDURAL_PATH = Path(__file__).resolve().parent.parent / 'procedural-tracks'
HAS_PROCEDURAL = (PROCEDURAL_PATH / 'main.py').is_file()


def fingerprint(env):
    return tuple(tuple(np.round(lane.position(s, 0), 7))
                 for lane in env.track_lanes for s in (0, lane.length / 2, lane.length))


class RandomTracksTests(unittest.TestCase):
    def generators(self):
        yield 'native'
        if HAS_PROCEDURAL:
            yield 'procedural'

    def make_env(self, generator='native', **extra):
        config = {'track_generator': generator, 'procedural_path': str(PROCEDURAL_PATH)}
        config.update(extra)
        return RandomRacetrackEnv(config=config)

    def test_reset_and_step_for_60_tracks(self):
        for generator in self.generators():
            env = self.make_env(generator)
            try:
                for seed in range(30):
                    with self.subTest(generator=generator, seed=seed):
                        obs, info = env.reset(seed=seed)
                        self.assertTrue(env.observation_space.contains(obs))
                        self.assertTrue(np.isfinite(obs).all())
                        self.assertTrue(env.vehicle.on_road)
                        self.assertEqual(len(env.road.vehicles), 1)
                        self.assertEqual(env.action_space.shape, (1,))
                        self.assertEqual(info['track_generator'], generator)
                        self.assertGreater(info['track_length'], 100)
                        # Consultar o RoadNetwork real nas emendas, incluindo a última.
                        for i, lane in enumerate(env.track_lanes):
                            following = env.track_lanes[(i + 1) % len(env.track_lanes)]
                            np.testing.assert_allclose(lane.position(lane.length, 0),
                                                       following.position(0, 0), atol=1e-5)
                            for s in (0, lane.length / 2, lane.length):
                                position = lane.position(s, 0)
                                index = env.road.network.get_closest_lane_index(position, lane.heading_at(s))
                                self.assertTrue(env.road.network.get_lane(index).on_lane(position))
                        env.action_space.seed(seed)
                        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                        self.assertTrue(env.observation_space.contains(obs))
                        self.assertTrue(np.isfinite(reward))
            finally:
                env.close()

    def test_seed_reproduces_track_spawn_and_sequence(self):
        for generator in self.generators():
            env = self.make_env(generator)
            try:
                env.reset(seed=42)
                first, spawn, heading = fingerprint(env), env.vehicle.position.copy(), env.vehicle.heading
                sequence = [first]
                for _ in range(4):
                    env.reset()
                    sequence.append(fingerprint(env))
                self.assertEqual(len(set(sequence)), 5)
                env.reset(seed=42)
                self.assertEqual(first, fingerprint(env))
                np.testing.assert_allclose(spawn, env.vehicle.position)
                self.assertEqual(heading, env.vehicle.heading)
                for expected in sequence[1:]:
                    env.reset()
                    self.assertEqual(expected, fingerprint(env))
            finally:
                env.close()

    def test_complete_laps_through_track_closure(self):
        for generator in self.generators():
            # A observação Kinematics acelera o teste longo; geração/spawn/reward/
            # física são os mesmos. O teste anterior usa OccupancyGrid padrão.
            env = self.make_env(generator, observation={'type': 'Kinematics'})
            try:
                for seed in (7, 42, 2026):
                    with self.subTest(generator=generator, seed=seed):
                        env.reset(seed=seed)
                        starts = np.r_[0, np.cumsum([lane.length for lane in env.track_lanes])]
                        previous = env.track_lanes[0].length / 2
                        wraps = 0
                        limit = math.ceil(env.track_length / env.vehicle.speed * env.config['policy_frequency'] * 1.2)
                        for _ in range(limit):
                            _, reward, terminated, truncated, _ = env.step(follower_action(env))
                            self.assertFalse(terminated, 'O seguidor saiu da pista durante a volta.')
                            self.assertFalse(truncated)
                            self.assertTrue(np.isfinite(reward))
                            index = env.track_lane_indices.index(env.vehicle.lane_index)
                            longitudinal, lateral = env.vehicle.lane.local_coordinates(env.vehicle.position)
                            current = starts[index] + longitudinal
                            if previous > 0.75 * env.track_length and current < 0.25 * env.track_length:
                                wraps += 1
                            previous = current
                            if wraps and current >= env.track_lanes[0].length / 2:
                                break
                        self.assertGreaterEqual(wraps, 1, 'Não completou o circuito.')
            finally:
                env.close()

    def test_gymnasium_api(self):
        env = gym.make(ENV_ID)
        try:
            check_env(env.unwrapped, skip_render_check=True)
        finally:
            env.close()

    @unittest.skipUnless(HAS_PROCEDURAL, 'Clone procedural-tracks não encontrado ao lado do projeto.')
    def test_procedural_does_not_change_global_random(self):
        state = random.getstate()
        env = self.make_env('procedural')
        try:
            env.reset(seed=8)
            self.assertEqual(state, random.getstate())
        finally:
            env.close()

    def test_generation_fallback_keeps_reset_usable(self):
        env = self.make_env(max_generation_attempts=1)
        try:
            with patch('pista_aleatoria.ConvexHull', side_effect=ValueError('formato inválido forçado')):
                observation, info = env.reset(seed=17)
                self.assertEqual(info['track_generator'], 'native-fallback')
                self.assertIn('reserva', info['generation_warning'])
                self.assertTrue(env.vehicle.on_road)
                self.assertTrue(env.observation_space.contains(observation))
                geometry = fingerprint(env)
                env.reset(seed=17)
                self.assertEqual(geometry, fingerprint(env))
        finally:
            env.close()

    def test_invalid_settings_fail_clearly(self):
        for config in ({'other_vehicles': 1}, {'controlled_vehicles': 2},
                       {'lane_width': 0}, {'initial_speed': 20},
                       {'track_generator': 'unknown'}, {'max_generation_attempts': 0},
                       {'track_generator': 'procedural', 'procedural_path': None}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                RandomRacetrackEnv(config=config)


if __name__ == '__main__':
    unittest.main(verbosity=2)

