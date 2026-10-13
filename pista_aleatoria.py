"""Pistas procedurais para HighwayEnv 1.10.2, sem alterar site-packages."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from highway_env.envs.racetrack_env import RacetrackEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork


ENV_ID = "racetrack-random-v0"


def _load_procedural(repository: str):
    """Importa o clone local, sem executar main() nem mudar random global."""
    root = Path(repository).expanduser().resolve()
    for name in ("constants.py", "main.py"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Arquivo ausente: {root / name}")

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    constants = load("_highway_ia_track_constants", root / "constants.py")
    previous = sys.modules.get("constants")
    try:
        # O repositório usa 'from constants import *'. Não importar nosso main.py.
        sys.modules["constants"] = constants
        module = load("_highway_ia_procedural", root / "main.py")
    finally:
        if previous is None:
            sys.modules.pop("constants", None)
        else:
            sys.modules["constants"] = previous
    module.rn = random.Random()
    return module


def _rounded_lanes(points, rng, width, speed_limit):
    """Arredonda vértices com arcos tangentes; aceita curvas para ambos os lados."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("A pista precisa de pelo menos quatro pontos 2D.")
    if not np.isfinite(points).all():
        raise ValueError("Coordenadas inválidas.")
    span = np.ptp(points, axis=0).max()
    if span < 1e-6:
        raise ValueError("Pontos coincidentes.")
    points = (points - points.mean(axis=0)) * (160.0 / span)
    # Os pontos do procedural estão em pixels e incluem detalhes menores que
    # o carro. Remover vértices muito próximos antes de criar curvas em metros.
    while len(points) > 4:
        edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        shortest = int(np.argmin(edge_lengths))
        if edge_lengths[shortest] >= 18:
            break
        points = np.delete(points, (shortest + 1) % len(points), axis=0)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < 2:
        raise ValueError("Vértices muito próximos.")
    outgoing = edges / lengths[:, None]
    incoming = np.roll(outgoing, 1, axis=0)
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    turns = np.arctan2(cross, np.sum(incoming * outgoing, axis=1))
    if np.any(np.abs(turns) > np.deg2rad(140)):
        raise ValueError("Curva excessivamente fechada.")
    radii = rng.uniform(15.0, 28.0, len(points))
    distances = radii * np.tan(np.abs(turns) / 2)
    # Reservar espaço entre duas curvas e garantir área suficiente para o carro.
    scale = max(1.0, np.max((distances + np.roll(distances, -1) + 12) / lengths))
    if scale > 3:
        raise ValueError("Formato estreito demais para este MVP.")
    points *= scale
    entries = points - incoming * distances[:, None]
    exits = points + outgoing * distances[:, None]
    lanes = []
    kwargs = dict(width=width, speed_limit=speed_limit,
                  line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS))
    for i in range(len(points)):
        # Reta que termina na curva do próximo vértice.
        j = (i + 1) % len(points)
        lanes.append(StraightLane(exits[i], entries[j], **kwargs))
        if abs(turns[j]) > 1e-6:
            sign = float(np.sign(turns[j]))
            normal = np.array([-incoming[j, 1], incoming[j, 0]])
            center = entries[j] + sign * radii[j] * normal
            phase = float(np.arctan2(*(entries[j] - center)[::-1]))
            lanes.append(CircularLane(center, float(radii[j]), phase,
                                      phase + float(turns[j]),
                                      clockwise=bool(sign > 0), **kwargs))
    # Começar no meio da maior reta, longe de uma mudança de trecho.
    first = max((i for i, lane in enumerate(lanes) if isinstance(lane, StraightLane)),
                key=lambda i: lanes[i].length)
    return lanes[first:] + lanes[:first]


def _validate_lanes(lanes, width):
    """Rejeita lacunas, cruzamentos e aproximações perigosas entre trechos."""
    samples, distances = [], []
    offset = 0.0
    for i, lane in enumerate(lanes):
        following = lanes[(i + 1) % len(lanes)]
        if lane.length < 0.01 or not np.isfinite(lane.length):
            raise ValueError("Trecho inválido.")
        if not np.allclose(lane.position(lane.length, 0), following.position(0, 0), atol=1e-5):
            raise ValueError("Lacuna entre trechos.")
        delta = lane.heading_at(lane.length) - following.heading_at(0)
        if abs(np.arctan2(np.sin(delta), np.cos(delta))) > 1e-5:
            raise ValueError("Curvas sem tangência.")
        s = np.linspace(0, lane.length, max(2, int(np.ceil(lane.length / 2))), endpoint=False)
        samples.extend(lane.position(float(x), 0) for x in s)
        distances.extend(offset + s)
        offset += lane.length
    samples, distances = np.array(samples), np.array(distances)
    gap = np.abs(distances[:, None] - distances[None, :])
    gap = np.minimum(gap, offset - gap)
    separation = np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1)
    # O limite extra cobre a distância entre amostras de até 2 m.
    if np.any((gap > 3 * width) & (separation < width + 4)):
        raise ValueError("Pista cruza ou se aproxima demais de si mesma.")
    return samples


class RandomRacetrackEnv(RacetrackEnv):
    """Uma pista nova em reset(); reset(seed=N) reproduz a mesma pista."""

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({"other_vehicles": 0, "controlled_vehicles": 1,
                       "track_generator": "native", "procedural_path": None,
                       "lane_width": 8.0, "initial_speed": 8.0,
                       "max_generation_attempts": 30})
        return config

    def _make_road(self):
        if self.config["controlled_vehicles"] != 1 or self.config["other_vehicles"] != 0:
            raise ValueError("Este MVP suporta um carro controlado e nenhum tráfego.")
        width = float(self.config["lane_width"])
        speed = float(self.config["initial_speed"])
        if not 4 <= width <= 10 or not 0 < speed <= 10:
            raise ValueError("Use lane_width entre 4 e 10 e initial_speed entre 0 e 10.")
        generator = self.config["track_generator"]
        if generator not in ("native", "procedural"):
            raise ValueError("track_generator deve ser 'native' ou 'procedural'.")
        attempts = self.config["max_generation_attempts"]
        if not isinstance(attempts, int) or attempts < 1:
            raise ValueError("max_generation_attempts deve ser um inteiro positivo.")
        if generator == "procedural":
            path = self.config["procedural_path"]
            if not path:
                raise ValueError("Informe procedural_path com o caminho do clone local.")
            if getattr(self, "_procedural_path", None) != path:
                self._procedural = _load_procedural(path)
                self._procedural_path = path

        self.track_generator_used = generator
        self.generation_warning = None
        last_error = None
        for attempt in range(attempts):
            try:
                if generator == "native":
                    count = int(self.np_random.integers(5, 10))
                    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
                    angles += self.np_random.uniform(-0.15, 0.15, count)
                    radii = self.np_random.uniform(65, 95, count)
                    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
                    points[:, 1] *= self.np_random.uniform(0.7, 1.1)
                    points = points[ConvexHull(points).vertices]
                else:
                    module = self._procedural
                    module.rn.seed(int(self.np_random.integers(0, 2**32)))
                    cloud = module.random_points()
                    hull = ConvexHull(cloud)
                    points = module.shape_track(module.get_track_points(hull, cloud))
                lanes = _rounded_lanes(points, self.np_random, width, self.config["speed_limit"])
                samples = _validate_lanes(lanes, width)
                break
            except (ValueError, ZeroDivisionError, QhullError) as error:
                last_error = error
        else:
            # Reserva determinística e também aleatória: mantém reset utilizável
            # quando todos os formatos complexos são rejeitados pela validação.
            height = self.np_random.uniform(100, 140)
            points = [[0, 0], [160, 0], [160, height], [0, height]]
            lanes = _rounded_lanes(points, self.np_random, width, self.config["speed_limit"])
            samples = _validate_lanes(lanes, width)
            self.track_generator_used = "native-fallback"
            self.generation_warning = f"{attempts} formatos rejeitados; usando pista de reserva: {last_error}"

        network = RoadNetwork()
        self.track_lanes = lanes
        self.track_lane_indices = [(f"p{i}", f"p{(i + 1) % len(lanes)}", 0) for i in range(len(lanes))]
        for index, lane in zip(self.track_lane_indices, lanes):
            network.add_lane(index[0], index[1], lane)
        self.track_length = float(sum(lane.length for lane in lanes))
        self.track_bounds = np.array([samples.min(axis=0), samples.max(axis=0)])
        self.generation_attempts = attempt + 1
        self.lane = lanes[0]
        self.road = Road(network=network, np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _make_vehicles(self):
        car = self.action_type.vehicle_class.make_on_lane(
            self.road, self.track_lane_indices[0],
            longitudinal=self.track_lanes[0].length / 2,
            speed=float(self.config["initial_speed"]))
        self.controlled_vehicles = [car]
        self.road.vehicles.append(car)
        if not car.on_road:
            raise RuntimeError("O carro não nasceu sobre a pista.")

    def _info(self, obs, action=None):
        info = super()._info(obs, action)
        info.update(track_generator=self.track_generator_used,
                    track_length=self.track_length,
                    generation_attempts=self.generation_attempts,
                    requested_track_generator=self.config["track_generator"],
                    generation_warning=self.generation_warning)
        return info

    def point_ahead(self, distance):
        """Ponto adiante do carro; usado apenas pelo controlador demonstrativo."""
        index = self.track_lane_indices.index(self.vehicle.lane_index)
        longitudinal, _ = self.vehicle.lane.local_coordinates(self.vehicle.position)
        longitudinal = max(0.0, float(longitudinal)) + distance
        for _ in range(len(self.track_lanes) + 1):
            lane = self.track_lanes[index]
            if longitudinal <= lane.length:
                return lane.position(longitudinal, 0)
            longitudinal -= lane.length
            index = (index + 1) % len(self.track_lanes)
        raise ValueError("Distância maior que uma volta.")


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="pista_aleatoria:RandomRacetrackEnv")


