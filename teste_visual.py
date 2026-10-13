"""Demonstração visual de pistas. O seguidor geométrico não aprende."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from pista_aleatoria import ENV_ID


def follower_action(env):
    """Pure pursuit: apenas geometria da pista, sem treino ou pesos de rede."""
    car = env.vehicle
    target = env.point_ahead(max(6.0, car.speed * 0.8))
    rear = car.position - car.LENGTH / 2 * np.array([np.cos(car.heading), np.sin(car.heading)])
    delta = target - rear
    alpha = np.arctan2(delta[1], delta[0]) - car.heading
    steering = np.arctan2(2 * car.LENGTH * np.sin(alpha), np.linalg.norm(delta))
    low, high = env.action_type.steering_range
    normalized = 2 * (steering - low) / (high - low) - 1
    return np.array([np.clip(normalized, -1, 1)], dtype=np.float32)


def overview(env):
    """Enquadra a pista inteira sem mudar a física ou a observação do carro."""
    if env.viewer is None:
        env.render()
    bounds = env.track_bounds
    center = bounds.mean(axis=0)
    size = bounds[1] - bounds[0] + 24
    env.viewer.observer_vehicle = SimpleNamespace(position=center)
    env.viewer.sim_surface.scaling = float(min(env.config["screen_width"] / size[0],
                                              env.config["screen_height"] / size[1]))


def main():
    parser = argparse.ArgumentParser(description="Pistas aleatórias no HighwayEnv (sem rede neural).")
    parser.add_argument("--gerador", choices=("native", "procedural"), default="native")
    parser.add_argument("--procedural-path", type=Path,
                        default=Path(__file__).resolve().parent.parent / "procedural-tracks")
    parser.add_argument("--controle", choices=("seguidor", "aleatorio"), default="seguidor")
    parser.add_argument("--pistas", type=int, default=3)
    parser.add_argument("--passos", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sem-janela", action="store_true")
    parser.add_argument("--seguir-carro", action="store_true")
    parser.add_argument("--salvar-imagens", type=Path, default=None)
    args = parser.parse_args()
    if args.pistas < 1 or args.passos < 1:
        parser.error("--pistas e --passos precisam ser positivos.")
    if args.salvar_imagens and not args.sem_janela:
        parser.error("Use --salvar-imagens junto com --sem-janela.")
    mode = "rgb_array" if args.sem_janela and args.salvar_imagens else None if args.sem_janela else "human"
    config = {"track_generator": args.gerador, "procedural_path": str(args.procedural_path),
              "offscreen_rendering": args.sem_janela, "screen_width": 800, "screen_height": 700}
    env = gym.make(ENV_ID, render_mode=mode, config=config)
    try:
        for episode in range(args.pistas):
            # Sem reusar a seed nos resets seguintes: a sequência continua.
            _, info = env.reset(seed=args.seed if episode == 0 else None)
            base = env.unwrapped
            if args.seed is not None:
                env.action_space.seed(args.seed + episode)
            if mode and not args.seguir_carro:
                overview(base)
            if base.done:
                break
            print(f"Pista {episode + 1}: {info['track_generator']}, "
                  f"{info['track_length']:.1f} m, {info['generation_attempts']} tentativa(s), "
                  f"carro na pista: {base.vehicle.on_road}")
            if info['generation_warning']:
                print(f"  Aviso: {info['generation_warning']}")
            if args.salvar_imagens:
                import pygame
                args.salvar_imagens.mkdir(parents=True, exist_ok=True)
                frame = env.render()
                surface = pygame.surfarray.make_surface(np.moveaxis(frame, 0, 1))
                pygame.image.save(surface, str(args.salvar_imagens / f"pista_{episode + 1:02d}.png"))
            for step in range(args.passos):
                if base.done:
                    return
                action = follower_action(base) if args.controle == "seguidor" else env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                if base.done:
                    return
                if terminated or truncated:
                    print(f"  Episódio encerrado no passo {step + 1}; gerando outra pista.")
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()

