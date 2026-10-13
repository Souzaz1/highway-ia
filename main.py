
import gymnasium as gym
import highway_env

# Criando a pista e o simulador
env = gym.make(
    "racetrack-v0",
    render_mode="human",
    config={"other_vehicles": 0}
)

# Inicializando o ambiente
observacao, info = env.reset()

print("Espaço de observações:", env.observation_space)
print("Espaço de ações:", env.action_space)

# Teste com comandos aleatórios
for passo in range(1000):

    acao = env.action_space.sample()

    observacao, recompensa, terminou, truncado, info = env.step(acao)

    print(f"Passo {passo} | Recompensa: {recompensa:.2f}")

    if terminou or truncado:
        observacao, info = env.reset()

env.close()
