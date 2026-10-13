# Pistas aleatórias — MVP do highway-ia

Implementado e validado com o Python 3.9.13 do projeto, HighwayEnv 1.10.2, Gymnasium 1.1.1, NumPy 2.0.2, SciPy 1.13.1 e Pygame 2.6.1. As dependências já estão no seu `.venv`: não é necessário instalar pacotes.

## Etapa 1: abrir e rodar no VS Code

1. Abra a pasta `C:\Users\schoo\highway-ia` no VS Code (Arquivo > Abrir Pasta).
2. Pressione `Ctrl+Shift+P`, escolha **Python: Select Interpreter** e selecione `C:\Users\schoo\highway-ia\.venv\Scripts\python.exe`. Se não aparecer, use **Enter interpreter path**.
3. Abra um terminal PowerShell no VS Code e execute:

```powershell
Set-Location 'C:\Users\schoo\highway-ia'
.\.venv\Scripts\python.exe .\teste_visual.py
```

A janela mostra a pista inteira e um carro que a acompanha. São três pistas, com até 200 passos por pista (aproximadamente 40 segundos de simulação por pista). Uma nova pista aparece no episódio seguinte. Feche a janela para sair; `Ctrl+C` no terminal também interrompe o teste e fecha o ambiente.

O comando usa diretamente o Python do `.venv`, portanto não depende de executar `Activate.ps1` nem de mudar a política de execução do PowerShell. O `main.py` original continua disponível para seu teste anterior.

Para uma sequência reproduzível e episódios mais longos:

```powershell
.\.venv\Scripts\python.exe .\teste_visual.py --seed 42 --pistas 5 --passos 600
```

Para aproximar a câmera do carro:

```powershell
.\.venv\Scripts\python.exe .\teste_visual.py --seguir-carro
```

## Etapa 2: usar o procedural-tracks

O clone real foi encontrado em `C:\Users\schoo\procedural-tracks`. Rode:

```powershell
.\.venv\Scripts\python.exe .\teste_visual.py --gerador procedural --seed 42
```

O programa procura o clone ao lado de `highway-ia`. Se o mover, informe o caminho:

```powershell
.\.venv\Scripts\python.exe .\teste_visual.py --gerador procedural --procedural-path 'C:\Users\schoo\procedural-tracks'
```

A integração importa as funções `random_points`, `get_track_points` e `shape_track` do `main.py` desse repositório, sem executar sua janela ou modificar seus arquivos. Os pontos são convertidos de pixels para metros, simplificados quando estão próximos demais e arredondados com as retas e curvas do HighwayEnv. O formato pode incluir curvas para os dois lados.

A spline desenhada pelo projeto original (`smooth_track`) não é copiada: ela retorna coordenadas inteiras em pixels. Neste MVP, o contorno recebe arcos tangentes, que se conectam corretamente à física do HighwayEnv. Portanto, a pista adaptada pode diferir do desenho original.

Formatos com curvas muito fechadas, cruzamentos ou trechos próximos demais são rejeitados. Até 30 tentativas são feitas. Se todas falharem, uma pista simples de reserva é gerada; o terminal avisa e `info['track_generator']` passa a ser `native-fallback`. Um caminho incorreto para o clone produz um erro claro, pois não há como importar o gerador.

Referência do gerador: https://github.com/juangallostra/procedural-tracks

## O que mudou

| Arquivo | Função |
| --- | --- |
| `pista_aleatoria.py` | Subclasse de `RacetrackEnv`, geradores, validação, posicionamento do carro e registro `racetrack-random-v0`. |
| `teste_visual.py` | Janela, visão da pista inteira, controlador demonstrativo e opções de teste. |
| `test_pistas.py` | Testes de geração, sementes, nascimento, voltas completas, reserva e API Gymnasium. |
| `README_PISTAS.md` | Este guia. |

A cada `reset()`, `_make_road()` constrói um novo `RoadNetwork` com uma sequência fechada de `StraightLane` e `CircularLane`. A posição e a direção das emendas coincidem, incluindo a última com a primeira. `_make_vehicles()` coloca um único carro no centro da maior reta, com direção calculada pela própria faixa e velocidade inicial de 8 m/s.

O ambiente herda ações, observações, recompensa, física e encerramento de `racetrack-v0`. O espaço de ações padrão continua `Box(-1, 1, (1,))`: um comando de direção, sem acelerador. O MVP usa uma faixa de 8 m de largura, um carro e nenhum tráfego. Não foram alterados a `.venv`, a instalação do HighwayEnv, o `procedural-tracks` ou seu `main.py` existente.

O gerador usa `self.np_random`; a integração procedural tem uma instância isolada de `random.Random`. `reset(seed=42)` reproduz pista e carro; os resets seguintes **sem seed** continuam a sequência e criam outras pistas. Repetir a mesma seed em todo reset repetirá a mesma pista.

Exemplo para reutilizar o ambiente em outra etapa:

```python
import gymnasium as gym
import pista_aleatoria  # registra o ambiente personalizado

env = gym.make('racetrack-random-v0', config={'other_vehicles': 0})
try:
    observacao, info = env.reset(seed=42)
    observacao, recompensa, terminou, truncado, info = env.step(env.action_space.sample())
    observacao, info = env.reset()  # outra pista
finally:
    env.close()
```

## Verificar sem janela

```powershell
.\.venv\Scripts\python.exe -m unittest test_pistas -v
.\.venv\Scripts\python.exe .\teste_visual.py --sem-janela --pistas 3 --passos 10 --seed 42
```

Para salvar imagens das pistas pelo renderizador do HighwayEnv:

```powershell
.\.venv\Scripts\python.exe .\teste_visual.py --sem-janela --pistas 3 --passos 1 --seed 42 --salvar-imagens '.\imagens_pistas'
```

Os testes verificam 30 sementes de cada gerador (60 pistas), reprodução da sequência, observações e recompensas finitas, carro sobre a pista, conexão dos trechos, seis voltas completas em três sementes por gerador e o caso de todas as tentativas serem rejeitadas. Se o clone não estiver ao lado do projeto, os testes opcionais dele são ignorados. O teste das voltas usa observação Kinematics para acelerar a execução; a geração e a física permanecem iguais. O teste de geração usa OccupancyGrid padrão.

A execução automatizada nesta sessão exigiu uma adaptação temporária de `Path.resolve` porque a proteção do Windows negava a consulta de metadados de pastas, mesmo após liberar os caminhos. A adaptação só calculou caminhos absolutos durante a execução dos testes; não foi incluída no projeto e não modificou a biblioteca. Os comandos acima são para seu terminal normal do VS Code.

## Etapa futura: rede neural

O modo padrão `--controle seguidor` usa **pure pursuit**, um cálculo geométrico que conhece a pista. Ele serve para demonstrar que o circuito é percorrível; não treina, não aprende e não usa rede neural.

Você também pode comparar com comandos aleatórios:

```powershell
.\.venv\Scripts\python.exe .\teste_visual.py --controle aleatorio --pistas 10
```

Recompensas aparecendo no terminal indicam avaliação da ação, e não aprendizagem por si só. Treinamento, checkpoints e avaliação da rede ficam para a próxima etapa. A recompensa atual continua sendo a de centralização do racetrack; este MVP não adiciona recompensa por progresso ou contagem de voltas para treino.
