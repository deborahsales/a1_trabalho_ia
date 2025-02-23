import ale_py
import os
import gymnasium as gym
import cv2
from collections import deque
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import datetime

# Criar ambiente Atari
games = ["ALE/Breakout-v5", "ALE/Pacman-v5"]
chosen_game = games[0]
env = gym.make(chosen_game, render_mode='rgb_array')
env.reset()

def preprocess_frame(frame):
    """Converte para tons de cinza, redimensiona e normaliza"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Converte para escala de cinza
    resized = cv2.resize(gray, (84, 84))  # Redimensiona para 84x84
    normalized = resized / 255.0  # Normaliza para [0,1]
    return normalized  # Mantém (84, 84) sem dimensão extra

def stack_frames(frames, frame, is_new_episode):
    """Empilha 4 quadros para capturar movimento"""
    if is_new_episode or frames is None:
        frames = deque([frame] * 4, maxlen=4)  # Preenche os 4 frames iniciais
    else:
        frames.append(frame)

    stacked_state = np.stack(frames, axis=0)  # Agora tem shape (4, 84, 84)
    return stacked_state, frames


# Teste de preprocessamento
obs, _ = env.reset()  # Desempacotar a tupla corretamente
processed_frame = preprocess_frame(obs)

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

# Criar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(env.action_space.n).to(device)

class MCTS:
    def __init__(self, env, simulations=100):
        self.env = env
        self.simulations = simulations

    def search(self, state):
        best_action = random.choice(range(self.env.action_space.n))
        return best_action

mcts = MCTS(env)

def select_action(state, model, mcts, epsilon=0.1):
    if np.random.rand() < epsilon:
        return mcts.search(state)  # MCTS para explorar
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            return model(state_tensor).argmax().item()  # DQN para exploração

# Parâmetros do treinamento
num_episodes = 3
gamma = 0.99
learning_rate = 0.0001
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
buffer_size = 100000
batch_size = 32

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
memory = deque(maxlen=buffer_size)
rewardsX = []  # Lista para armazenar recompensas

# Loop de treinamento
for episode in range(num_episodes):
    total_reward = 0
    state, _ = env.reset()
    state, frames = stack_frames(None, preprocess_frame(state), True)
    done = False
    total_reward = 0

    while not done:
        action = select_action(state, model, mcts, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state, frames = stack_frames(frames, preprocess_frame(next_state), False)

        # Exibir o jogo em uma janela enquanto treina
        frame_rgb = env.render()
        if frame_rgb is not None:
            cv2.imshow("Atari Training", frame_rgb)
            cv2.waitKey(1)  # Pequeno delay para atualização da tela

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            q_values = model(states).gather(1, actions).squeeze(1)
            next_q_values = model(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewardsX.append(total_reward)
    print(f"Episode {episode + 1}: Reward {total_reward}")

# Criar diretório para salvar modelos
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

# Corrigir nome do jogo removendo barras
safe_game_name = chosen_game.replace("/", "_")

# Gerar timestamp no formato AAAA-MM-DD_HH-MM-SS
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Criar nomes de arquivo com timestamp
filename_state_dict = f"dqn_model_{safe_game_name}_{num_episodes}episodes_{timestamp}.pth"
filename_complete = f"dqn_model_complete_{safe_game_name}_{num_episodes}episodes_{timestamp}.pth"

# Caminho completo
filepath_state_dict = os.path.join(save_dir, filename_state_dict)
filepath_complete = os.path.join(save_dir, filename_complete)

# Salvar os modelos
torch.save(model.state_dict(), filepath_state_dict)
torch.save(model, filepath_complete)

# Criar diretório para salvar gráficos
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Criar nome do arquivo do gráfico
plot_filename = f"reward_plot_{safe_game_name}_{num_episodes}episodes_{timestamp}.png"
plot_filepath = os.path.join(plot_dir, plot_filename)

# Criar e salvar o gráfico
plt.plot(rewardsX)
plt.xlabel("Episódios")
plt.ylabel("Recompensa")
plt.title("Evolução da Recompensa ao longo do Treinamento")
plt.savefig(plot_filepath)  # Salvar em arquivo
plt.close() 