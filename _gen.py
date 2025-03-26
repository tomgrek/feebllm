import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        #self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.fc(x)#self.sm(self.fc(x))

class PPO:
    def __init__(self, model, lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.loss_fn = nn.MSELoss()

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def update(self, states, actions, rewards, dones, next_state):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        old_values = self.model(states).detach()
        next_value = self.model(next_state).detach()
        returns = self.compute_returns(rewards, dones, next_value)
        returns = torch.tensor(returns, dtype=torch.float32).detach()

        for _ in range(60):  # Update for 60 epochs
            values = self.model(states)
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages

            # Assuming the model outputs log probabilities
            log_probs = self.model(states)
            old_log_probs = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1).detach()
            log_probs = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.loss_fn(values, returns.unsqueeze(-1))
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Example usage
input_dim = 4
output_dim = 1
model = SimpleModel(input_dim, output_dim)
ppo = PPO(model)


def generate_episodes():
    states = [
        [0.2, 0.2, 0.3, 0.4],
        [0.1, 0.3, 0.33, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.5],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
        [0.4, 0.5, 0.6, 0.7],
        [0.5, 0.6, 0.7, 0.8],

        [0.5, 0.6, 0.7, 0.8],
        [0.4, 0.5, 0.6, 0.7],
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.24, 0.23, 0.4],
    ]
    actions = [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9] + [0.7, 0.1, 0.4, 0.3, 0.9, 0.1, 0.2, 0.3, 0.9]
    rewards = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] + [0.0]*9
    dones = [1]*len(rewards)# [0, 0, 0, 0, 1]
    # actions = np.random.rand(5, output_dim)
    # rewards = np.random.rand(5)
    #dones = np.random.randint(0, 2, size=5)
    next_state = np.random.rand(input_dim)
    # next_state = [
    #     [0.5, 0.6, 0.7, 0.8],
    #     [0.6, 0.7, 0.8, 0.9],
    #     [0.7, 0.8, 0.9, 1.0],
    #     [0.8, 0.9, 1.0, 1.1],
    #     [0.9, 1.0, 1.1, 1.2],
    # ]
    return states, actions, rewards, dones, next_state

# states = np.random.rand(5, input_dim)
# actions = np.random.rand(5, output_dim)
# rewards = np.random.rand(5)
# dones = np.random.randint(0, 2, size=5)
# next_state = np.random.rand(input_dim)

episodes = generate_episodes()
for _ in range(3000):
    ppo.update(episodes[0], episodes[1], episodes[2], episodes[3], episodes[4])
print(model(torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)))
print(model(torch.tensor([[0.3, 0.4, 0.5, 0.6]], dtype=torch.float32)))