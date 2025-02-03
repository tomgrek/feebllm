# text generating model
# given some token and a mask, predict the next token

import torch
import string
import random

tokens = string.ascii_lowercase + "#" # Add padding token

print(tokens)

# Context window is 10, for next token prediction we can generate up to 9; for PPO upto 7
def generate_data(n_samples, max_len=9, for_ppo=False, max_seq=10):
    samps = []
    max_seq -= 1 # at least one padding token
    for i in range(n_samples):
        seq_len = random.randint(1, max_len)
        seq = [random.randint(0, len(tokens) - 1) for _ in range(seq_len)]
        mask = [1] * seq_len + [0] * (max_seq - seq_len) + [0]
        # Target should be the sum of the tokens mod 26, for basic model.
        # For PPO, the target should be the sum of the tokens mod 26, the next token should be that sum + 1 mod 26, and the next one should be that sum + 2 mod 26
        target = sum(seq) % 26
        if for_ppo:
            target = [target, (target + 1) % 26, (target + 2) % 26]
        seq += [26] * (max_seq - seq_len) + [26]
        samps.append((seq, target, mask))
    return samps

examples = generate_data(2000, max_len=9, max_seq=10)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(len(tokens), 16)
        self.lstm = torch.nn.LSTM(16, 128, batch_first=True)
        self.fc = torch.nn.Linear(128, len(tokens))
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = x * mask.unsqueeze(2)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
model = Model()

def train(model, data, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for seq, target, mask in data:
            seq = torch.tensor([seq])
            target = torch.tensor([target] * seq.size(1))
            mask = torch.tensor([mask])
            output = model(seq, mask)
            output = output.squeeze(0)
            target = target.view(-1)
            mask = mask.view(-1)
            loss = torch.nn.functional.cross_entropy(output[mask == 1], target[mask == 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

#train(model, examples)
test1 = model(torch.tensor([[10, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
print(f"Prediction: {test1.squeeze(0)[1].argmax().item()}") # hope for 10
test2 = model(torch.tensor([[12, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])) # 10 + 1
print(f"Prediction: {test2.squeeze(0)[2].argmax().item()}") # hope for 13
test3 = model(torch.tensor([[14, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])) # 10 + 1 + 2
print(f"Prediction: {test3.squeeze(0)[3].argmax().item()}") # hope for 17

# Now modify it, the example data should change so that the next token is the sum (still), the next one is that sum + 1 (%26) and again + 1 for the next one
# Note (and this is NO LONGER true currently as well) the output shape should be larger, ie max seq len needs to reduce by 3 (and by 1 currently)
# Use PPO to learn this, starting from the trained model.


class PPO:
    def __init__(self, model, data):
        self.actor = model
        self.critic = Model()
        if self.critic:
            self.critic.load_state_dict(self.actor.state_dict())
        if self.actor:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.1)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.1)
            self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.1)
            self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.1)
        self.gamma = 0.99
        self.eps_clip = 0.5
        self.K_epochs = 4
        self.data = data

        self.cov_mat = torch.eye(len(tokens))
    
    def get_batch(self, bs=8):
        batch = random.sample(self.data, bs)
        seq, target, mask = zip(*batch)
        batch_seq = torch.tensor([seq]).reshape(bs, len(seq[0]), 1)
        batch_target = torch.tensor([target]).reshape(bs, len(target[0]), 1)
        batch_mask = torch.tensor([mask]).reshape(bs, len(mask[0]), 1)
        return batch_seq, batch_target, batch_mask
    
    def eval(self):
        batch = self.get_batch(2)
        output = self.actor(batch[0].squeeze(-1), batch[2].squeeze(-1)).argmax(dim=-1)
        #import ipdb; ipdb.set_trace()
        print(output, batch[1], batch[2].squeeze())


    
    def train(self, data, timesteps=10000):
        t = 0
        while t < timesteps:
            t += 1
            seq, target, mask = self.get_batch()

            # Get action probabilities and values from the actor and critic
            # seq and mask are (bs, seq_len, 1) and target is (bs, seq_len, 1)
            with torch.autograd.set_detect_anomaly(True):
                action_probs = self.actor(seq.squeeze(-1), mask.squeeze(-1)) # (bs, seq_len, n_tokens) aka V
                old_probs = self.critic(seq.squeeze(-1), mask.squeeze(-1)) # (bs, seq_len, n_tokens)

                # get log probs. First get the first predicted token index, using the mask
                first_predicted_token = (mask==0).float().argmax(1)  # shape: bs, 1
                # gather only the 3 tokens starting with first predicted token index
                pred_mask = torch.arange(0, 3) + first_predicted_token.unsqueeze(-1)
                # get the log probs for these tokens
                pred_mask = pred_mask.squeeze(1).unsqueeze(-1)  # shape: bs, 3, 1
                pred_mask = pred_mask.repeat(1, 1, len(tokens))  # shape: bs, 3, n_tokens
                action_probs = action_probs.gather(1, pred_mask)

                old_probs = old_probs.gather(1, pred_mask)

                # Calculate the ratio
                ratio = torch.exp(action_probs - old_probs) # action_probs / old_probs

                # Calculate the rewards
                selected_action = action_probs.argmax(-1)
                dist = torch.distributions.Categorical(action_probs)
                selected_action = dist.sample()
                #import ipdb; ipdb.set_trace()
                # log_probs = dist.log_prob(selected_action)
                rewards = []
                for i in range(target.size(0)): # bs
                    discount_factor = 1.0#self.gamma
                    ep_reward = []
                    for j in range(target.size(1) - 1, -1, -1): # seq_len
                        ep_reward.insert(0, (target[i, j] == selected_action[i, j]).float().item() * discount_factor)
                        discount_factor *= self.gamma
                    rewards.append(ep_reward)
                
                rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(-1)
                #advantage = rewards - action_probs


                # Calculate the surrogate loss
                # import ipdb; ipdb.set_trace()
                # TODO RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [128, 27]]
            
                advantages = rewards.clone() - old_probs.clone().detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                #import ipdb; ipdb.set_trace()
                
                critic_loss = torch.nn.MSELoss()(old_probs.clone().gather(-1, pred_mask).mean(dim=-1).unsqueeze(-1), rewards.clone())
            
            
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()
            
            

            print(f"Step {t}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}")

            

if __name__ == "__main__":
    examples = generate_data(100, max_seq=10, max_len=7, for_ppo=True)
    ppo = PPO(model, examples)
    ppo.train(examples, timesteps=2000)
    ppo.eval()