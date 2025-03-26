# text generating model
# given some token and a mask, predict the next token

import torch
from torch.distributions import Categorical
import string
import random

tokens = string.ascii_lowercase + "#" # Add padding token

print(tokens)

# Context window is 10, for next token prediction we can generate up to 9; for PPO upto 7
def generate_data(n_samples, max_len=6, for_ppo=False, total_length=10):
    samps = []
    max_seq = total_length - 1 # at least one padding token
    for i in range(n_samples):
        seq_len = random.randint(1, max_len)
        seq = [random.randint(0, len(tokens) - 2) for _ in range(seq_len)] # 0-25 and hash is 26
        mask = [1] * seq_len + [0] * (total_length - seq_len)
        # Target should be the sum of the tokens mod 26, for basic model.
        # For PPO, the target should be the sum of the tokens mod 26, the next token should be that sum + 1 mod 26, and the next one should be that sum + 2 mod 26
        target = [seq[seq_len-1]]#sum(seq) % 26
        if for_ppo:
            target = [target, (target + 1) % 26, (target + 2) % 26]
        if len(seq) <= total_length:
            seq += [tokens.index('#')] * (total_length - seq_len)
        assert len(seq) == total_length == len(mask)
        samps.append((seq, target, mask))
    return samps

def get_batch(data, bs=8):
        batch = random.sample(data, bs)
        seq, target, mask = zip(*batch)
        batch_seq = torch.tensor([seq]).reshape(bs, len(seq[0]), 1)
        batch_target = torch.tensor([target]).reshape(bs, len(target[0]), 1)
        batch_mask = torch.tensor([mask]).reshape(bs, len(mask[0]), 1)
        return batch_seq, batch_target, batch_mask


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(len(tokens), 8, padding_idx=26) # emb_dim = 4
        # self.lstm = torch.nn.LSTM(4, 64, batch_first=True)
        # self.fc = torch.nn.Linear(64, len(tokens))
        self.attention = torch.nn.MultiheadAttention(embed_dim=8, num_heads=4, batch_first=True)
        self.fc = torch.nn.Linear(8, len(tokens))
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x, mask):
        x = self.embedding(x)

        new_mask = mask.clone()
        relevant_index = (new_mask.squeeze(0) == 1).nonzero(as_tuple=True)[0][-1].item()
        if relevant_index < new_mask.size(1) - 1:
            new_mask[:, relevant_index + 1] = 1

        x = x * new_mask.unsqueeze(2)
        # x, _ = self.lstm(x)
        x, _ = self.attention(x, x, x, key_padding_mask=(new_mask == 0))
        x = self.fc(x)
        # x = self.softmax(x)
        return x



def train(model, data, epochs=20):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        total_loss = 0
        #datas = [get_batch(data, bs=8) for _ in range(100)]
        for seq, target, mask in data:
            seq = torch.tensor([seq])
            mask = torch.tensor([mask])

            output = model(seq, mask)
            output = output.squeeze(0)
            
            target = torch.tensor(target)
            mask = mask.view(-1)

            # GOAL:
            # target: [10] mask: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] -> [10, 10, 10, 26, 26, 26, 26, 26, 26, 26]
            padding_value = 26
            # for i in range(batch_size): # FOR WHEN I GET TO BATCHES
            #     relevant_index = (mask[i].squeeze(0) == 1).nonzero(as_tuple=True)[0][-1].item()
            #     relevant_indices.append(relevant_index)
            relevant_index = (mask == 1).nonzero(as_tuple=True)[0][-1].item() # gets the last 1 in the mask
            new_target = torch.full((mask.size(0),), padding_value, dtype=torch.long)
            new_target[:relevant_index + 1] = target  # sets target values only up to the last 1 in the mask (inclusive)
            new_target[relevant_index + 1] = target  # first padding token is the actual output we want

            next_token_only_mask = torch.zeros_like(mask)
            next_token_only_mask[relevant_index + 1] = 1

            target = torch.tensor([target] * seq.size(1))
            loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
            loss = (loss * next_token_only_mask).sum() / mask.sum()
            # This DOES work and seems the FASTEST
            # target = new_target.view(-1)
            # relevant_index = (mask == 1).nonzero(as_tuple=True)[0][-1].item() moved it up
            # loss = torch.nn.functional.cross_entropy(
            #     output[relevant_index].unsqueeze(0),
            #     target[relevant_index].unsqueeze(0),
            #     ignore_index=26
            # )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 3 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

examples = generate_data(200, max_len=9, total_length=10)
examples.append(([10, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
examples.append(([10, 1, 2, 3, 4, 5, 6, 7, 8, 9], [11], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
examples.append(([12, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
examples.append(([12, 19, 2, 3, 4, 5, 6, 7, 8, 9], [19], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
examples.append(([14, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]))
examples.append(([14, 2, 11, 3, 4, 5, 6, 7, 8, 9], [11], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]))
examples.append(([1, 2, 4, 9, 20, 5, 6, 7, 8, 9], [20], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))
#import ipdb; ipdb.set_trace()

model = Model()
try:
    train(model, examples, epochs=1000)
except KeyboardInterrupt:
    pass

test1 = model(torch.tensor([[10, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
print(f"Prediction: {test1.squeeze(0)[1].argmax().item()}") # hope for 10
test2 = model(torch.tensor([[12, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])) # 10 + 1
print(f"Prediction: {test2.squeeze(0)[2].argmax().item()}") # hope for 1
test3 = model(torch.tensor([[14, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])) # 10 + 1 + 2
print(f"Prediction: {test3.squeeze(0)[3].argmax().item()}") # hope for 2
test4 = model(torch.tensor([[15, 1, 2, 3, 6, 5, 6, 7, 8, 9]]), torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])) # 10 + 1 + 2
print(f"Prediction: {test4.squeeze(0)[3].argmax().item()}") # hope for 6
import sys; sys.exit(1)

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
            self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.3)
            self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.3)
        self.gamma = 0.99
        self.eps_clip = 0.5
        self.K_epochs = 4
        self.data = data
    
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
        print(output)
        print(f"Input: {batch[0]}")
        print(f"Target: {batch[1]}")
        print(f"Mask: {batch[2].squeeze()}")


    
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

                dist = Categorical(action_probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                old_dist = Categorical(old_probs)
                old_log_probs = old_dist.log_prob(actions)

                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

                # Calculate the rewards
                first_predicted_token = (mask==0).float().argmax(1)
                rewards = []
                for i in range(target.size(0)): # bs
                    # discount_factor = 1.0
                    start_idx = first_predicted_token[i].item()
                    ep_reward = [0.] * start_idx
                    for target_j, j in zip([2, 1, 0], range(start_idx + 2, start_idx - 1, -1)): # seq_len
                        ep_reward.insert(0, (target[i, target_j] == actions[i, j]).float().item())# * discount_factor)
                        # discount_factor *= self.gamma
                    ep_reward += [0.] * (len(actions[0]) - len(ep_reward))
                    rewards.append(ep_reward)

                
                rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(-1)

                returns = []
                for i in range(rewards.size(0)):
                    R = 0
                    discounted_return = []
                    for r in rewards[i].flip(dims=(0,)):
                        R = r + self.gamma * R
                        discounted_return.insert(0, R)
                    returns.append(discounted_return)
                returns = torch.tensor(returns, dtype=torch.float).unsqueeze(-1)

                returns = returns.expand_as(old_probs)

                advantages = returns - old_probs.clone().detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # I feel like this is where it goes a bit shady
                predicted_mask = (mask == 0).float()
                # but only for the first 3 tokens starting from the first positive one
                not_before_this_token = (torch.arange(mask.size(1)).unsqueeze(0) >= first_predicted_token.unsqueeze(-1) - 2).float()
                not_after_this_token = (torch.arange(mask.size(1)).unsqueeze(0) <= first_predicted_token.unsqueeze(-1) + 2).float()

                not_before_this_token = not_before_this_token.permute(0, -1, 1)
                not_after_this_token = not_after_this_token.permute(0, -1, 1)
                mask = predicted_mask * not_before_this_token * not_after_this_token
                #import ipdb; ipdb.set_trace()

                ratio = ratio.unsqueeze(-1) * mask
                clipped_ratio = clipped_ratio.unsqueeze(-1) * mask

                surr1 = ratio * advantages
                surr2 = clipped_ratio * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = torch.nn.MSELoss()(old_probs * mask, returns * mask)
                #value_loss = torch.nn.MSELoss()(old_probs, returns)
                #loss = policy_loss + value_loss

                #import ipdb; ipdb.set_trace()

                self.actor_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward(retain_graph=False)
                self.critic_optimizer.step()

                self.actor_scheduler.step()
                self.critic_scheduler.step()
            if t % 100 == 0:
                #self.critic.load_state_dict(self.actor.state_dict())
                print(f"Step {t}, Policy Loss: {policy_loss.item()} Value Loss: {value_loss.item()}")
            if abs(policy_loss.item()) < 0.001:
                print("early stopping")
                break

            

if __name__ == "__main__":
    examples = generate_data(10, max_seq=10, max_len=7, for_ppo=True)
    ppo = PPO(model, examples)
    ppo.train(examples, timesteps=5000)
    ppo.eval()