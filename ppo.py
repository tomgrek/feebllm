# text generating model
# given some token and a mask, predict the next token

import torch
import string
import random

tokens = string.ascii_lowercase + "#" # Add padding token

print(tokens)

def generate_data(n_samples):
    samps = []
    for i in range(n_samples):
        seq_len = random.randint(1, 9)
        seq = [random.randint(0, len(tokens) - 1) for _ in range(seq_len)]
        mask = [1] * seq_len + [0] * (9 - seq_len) + [0]
        # Target should be the sum of the tokens mod 26
        target = 10#sum(seq) % 26
        seq += [26] * (9 - seq_len) + [26]
        samps.append((seq, target, mask))
    return samps

examples = generate_data(2000)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(len(tokens), 16)
        self.lstm = torch.nn.LSTM(16, 128, batch_first=True)
        self.fc = torch.nn.Linear(128, len(tokens))

    def forward(self, x, mask):
        x = self.embedding(x)
        x = x * mask.unsqueeze(2)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = torch.nn.functional.log_softmax(x, dim=2)
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

train(model, examples)
test1 = model(torch.tensor([[10, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
print(f"Prediction: {test1.squeeze(0)[1].argmax().item()}") # hope for 10
test2 = model(torch.tensor([[12, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])) # 10 + 1
print(f"Prediction: {test2.squeeze(0)[2].argmax().item()}") # hope for 13
test3 = model(torch.tensor([[14, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])) # 10 + 1 + 2
print(f"Prediction: {test3.squeeze(0)[3].argmax().item()}") # hope for 17

# Now modify it, the example data should change so that the next token is the sum (still), the next one is that sum + 1 (%26) and again + 1 for the next one
# Note (and this is NO LONGER true currently as well) the output shape should be larger, ie max seq len needs to reduce by 3 (and by 1 currently)
# Use PPO to learn this, starting from the trained model.


#TODO i think this is off and haven't checked it, looks like target is wrong and mask is not long enough
def generate_data(n_samples):
    samps = []
    for i in range(n_samples):
        seq_len = random.randint(1, 6)
        seq = [random.randint(0, len(tokens) - 1) for _ in range(seq_len)]
        mask = [1] * seq_len + [0] * (6 - seq_len) + [0]
        # Target should be the sum of the tokens mod 26
        target = sum(seq) % 26
        seq += [26] * (6 - seq_len) + [26]
        assert len(seq) == len(mask)
        samps.append((seq, target, mask))
    return samps

examples = generate_data(2)
print(examples)

class PPO:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        self.gamma = 0.99
        self.eps = 0.2
        self.lmbda = 0.95
        self.K = 3
        self.epochs = 10

    def train(self, data):
        for epoch in range(self.epochs):
            for seq, target, mask in data:
                seq = torch.tensor([seq])
                target = torch.tensor([target] * seq.size(1))
                mask = torch.tensor([mask])
                old_probs = torch.exp(self.model(seq, mask))
                old_probs = old_probs.gather(2, seq.unsqueeze(2)).squeeze(2)
                old_probs = old_probs * mask
                old_probs = old_probs.view(-1)
                old_probs = old_probs[old_probs != 0]
                old_probs = old_probs.detach()

                for _ in range(self.K):
                    output = self.model(seq, mask)
                    output = output.squeeze(0)
                    target = target.view(-1)
                    mask = mask.view(-1)
                    probs = torch.exp(output)
                    probs = probs.gather(1, seq)
                    probs = probs.squeeze(1)
                    probs = probs * mask
                    probs = probs.view(-1)
                    probs = probs[mask == 1]
                    ratio = probs / old_probs
                    advantage = target - seq
                    import ipdb; ipdb.set_trace()
                    advantage = advantage[mask == 1]
                    advantage = advantage.view(-1)
                    advantage = advantage.detach()
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
                    loss = -torch.min(surr1, surr2)
                    loss = loss.mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

ppo = PPO(model)
ppo.train(examples)