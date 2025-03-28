import random
from collections import Counter
from copy import deepcopy

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PADDING_IDX = 0
TOTAL_SEQUENCE_LENGTH = 20
PREDICT_N_TOKENS_AT_A_TIME = 1
MAX_VOCAB_SIZE = 29 # Must be <= number of tokens in the corpus
BATCH_SIZE = 128  # Max, will use smaller batches sometimes
TEMPERATURE = 1.5
next_temp = TEMPERATURE

corpus = """
a b c d e f g h i j k l m n o p q r s t u v w x y z
a b c d e f g h i j k l m n o p q r s t u v w x y z
a b c d e f g h i j k l m n o p q r s t u v w x y z
"""
# Using # for padding so strip it
corpus_tokens = corpus.replace("#", "").split()


class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_token = False
        self.token_id = None

class TrieTokenizer:
    def __init__(self, max_tokens=50):
        self.root = TrieNode()
        self.id_to_token = {0: '#', 1: ' ', 2: '\n'}
        self.token_to_id = {'#': 0, ' ': 1, '\n': 2}
        self.next_id = 3
        self.max_tokens = max_tokens

    def insert(self, token):
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_of_token = True
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            node.token_id = self.next_id
            self.next_id += 1

    def fit(self, corpus_tokens):
        """
        1. Collect all single characters from the corpus and insert
        2. Use remaining slots for the most frequent multi-character tokens.
        """
        freqs = Counter(corpus_tokens)
        unique_chars = set()
        for token in corpus_tokens:
            for ch in token:
                unique_chars.add(ch)

        for ch in sorted(unique_chars):
            if self.next_id < self.max_tokens:
                self.insert(ch)

        multi_char_tokens = [(t, f) for t, f in freqs.items() if len(t) > 1]
        multi_char_tokens.sort(key=lambda x: x[1], reverse=True)

        for token, _ in multi_char_tokens:
            if self.next_id >= self.max_tokens:
                break
            self.insert(token)

    def tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            node = self.root
            last_match_id = None
            last_match_pos = i
            j = i
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.end_of_token:
                    last_match_id = node.token_id
                    last_match_pos = j
            if last_match_id is not None:
                tokens.append(last_match_id)
                i = last_match_pos
            else:
                char_token = text[i]
                tokens.append(self.token_to_id[char_token])
                i += 1
        return tokens

    def decode(self, ids):
        return ''.join(self.id_to_token[i] for i in ids)

# Tokenize the corpus and have possible sequences start only after whitespace
tokenizer = TrieTokenizer(max_tokens=MAX_VOCAB_SIZE)
tokenizer.fit(corpus_tokens)
print(tokenizer.token_to_id)
corpus = tokenizer.tokenize(corpus)
possible_starting_points = [i for i, x in enumerate(corpus) if (tokenizer.decode([x]) in [" ", "\n"] and i + PREDICT_N_TOKENS_AT_A_TIME < len(corpus))]


def generate_data(n_samples, max_len=6, total_length=10, min_len=5):
    samps = []
    max_seq = total_length - 1 # at least one padding token
    for i in range(n_samples):
        seq_len = random.randint(max(PREDICT_N_TOKENS_AT_A_TIME, min_len), max_len)
        starting_point = random.choice(possible_starting_points) # or random.randint(0, len(corpus) - max_seq) to start anywhere, not after whitespace
        excerpt = corpus[starting_point:starting_point + seq_len + max_seq]
        seq = excerpt[:seq_len]
        mask = [1] * seq_len + [0] * (total_length - seq_len)
        target = excerpt[seq_len:seq_len + PREDICT_N_TOKENS_AT_A_TIME]

        if len(target) != PREDICT_N_TOKENS_AT_A_TIME:
            continue  # TODO fix some edge case

        if len(seq) <= total_length:
            seq += [PADDING_IDX] * (total_length - seq_len)
        assert len(seq) == total_length == len(mask)
        assert len(target) == PREDICT_N_TOKENS_AT_A_TIME
        samps.append((seq, target, mask))
    return samps


def get_batch(data, bs=8):
        batch = random.sample(data, bs)
        seq, target, mask = zip(*batch)
        batch_seq = torch.tensor([seq], device=device).reshape(bs, len(seq[0]), 1)
        batch_target = torch.tensor([target], device=device).reshape(bs, len(target[0]), 1)
        batch_mask = torch.tensor([mask], device=device).reshape(bs, len(mask[0]), 1)

        seq_length = batch_mask.size(1)
        new_target = torch.full((bs, seq_length), PADDING_IDX, dtype=torch.long, device=device)
        next_tokens_only_mask = torch.zeros_like(batch_mask, device=device)
        

        for i in range(bs):
            relevant_index = (batch_mask[i].squeeze(0) == 1).nonzero(as_tuple=True)[0][-1].item()
            for j in range(1, PREDICT_N_TOKENS_AT_A_TIME + 1):
                new_target[i, relevant_index + j] = batch_target[i][j - 1].item()
                next_tokens_only_mask[i, relevant_index + j] = 1

        return batch_seq, new_target, batch_mask, next_tokens_only_mask


class TransformerBlock(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm1 = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embedding_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ff_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_hidden_dim, embedding_dim)
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.attention(x, x, x, attn_mask=self.causal_mask(x.size(1)), key_padding_mask=(mask == 0).squeeze(-1).float())
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x
    
    def causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)
    
class Model(torch.nn.Module):
    def __init__(self, embedding_dim=32, num_heads=4, ff_hidden_dim=128, num_layers=6,
                 max_len=TOTAL_SEQUENCE_LENGTH,
                 num_tokens=100):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(num_tokens, embedding_dim)
        self.pos_embedding = torch.nn.Embedding(max_len, embedding_dim)
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)
        ])
        self.fc = torch.nn.Linear(embedding_dim, num_tokens)
        self.dropout = torch.nn.Dropout(0.1)
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.value_head = torch.nn.Linear(embedding_dim, 1)

    def forward(self, x, mask):
        seq_length = x.size(1)
        pos = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(x.size(0), -1)
        
        x = self.embedding(x) + self.pos_embedding(pos.unsqueeze(2))
        x = x.squeeze(2)
        x = x * mask

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        x = self.layer_norm(x)
        x = self.dropout(x)
        policy_logits = self.fc(x)
        value = self.value_head(x)
        return policy_logits, value

def train(model, data, epochs=20, early_stop=-float("inf")):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    for epoch in range(epochs):
        if random.random() < 0.1:
            mini_batch_size = BATCH_SIZE // 4  # Some smaller batches to maybe escape local minima
        elif random.random() < 0.2:
            mini_batch_size = BATCH_SIZE // 2
        else:
            mini_batch_size = BATCH_SIZE
        datas = [get_batch(data, bs=mini_batch_size) for _ in range(len(data) // BATCH_SIZE)]
        for seq, target, input_mask, output_mask in datas:
            output, _ = model(seq, input_mask)
            
            loss = torch.nn.functional.cross_entropy(output.permute(0, 2, 1), target, reduction='none')
            
            # in_plus_next_mask = torch.max(input_mask, output_mask) # predict seq as well as next token, or just use output_mask for next token only
            chosen_mask = output_mask  # This is probably better.
            loss = (loss * chosen_mask.squeeze(-1)).sum() / chosen_mask.sum()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        if epoch % 3 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        if loss.item() < early_stop:
            break

examples = generate_data(10000, max_len=TOTAL_SEQUENCE_LENGTH - PREDICT_N_TOKENS_AT_A_TIME,
                         total_length=TOTAL_SEQUENCE_LENGTH, min_len=4)

model = Model(embedding_dim=64, num_tokens=MAX_VOCAB_SIZE).to(device)
model.train()
try:
    train(model, examples, epochs=10000, early_stop=0.2)
except KeyboardInterrupt:
    pass
model.eval()


num_eval = 50
eval_examples = generate_data(num_eval, max_len=TOTAL_SEQUENCE_LENGTH - PREDICT_N_TOKENS_AT_A_TIME, total_length=TOTAL_SEQUENCE_LENGTH)
total_correct = 0

for i in range(num_eval):
    seq, target, input_mask, output_mask = get_batch(eval_examples, bs=1)
    output, _ = model(seq, input_mask)
    output = output.squeeze(0)
    relevant_index = (output_mask == 1).nonzero(as_tuple=True)[1][0].item()
    
    target = target.squeeze(0)
    predictions = output[relevant_index:relevant_index + PREDICT_N_TOKENS_AT_A_TIME].argmax(dim=-1)
    targets = target[relevant_index:relevant_index + PREDICT_N_TOKENS_AT_A_TIME]
    correct = (predictions == targets).sum().item()
    predictions = [x.item() for x in predictions]
    token_pred = tokenizer.decode(predictions)
    targets = [x.item() for x in targets]
    token_targets = tokenizer.decode(targets)
    print(f"Prediction: {predictions}, Target: {targets}")
    print(f"English prediction: {token_pred}, English target: {token_targets}")
    total_correct += correct / PREDICT_N_TOKENS_AT_A_TIME
print(f"Accuracy: {total_correct / num_eval}")

def generate(prompt, iterations=3,
             max_len=TOTAL_SEQUENCE_LENGTH - PREDICT_N_TOKENS_AT_A_TIME,
             total_length=TOTAL_SEQUENCE_LENGTH,
             beam_width=5):
    print(f"---- STARTING PROMPT: {prompt} ----")
    if len(prompt) > max_len:
        raise ValueError("Prompt too long")
    beams = [(prompt, 0.0, prompt)]

    for _ in range(iterations):
        new_beams = []
        
        for seq, cum_log_prob, full_seq in beams:
            int_seq = tokenizer.tokenize(seq)
            mask = [1] * len(int_seq) + [0] * (total_length - len(int_seq))
            true_seq = deepcopy(int_seq)
            if len(int_seq) < total_length:
                int_seq += [PADDING_IDX] * (total_length - len(int_seq))
            assert len(int_seq) == len(mask)
            seq_tensor = torch.tensor([int_seq], device=device).reshape(1, len(int_seq), 1)
            mask_tensor = torch.tensor([mask], device=device).reshape(1, len(mask), 1)
            assert mask_tensor.view(-1)[-1] == 0

            with torch.no_grad():
                output, _ = model(seq_tensor, mask_tensor)
            output = output.squeeze(0)
            relevant_index = (mask_tensor == 1).nonzero(as_tuple=True)[1][-1].item() + 1
            logits = output[relevant_index:relevant_index + PREDICT_N_TOKENS_AT_A_TIME]
            
            topk_probs, topk_indices = torch.topk(torch.nn.functional.log_softmax(logits, dim=-1), beam_width, dim=-1)
            for i in range(beam_width):
                best = topk_indices[:PREDICT_N_TOKENS_AT_A_TIME, i].tolist()
                new_seq = seq + tokenizer.decode(best)
                new_cum_log_prob = cum_log_prob + topk_probs[:PREDICT_N_TOKENS_AT_A_TIME, i].sum().item()
                full_seq += tokenizer.decode(best)
                if len(true_seq + best) >= max_len:
                    new_seq = tokenizer.decode((true_seq + best)[PREDICT_N_TOKENS_AT_A_TIME + 1:]) # Extra one as at least one padding token
                new_beams.append((new_seq, new_cum_log_prob, full_seq))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return [(x[2], round(x[1], 3)) for x in beams]

print(generate("a b c", 25))
print(generate("l m n o", 25))
#print(generate("In some systems,", 25))
# print(generate("The King", 25))
# print(generate("feudal lords", 25))
# print(generate("you peasant", 25))

#############################################
model.train()

from torch.distributions import Categorical

class PolicyNetwork(torch.nn.Module):
    def __init__(self, model):
        super(PolicyNetwork, self).__init__()
        self.model = model

    def forward(self, x, mask):
        logits, _ = self.model(x, mask)
        return logits

class ValueNetwork(torch.nn.Module):
    def __init__(self, model):
        super(ValueNetwork, self).__init__()
        self.model = model

    def forward(self, x, mask):
        _, value = self.model(x, mask)
        return value


class PPO:
    def __init__(self, policy_net, value_net, policy_lr=0.0001, value_lr=0.0001, gamma=0.99, clip_epsilon=0.2,
                 update_epochs=10, batch_size=BATCH_SIZE):
        self.policy_net = policy_net
        self.value_net = value_net
        #self.policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=policy_lr)
        ###### UGGHHHHH model.value_head.
        #self.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr)
        # params = list(policy_net.parameters()) + list(value_net.parameters())
        params = set(policy_net.parameters()).union(set(value_net.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=policy_lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
    
    def normalize(self, tensor):
        # tensor = torch.tensor(tensor, device=device)
        mean = tensor.mean()
        std = tensor.std()
        return ((tensor - mean) / (std + 1e-8))#.tolist()

    def compute_advantages(self, rewards, values, masks):
        advantages = []
        returns = []
        # rewards = self.normalize(rewards)
        return_ = torch.zeros_like(values[0], device=device)
        
        returns.append(rewards[-1])
        # Initialize the last advantage with the last TD error
        advantages.append(values[-1] - return_)
        # Previously I had advantages.append(rewards[-1] - values[-1])

        for i in reversed(range(len(rewards) - 1)):
            return_ = rewards[i] + self.gamma * return_  # Compute the return for the current step
            returns.insert(0, return_)  # Insert at the beginning of the list and detach

            td_error = rewards[i] + self.gamma * values[i + 1] - values[i]
            advantage = td_error + self.gamma * advantages[0]
            advantages.insert(0, advantage)

        return torch.tensor(advantages, device=device), torch.tensor(returns, device=device)

    def get_batch(self, trajectories):
        batch = random.sample(trajectories, self.batch_size)
        states, actions, rewards, masks, old_log_probs, values = zip(*batch)
        batch_states = torch.cat(states, dim=0)
        batch_actions = torch.cat(actions, dim=0)
        batch_rewards = torch.tensor(rewards, device=device).reshape(self.batch_size, 1)
        batch_masks = torch.cat(masks, dim=0)
        batch_relevant_indices = [((mask == 1).nonzero(as_tuple=True)[1][-1].item() + 1) for mask in masks]
        batch_relevant_indices = torch.tensor(batch_relevant_indices, device=device)
        batch_old_log_probs = torch.cat(old_log_probs, dim=0)
        batch_values = torch.cat(values, dim=0)
        return batch_states, batch_actions, batch_rewards, batch_masks, batch_relevant_indices, batch_old_log_probs, batch_values


    def update(self, trajectories):
        for _ in range(self.update_epochs):
            epoch_value_loss, epoch_policy_loss = 0.0, 0.0
            for i in range(max(len(trajectories) // self.batch_size, 1)):
                batch_states, batch_actions, batch_rewards, batch_masks, batch_relevant_indices, batch_old_log_probs, batch_values = self.get_batch(trajectories)
                advantages, returns = self.compute_advantages(batch_rewards, batch_values, batch_masks)

                logits = self.policy_net(batch_states, batch_masks)
                
                these_logits = torch.gather(logits, 1, batch_relevant_indices.view(-1, 1, 1).expand(-1, -1, logits.size(-1)))  # correct
                # TODO this breaks PREDICT_N_TOKENS_AT_A_TIME > 1
                dist = Categorical(logits=these_logits / TEMPERATURE)

                log_probs = dist.log_prob(batch_actions.view(self.batch_size, 1)).view_as(batch_old_log_probs)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value = self.value_net(batch_states, batch_masks)
                
                
                # This works but value is bs x seqlen x 1 while returns is bs x 1
                # value_loss = (returns - value).pow(2).mean()
                # This seems more correct but doesn't work
                # value_loss = (returns - value.squeeze(-1).mean(-1)).pow(2).mean()
                value_loss = (returns - value.squeeze(-1).mean(dim=-1)).pow(2).mean()
                
                #self.value_optimizer.zero_grad()
                #value_loss.backward()
                #self.value_optimizer.step()
                #self.policy_optimizer.zero_grad()
                self.optimizer.zero_grad()
                #policy_loss.backward()
                #self.policy_optimizer.step()
                (policy_loss + value_loss).backward()
                self.optimizer.step()
                epoch_value_loss += value_loss.item()
                epoch_policy_loss += policy_loss.item()
            print(f"Policy loss: {epoch_policy_loss}, Value loss: {epoch_value_loss}")



def collect_trajectories(policy_net, value_net, prompts,
                         max_len=TOTAL_SEQUENCE_LENGTH - PREDICT_N_TOKENS_AT_A_TIME,
                         total_length=TOTAL_SEQUENCE_LENGTH,
                         epsilon=-float("inf")):
    global next_temp
    states = []
    actions = []
    rewards = []
    masks = []
    log_probs = []
    values = []

    avg_reward = 0.0

    TEMPERATURE = next_temp
    print(f"Temperature: {TEMPERATURE}")

    for prompt in prompts:
        int_seq = tokenizer.tokenize(prompt) 
        iterations = TOTAL_SEQUENCE_LENGTH - len(int_seq) - PREDICT_N_TOKENS_AT_A_TIME
        while iterations > 0:
            true_seq = deepcopy(int_seq)   
            mask = [1] * len(int_seq) + [0] * (total_length - len(int_seq))
            if len(int_seq) < total_length:
                int_seq += [PADDING_IDX] * (total_length - len(int_seq))
            seq_tensor = torch.tensor([int_seq], device=device).reshape(1, len(int_seq), 1)
            mask_tensor = torch.tensor([mask], device=device).reshape(1, len(mask), 1)

            states.append(seq_tensor)
            masks.append(mask_tensor)
            
            assert len(int_seq) == len(mask)
            assert mask_tensor.view(-1)[-1] == 0

            with torch.no_grad():
                output = policy_net(seq_tensor, mask_tensor)
                value = value_net(seq_tensor, mask_tensor)

            output = output.detach().squeeze(0)  # This chops off the batch dimension??????????
            relevant_index = (mask_tensor == 1).nonzero(as_tuple=True)[1][-1].item() + 1
            logits = output[relevant_index:relevant_index + PREDICT_N_TOKENS_AT_A_TIME]
            
            dist = Categorical(logits=logits / TEMPERATURE)

            if random.random() < epsilon:
                action = torch.tensor([random.randint(0, logits.size(-1) - 1)], device=device)
                log_prob = torch.tensor([0.0], device=device)
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action) # TODO make it work with PREDCIT_N_TOKENS_AT_A_TIME
            
            #print(f"Chosen action ({action.item()}): {tokenizer.decode([action.item()])}, log prob: {log_prob.item()}")

            int_seq = true_seq + action.tolist()
            
            actions.append(action)
            log_probs.append(log_prob)

            values.append(value.squeeze(-1).mean(dim=-1))
            iterations -= 1
            seq_text = tokenizer.decode(int_seq)
            good_chars = seq_text.count("a") + seq_text.count("b")
            bad_chars = seq_text.count("e") - prompt.count("e")
            all_chars = set(seq_text)
            unique_chars = len(all_chars)
            if iterations == 0:
                num_whitespaces = (seq_text.count(" ") + seq_text.count("\n"))
                reward = (good_chars * 2) - bad_chars
                print(f"---> Desired chars: {good_chars} vs whitespace: {num_whitespaces} vs unique chars: {unique_chars} ------ Reward: {reward}")
                rewards.append(reward)
            else:

                rewards.append(0.0)#(0.02*unique_chars)#0.01 * int(good_chars > bad_chars))#rewards.append(0.0)
            
            avg_reward += rewards[-1]
    avg_reward /= len(prompts)
    print(f"Average reward: {avg_reward}")
    if False:#avg_reward > 5:
        next_temp = TEMPERATURE * 0.9
    elif False:#avg_reward < 3:
        next_temp = TEMPERATURE * 1.1
    else:
        next_temp = TEMPERATURE
    print(f"Next temperature should be: {next_temp}")
    next_temp = max(0.9, min(4.0, next_temp))
    print(f"Next temperature will be: {next_temp}")
    
    return [(x, y, z, a, b, c) for x, y, z, a, b, c in zip(states, actions, rewards, masks, log_probs, values)]
    #return zip(states, actions, rewards, masks, log_probs, values)


policy_net = PolicyNetwork(model).to(device)
value_net = ValueNetwork(model).to(device)

# Initialize PPO
ppo = PPO(policy_net, value_net, update_epochs=10, batch_size=12)

prompts = [
    "a",
    "a b ",
    "a b",
    "ab",
    "b c d ",
    "b c d e ",
    "a b c d e",
    # "c d e ",
    # "a b c d e ",
    # "a b c a b",
    # "a b c a b c a b c",
    # "a b c d e",
    # "f g h b c",
    # "l m n o p q r",
    # "l m nod d d",
    "d e f",
    "x y z",
    "w x y z",
    "r s t u v",
    "w x y z\na b c",
    "z\na b c d"
]

num_epochs = 100
num_steps = 2048
try:
    for epoch in range(num_epochs):
        trajectories = collect_trajectories(policy_net, value_net, prompts)
        ppo.update(trajectories)
        print(f"Epoch {epoch} completed")
except KeyboardInterrupt:
    pass

for p in prompts:
    print(generate(p, 25))
# print(generate("In some systems,", 25))
# print(generate("The King", 25))
# print(generate("feudal lords", 25))
# print(generate("you peasant", 25))