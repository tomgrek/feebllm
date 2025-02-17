import random
from collections import Counter
from copy import deepcopy

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PADDING_IDX = 0
TOTAL_SEQUENCE_LENGTH = 20
PREDICT_N_TOKENS_AT_A_TIME = 1
MAX_VOCAB_SIZE = 200
BATCH_SIZE = 128  # Max, will use smaller batches sometimes

corpus = """
In some systems, it is obvious how economic dominant minorities control the state. In feudalism, for example, the land was owned by the feudal lords who exploited the peasantry directly. Economic and political power were merged into the same set of hands, the landlords. Absolutism saw the monarch bring the feudal lords under his power and the relative decentralised nature of feudalism was replaced by a centralised state.
It was this centralised state system which the raising bourgeoisie took as the model for their state. The King was replaced by a Parliament, which was initially elected on a limited suffrage. In this initial form of capitalist state, it is (again) obvious how the elite maintain control of the state machine. As the vote was based on having a minimum amount of property, the poor were effectively barred from having any (official) say in what the government did. This exclusion was theorised by philosophers like John Locke—the working masses were considered to be an object of state policy rather than part of the body of people (property owners) who nominated the government. In this perspective the state was like a joint-stock company. The owning class were the share-holders who nominated the broad of directors and the mass of the population were the workers who had no say in determining the management personnel and were expected to follow orders.
As would be expected, this system was mightily disliked by the majority who were subjected to it. Such a "classical liberal" regime was rule by an alien, despotic power, lacking popular legitimacy, and utterly unaccountable to the general population. It is quite evident that a government elected on a limited franchise could not be trusted to treat those who owned no real property with equal consideration. It was predictable that the ruling elite would use the state they controlled to further their own interests and to weaken potential resistance to their social, economic and political power. Which is precisely what they did do, while masking their power under the guise of "good governance" and "liberty." Moreover, limited suffrage, like absolutism, was considered an affront to liberty and individual dignity by many of those subject to it.
This is unsurprising as we, as a species, shape our environment and, consequently, whatever shapes us will impact how we do so. This means that the individuals produced by the hierarchy (and the authoritarian mentality it produces) will shape the planet in specific, harmful, ways. This is to be expected as humans act upon their environment deliberately, creating what is most suitable for their mode of existence. If that mode of living is riddled with hierarchies, classes, states and the oppression, exploitation and domination they create then our relations with the natural world will hardly be any better. In other words, social hierarchy and class legitimises our domination of the environment, planting the seeds for the believe that nature exists, like other people, to be dominated and used as required.
As the gap grows between rich and poor (indicating an increase in social hierarchy within and outwith of workplaces) the health of a people deteriorates and the social fabric unravels. The psychological hardship of being low down on the social ladder has detrimental effects on people, beyond whatever effects are produced by the substandard housing, nutrition, air quality, recreational opportunities, and medical care enjoyed by the poor.
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
        x = self.fc(x)
        return x

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
            output = model(seq, input_mask)
            
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
    train(model, examples, epochs=10000, early_stop=0.01)
except KeyboardInterrupt:
    pass
model.eval()


num_eval = 50
eval_examples = generate_data(num_eval, max_len=TOTAL_SEQUENCE_LENGTH - PREDICT_N_TOKENS_AT_A_TIME, total_length=TOTAL_SEQUENCE_LENGTH)
total_correct = 0

for i in range(num_eval):
    seq, target, input_mask, output_mask = get_batch(eval_examples, bs=1)
    output = model(seq, input_mask)
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
                output = model(seq_tensor, mask_tensor)
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

print(generate("In some systems,", 25))
print(generate("The King", 25))
print(generate("feudal lords", 25))
print(generate("you peasant", 25))

#############################################

from torch.distributions import Categorical

class PolicyNetwork(torch.nn.Module):
    def __init__(self, model):
        super(PolicyNetwork, self).__init__()
        self.model = model

    def forward(self, x, mask):
        logits = self.model(x, mask)
        return logits

class ValueNetwork(torch.nn.Module):
    def __init__(self, embedding_dim, num_tokens):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(embedding_dim, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value


class PPO:
    def __init__(self, policy_net, value_net, policy_lr=0.0002, value_lr=0.0001, gamma=0.999, clip_epsilon=0.1, update_epochs=10):
        self.policy_net = policy_net
        self.value_net = value_net
        self.policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

    def compute_advantages(self, rewards, values, masks):
        advantages = []
        returns = []
        advantage = 0
        return_ = 0
        for i in reversed(range(len(rewards) - 1)):
            return_ = rewards[i] + self.gamma * return_ * masks[i]
            td_error = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            advantage = td_error + self.gamma * advantage * masks[i]
            returns.insert(0, return_.clone().detach())
            advantages.insert(0, advantage.clone().detach())
        return advantages, returns

    def update(self, trajectories):
        states, actions, rewards, masks, old_log_probs, values = trajectories
        advantages, returns = self.compute_advantages(rewards, values, masks)

        for _ in range(self.update_epochs):
            for state, action, mask, old_log_prob, advantage, return_ in zip(states, actions, masks, old_log_probs, advantages, returns):
                logits = self.policy_net(state, mask)
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(action)
                ratio = torch.exp(log_prob - old_log_prob)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                policy_loss = torch.min(surr1, surr2).mean() # SHOULD BE A MINUS !!!!!!!!!!!!!!!!!!!!!!!!!!!!! THERE WAS A MINUS HERE!!!!!!!!!!!!!!!!!

                with torch.no_grad():
                    embeddings = self.policy_net.model.embedding(state)
                value = self.value_net(embeddings.squeeze(2))
                value_loss = (return_ - value).pow(2).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
            print(f"Policy loss: {policy_loss.item()}, Value loss: {value_loss.item()}")


def collect_trajectories(policy_net, value_net, prompts,
                         max_len=TOTAL_SEQUENCE_LENGTH - PREDICT_N_TOKENS_AT_A_TIME,
                         total_length=TOTAL_SEQUENCE_LENGTH):
    states = []
    actions = []
    rewards = []
    masks = []
    log_probs = []
    values = []

    for prompt in prompts:
        int_seq = tokenizer.tokenize(prompt) 
        iterations = TOTAL_SEQUENCE_LENGTH - len(int_seq) - PREDICT_N_TOKENS_AT_A_TIME
        cum_log_prob = 0.0
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

            output = output.squeeze(0)
            relevant_index = (mask_tensor == 1).nonzero(as_tuple=True)[1][-1].item() + 1
            logits = output[relevant_index:relevant_index + PREDICT_N_TOKENS_AT_A_TIME]
            
            dist = Categorical(logits=logits)
            action = dist.sample() # TODO make it work with PREDCIT_N_TOKENS_AT_A_TIME
            log_prob = dist.log_prob(action) # TODO make it work with PREDCIT_N_TOKENS_AT_A_TIME
            cum_log_prob += log_prob.item() # TODO make it work with PREDCIT_N_TOKENS_AT_A_TIME

            int_seq = true_seq + action.tolist()
            
            actions.append(action)
            log_probs.append(log_prob.detach())
            with torch.no_grad():
                embeddings = policy_net.model.embedding(seq_tensor.detach()).detach()
            values.append(value_net(embeddings.squeeze(2)))
            iterations -= 1
            if iterations == 0:
                rewards.append(cum_log_prob)
            else:
                rewards.append(0.0)
    
    return states, actions, rewards, masks, log_probs, values


policy_net = PolicyNetwork(model).to(device)
value_net = ValueNetwork(embedding_dim=64, num_tokens=MAX_VOCAB_SIZE).to(device)

# Initialize PPO
ppo = PPO(policy_net, value_net)

# Training loop
num_epochs = 1000
num_steps = 2048

prompts = [
    "In some systems,",
    "The King",
    "feudal lords",
    "you peasant"
]

try:
    for epoch in range(num_epochs):
        trajectories = collect_trajectories(policy_net, value_net, prompts, num_steps)
        ppo.update(trajectories)
        print(f"Epoch {epoch} completed")
except KeyboardInterrupt:
    pass

print(generate("In some systems,", 25))
print(generate("The King", 25))
print(generate("feudal lords", 25))
print(generate("you peasant", 25))