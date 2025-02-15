from collections import Counter

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
        # Only assign a new ID if this exact token isn't already in self.token_to_id
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            node.token_id = self.next_id
            self.next_id += 1

    def fit(self, corpus_tokens):
        """
        1. Collect all single characters from the corpus and insert them (ensuring coverage).
        2. Use remaining slots for the most frequent multi-character tokens.
        """
        freqs = Counter(corpus_tokens)
        unique_chars = set()
        for token in corpus_tokens:
            for ch in token:
                unique_chars.add(ch)

        # Step 1: Insert all unique single characters (e.g. 'a', 'b') to guarantee coverage
        for ch in sorted(unique_chars):
            if self.next_id < self.max_tokens:
                self.insert(ch)

        # Step 2: Insert the most common multi-character tokens until we run out of space
        # Filter out single characters from the counter, so we only consider multi-char tokens
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
                # If there's no match, fallback to single-character coverage
                char_token = text[i]
                # Because we guaranteed single-char coverage in fit(), it must exist
                tokens.append(self.token_to_id[char_token])
                i += 1
        return tokens

    def decode(self, ids):
        return ''.join(self.id_to_token[i] for i in ids)

# text generating model
# given some token and a mask, predict the next token

import torch
from torch.distributions import Categorical
import string
import math
import random

tokens = string.ascii_lowercase + " .#" # Add space period and padding token
PADDING_IDX = 0# tokens.index("#")
TOTAL_SEQUENCE_LENGTH = 20
PREDICT_N_TOKENS_AT_A_TIME = 2

corpus = """
The quick brown fox jumps over the lazy dog.
You will rejoice to hear that no disaster has accompanied the commencement of an enterprise which you have regarded with such evil forebodings. I arrived here yesterday, and my first task is to assure my dear sister of my welfare and increasing confidence in the success of my undertaking.
I am already far north of London, and as I walk in the streets of Petersburgh, I feel a cold northern breeze play upon my cheeks, which braces my nerves and fills me with delight. Do you understand this feeling? This breeze, which has travelled from the regions towards which I am advancing, gives me a foretaste of those icy climes. Inspirited by this wind of promise, my daydreams become more fervent and vivid. I try in vain to be persuaded that the pole is the seat of frost and desolation; it ever presents itself to my imagination as the region of beauty and delight.
Once upon a midnight dreary, while I pondered, weak and weary, over many a quaint and curious volume of forgotten lore. While I nodded, nearly napping, suddenly there came a tapping, as of someone gently rapping, rapping at my chamber door. 'Tis some visitor, I muttered, tapping at my chamber door. Only this and nothing more.
Having had some time at my disposal when in London, I had visited the British Museum, and made search among the books and maps in the library regarding Transylvania; it had struck me that some foreknowledge of the country could hardly fail to have some importance in dealing with a nobleman of that country. I find that the district he named is in the extreme east of the country, just on the borders of three states, Transylvania, Moldavia and Bukovina, in the midst of the Carpathian mountains; one of the wildest and least known portions of Europe.
Puck (or as he was sometimes called, Robin Goodfellow) was a shrewd and knavish sprite, that used to play comical pranks in the neighbouring villages; sometimes getting into the dairies and skimming the milk, sometimes plunging his light and airy form into the butter-churn, and while he was dancing his fantastic shape in the churn, in vain the dairymaid would labour to change her cream into butter: nor had the village swains any better success; whenever Puck chose to play his freaks in the brewing copper, the ale was sure to be spoiled.
All this time Dorothy and her companions had been walking through the thick woods. The road was still paved with yellow brick, but these were much covered by dried branches and dead leaves from the trees, and the walking was not at all good.
Even with eyes protected by the green spectacles, Dorothy and her friends were at first dazzled by the brilliancy of the wonderful City. The streets were lined with beautiful houses all built of green marble and studded everywhere with sparkling emeralds. They walked over a pavement of the same green marble, and where the blocks were joined together were rows of emeralds, set closely, and glittering in the brightness of the sun. The window panes were of green glass; even the sky above the City had a green tint, and the rays of the sun were green.
"""
#corpus += 10*" robin goodfellow was a shrewd and knavish sprite that used to play comical pranks in the neighbouring villages sometimes getting into the dairies and skimming the milk sometimes plunging his light and airy form into the butter churn and while he was dancing his fantastic shape in the churn in vain the dairymaid would labour to change her cream into butter."
# corpus = "abcdefghijklmnopqrstuvwxyz " * 5

corpus_tokens = corpus.replace("#", "").split()

tokenizer = TrieTokenizer(max_tokens=54)
tokenizer.fit(corpus_tokens)
print(tokenizer.token_to_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# corpus = [x for x in corpus.lower() if x in tokens.replace("#", "")]
corpus = tokenizer.tokenize(corpus)

# Context window is 10, for next token prediction we can generate up to 9; for PPO upto 7
def generate_data(n_samples, max_len=6, total_length=10, min_len=5):
    samps = []
    max_seq = total_length - 1 # at least one padding token
    for i in range(n_samples):
        seq_len = random.randint(max(PREDICT_N_TOKENS_AT_A_TIME, min_len), max_len)
        starting_point = random.randint(0, len(corpus) - max_seq)
        excerpt = corpus[starting_point:starting_point + seq_len + max_seq]
        seq = excerpt[:seq_len]
        mask = [1] * seq_len + [0] * (total_length - seq_len)
        target = excerpt[seq_len:seq_len + PREDICT_N_TOKENS_AT_A_TIME]

        if len(target) != PREDICT_N_TOKENS_AT_A_TIME:
            continue  # TODO fix some edge case
        
        # seq = [tokens.index(x) for x in seq]
        # target = [tokens.index(x) for x in target]

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
                #batch_mask[i, relevant_index + j] = 1
        
        # Input sequence, target, input mask, output mask
        # Target is [padding, padding, target1, target2, target3, padding...]
        # Batch mask is the input mask: 1 for both sequence input positions + target positions
        # Output mask is 1 for only the target positions
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
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x, attn_mask=self.causal_mask(x.size(1)), key_padding_mask=(mask == 0).squeeze(-1).float())
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
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


batch_size = 128
def train(model, data, epochs=20, early_stop=-float("inf")):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    for epoch in range(epochs):
        datas = [get_batch(data, bs=batch_size) for _ in range(len(data) // batch_size)]
        for seq, target, input_mask, output_mask in datas:
            output = model(seq, input_mask) # both are (bs, seq_len, 1). Target is (bs, seq_len).
            
            loss = torch.nn.functional.cross_entropy(output.permute(0, 2, 1), target, reduction='none')
            
            # in_plus_next_mask = torch.max(input_mask, output_mask) # predict seq as well as next token, or just use output_mask for next token only
            chosen_mask = output_mask
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

model = Model(embedding_dim=64, num_tokens=54).to(device)#len(tokenizer.id_to_token)).to(device)
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
             beam_width=10):
    print(f"---- STARTING PROMPT: {prompt} ----")
    if len(prompt) > max_len:
        raise ValueError("Prompt too long")
    beams = [(prompt, 0.0, prompt)]

    for _ in range(iterations):
        new_beams = []
        
        for seq, cum_log_prob, full_seq in beams:
            int_seq = tokenizer.tokenize(seq)
            mask = [1] * len(int_seq) + [0] * (total_length - len(int_seq))
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
                new_cum_log_prob = cum_log_prob + topk_probs[:PREDICT_N_TOKENS_AT_A_TIME, i].sum().item() #topk_probs[0, i].item()
                full_seq += tokenizer.decode(best)
                if len(int_seq + best) >= max_len:
                    new_seq = tokenizer.decode((int_seq + best)[PREDICT_N_TOKENS_AT_A_TIME + 1:]) # Extra one as at least one padding token
                new_beams.append((new_seq, new_cum_log_prob, full_seq))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return [(x[2], round(x[1], 3)) for x in beams]

# print(generate("abcdefghi", 30))
# print(generate(" abcdefgh", 30))
# print(generate("abc", 22))
# print(generate("abcd", 25))
# print(generate("wxyz", 15))

print(generate("the quick", 25))
print(generate("puck or as he", 25))
print(generate("puck or as he w", 25))
print(generate("goodf", 20)) # Robin Goodfellow was a shrewd
print(generate("goodfe", 20)) # Robin Goodfellow was a shrewd
print(generate("goodfel", 20)) # Robin Goodfellow was a shrewd
print(generate("goodfell", 20)) # Robin Goodfellow was a shrewd
print(generate("goodfello", 20)) # Robin Goodfellow was a shrewd
print(generate("goodfellow", 20)) # Robin Goodfellow was a shrewd
print(generate("goodfellow ", 20)) # Robin Goodfellow was a shrewd
print(generate("goodfellow w", 20)) # Robin Goodfellow was a shrewd
print(generate("goodfellow wa", 20)) # Robin Goodfellow was a shrewd
print(generate("goodfellow was", 20)) # Robin Goodfellow was a shrewd
print(generate("robin ", 20)) # Robin Goodfellow was a shrewd
print(generate("robin goo", 20)) # Robin Goodfellow was a shrewd
print(generate("comical", 20)) # comical pranks
print(generate("comical ", 20)) # comical pranks
print(generate("comical p", 20)) # comical pranks
print(generate("comical pr", 20)) # comical pranks

import sys; sys.exit(1)

# Now modify it, the example data should change so that the next token is the sum (still), the next one is that sum + 1 (%27) and again + 1 for the next one
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
    examples = generate_data(10, max_seq=10, max_len=7)
    ppo = PPO(model, examples)
    ppo.train(examples, timesteps=5000)
    ppo.eval()

# Example usage
corpus = "The quick brown fox jumps over the lazy dog a b c d e f g h i j k l m n o p q r s t u v w x y z"
corpus_tokens = corpus.split()  # or any list of tokens

tokenizer = TrieTokenizer()
tokenizer.fit(corpus_tokens)

test_text = "The quick brown fox y z boab"
encoded = tokenizer.tokenize(test_text)
decoded = tokenizer.decode(encoded)

print("Encoded IDs:", encoded)
print("Decoded text:", decoded)