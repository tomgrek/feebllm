# FeebLLM

# Status

Attention works, in batches, to predict sequences :)
It's learning the corpus, but not doing a fantastic job at that. Code is refactored to enable different seq lengths and num predicted tokens. Predicting a single token seems to be better.
Learns and generates successfully (`tokenbeam.py`) with variable length sequences, trie tokenization (variable length), CUDA, beam search

# TODO

* Positional embeddings may lose meaning when its possibly single tokens and possibly whole words!?!?
* Convert the get_batch into a dataloader with pin memory.
* Lines 123 and 114 are a bit sus

* Add PPO
* Change PPO to GRPO

# Learnings
* No softmax in the last layer if using cross entropy loss
* Masking is helpful
* Batch norm (in the tranformer blocks or various parts of the model) is not helpful
* Learned positional encodings are much better than sinusoidal
* padding_idx=26 in the embedding definition was unhelpful, it learns faster without that. Maybe cos it's not really "padding" the way I'm using it, rather, the 26 token denotes end of sequence.

# The original dream

PPO should be working... 

kind of seems it is, but I think my problem is that the network is predicting 10 tokens
and I sort of assumed that some of them are the input. BUT THEY ARE NOT!!!!!!!!!!!!!!

It's predicting the next 3 tokens, so I should just look at only those. Then I can forget
the crazy diffs masking stuff.