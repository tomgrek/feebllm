# FeebLLM

# Status

Attention works, in batches, to predict sequences :)

# TODO

* Clean up. Only generate the tensors (seq, mask, target -- and make it the ACTUAL target !!!) in one place.
* Have it actually learn from a corpus and generate some text. This is interesting cos it's a multi-token output whereas LLMs are usually single token autoregressive... which means temperature/sampling sort of makes less sense.. but still poss..
* Add PPO
* Change PPO to GRPO

# Learnings
* No softmax in the last layer if using cross entropy loss
* Masking is helpful
* The positional encodings really work and are necessary.
* padding_idx=26 in the embedding definition was unhelpful, it learns faster without that. Maybe cos it's not really "padding" the way I'm using it, rather, the 26 token denotes end of sequence.

# The original dream

PPO should be working... 

kind of seems it is, but I think my problem is that the network is predicting 10 tokens
and I sort of assumed that some of them are the input. BUT THEY ARE NOT!!!!!!!!!!!!!!

It's predicting the next 3 tokens, so I should just look at only those. Then I can forget
the crazy diffs masking stuff.