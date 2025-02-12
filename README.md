# FeebLLM

# Status

Attention works, in batches, to predict sequences :)
It's learning the corpus, but not doing a fantastic job at that. Code is refactored to enable different seq lengths and num predicted tokens. Predicting a single token seems to be better.


# TODO

* Look at the generate() output after it reaches the max length. The last token isn't set correctly, it is just set to PADDING_IDX. Mask is fine. FIX!!!!!
* Get it generating the alphabet, at least...
* Add multiple transformer blocks... see chat...
* Have it actually learn from a corpus and generate some text. This is interesting cos it's a multi-token output whereas LLMs are usually single token autoregressive... which means temperature/sampling sort of makes less sense.. but still poss..
* Add PPO
* Change PPO to GRPO

# Learnings
* No softmax in the last layer if using cross entropy loss
* Masking is helpful
* The positional encodings really work and are necessary.
* * Learned positional encodings are much better than sinusoidal
* padding_idx=26 in the embedding definition was unhelpful, it learns faster without that. Maybe cos it's not really "padding" the way I'm using it, rather, the 26 token denotes end of sequence.

# The original dream

PPO should be working... 

kind of seems it is, but I think my problem is that the network is predicting 10 tokens
and I sort of assumed that some of them are the input. BUT THEY ARE NOT!!!!!!!!!!!!!!

It's predicting the next 3 tokens, so I should just look at only those. Then I can forget
the crazy diffs masking stuff.