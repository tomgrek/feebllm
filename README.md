# FeebLLM

# Status

Attention works, in batches, to predict sequences :)

# TODO

* Figure out whether padding_idx=26 in the embedding definition is useful or a hinderance.
* Clean up. Only generate the tensors (seq, mask, target -- and make it the ACTUAL target !!!) in one place.
* Add PPO
* Change PPO to GRPO

# Learnings
* No softmax in the last layer if using cross entropy loss
* Masking is helpful
* The positional encodings really work and are necessary.

# The original dream

PPO should be working... 

kind of seems it is, but I think my problem is that the network is predicting 10 tokens
and I sort of assumed that some of them are the input. BUT THEY ARE NOT!!!!!!!!!!!!!!

It's predicting the next 3 tokens, so I should just look at only those. Then I can forget
the crazy diffs masking stuff.