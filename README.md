# FeebLLM

# Status

Attention works, in batches, to predict sequences :)
It's learning the corpus, but not doing a fantastic job at that. Code is refactored to enable different seq lengths and num predicted tokens. Predicting a single token seems to be better.


# TODO

* Instead of limiting the vocab, have it only pick training examples from the text where the previous token is a whitespace. That will ensure
it's not splitting words and gunking up the learning with sometimes full words and sometimes individual tokens.
* Positional embeddings may lose meaning when its possibly single tokens and possibly whole words!?!?
* Batch norm might be helpful?? See chat. Probably not since batches are vee large now so the noise is low.
* WIP Make the generate method work with multiple tokens. It "works" now, but not well cos it only concatenates one. ADDENDUM: I actually made
it work last thing with the `num_predicted_tokens` parameter. But I totally didn't do any validation that it's working correctly.
* Convert the get_batch into a dataloader with pin memory.


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