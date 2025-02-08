# FeebLLM

# Status

LSTM worked.
Attention works too but seems slower to train.
Softmax makes LSTM slower to train.
* Make it work in batches
* Works with sequence lengths of 3


# TODO

* Make it work with arbitrary sequence lengths (i.e. 3 token outputs)

It's possible to add one, as long as all 3 targets are the same value. It's no good at learning changing sequences.

# The original dream

PPO should be working... 

kind of seems it is, but I think my problem is that the network is predicting 10 tokens
and I sort of assumed that some of them are the input. BUT THEY ARE NOT!!!!!!!!!!!!!!

It's predicting the next 3 tokens, so I should just look at only those. Then I can forget
the crazy diffs masking stuff.