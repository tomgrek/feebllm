# FeebLLM

# Status

Attention works, in batches, to predict sequences :)
It's learning the corpus, but not doing a fantastic job at that. Code is refactored to enable different seq lengths and num predicted tokens. Predicting a single token seems to be better.
Learns and generates successfully (`tokenbeam.py`) with variable length sequences, trie tokenization (variable length), CUDA, beam search
qp.py WORKS

# TODO

It did work not long ago. All I did since then was mess around with the reward function. (To revert maybe try setting in-episode reward to 0.0 not giving interim rewards. I think it has an "aha" moment. TODO get the normalize function working properly, right now it normalizes the entire array, but it should norm over the sequence for each batch element. TODO check the mask, is it masking the right thing? TODO check the .probs of the categorical dist. TODO add entropy

bibbity. Both losses are going down (over time) in crossqp.. but it's not great still.

Nah, value and policy loss are both just bobbling around pretty much. Find the script/commit where they were actually going down consistently, figure out where I went wrong. [UPDATE: well now it's working semi decently!?!?]

Retrograde progress today. Added additional generate method to head_shared that's like the collect_trajectories one, but it's no better. Faffed with params. It's back working again (I think), but only
cos I reinstated the aax etc training examples to the original corpus. Next I should rewind the other changes (see git diff - temp and epsilon and annealing) and see if they help.

Still works: added temp annealing. Higher epsilon seems good/necessary; clip_epsilon really does its thing but keeping it around 0.2, or higher, seems fine.
There is some diffs between the generate method and how the collect_trajectories method works. Just added epsilon annealing too, unknown the results yet. [EDIT: seems BAD]
Feel like if I can get it to self generate the `action = random.choice(["x", "a", "b", "a", "x"])` if epsilon bit then that's AGI... if I can get it to backtrack when 
it gets worse, that's an even stronger AGI....

IS IT POSSIBLE THAT HEAD_SHARED ACTUALLY DID WORK, BUT FOR MOST THINGS IN THE BATCH THE REWARD/RETURN WAS 0?? LIKE WHAT IF I DID THE GAMMA DISCOUNT INSIDE GATHER_TRAJECTORIES SO THEN
I KNOW IT WORKED FOR EVERY SINGLE STATE/ACTION PAIR???

I don't trust any of it. head_shared worked for a minute, ish, but clearly isn't truly working. I have tried to make everything bs x seq_len x whatever so the advantages/returns calculation works, WIP


Well linear.py and expt.py didn't work cos they were doing nothing to calculate returns and advantages...shape was wrong. Gave up on linear. Fixed it in expt but sceptical that it really works.
uhhhh looks like (fri) expt really dun work... since it can't generate "h i" but only "d e" which is weird. (partially cured by removing "a b c d e d e ..." from the training data but it still dun work).
And why it likes to generate double spaces so much, particularly at the FIRST generation?

Linear.py is expt but without the value.mean() "bug" ie not doing value_loss = (returns - value.squeeze(-1).mean(dim=-1)).pow(2).mean(), value is the right shape so i could just subtract.
Doesn't work better than expt.py. Also, in expt, normalization of advantages seems bad. Jury still out on normalizing rewards.



expt works and works with batches AND this is now with a shared model/head too. Important: temp set to 1.0 and have SOME EXAMPLE
OF WHAT YOURE TRYING TO TEACH IT IN THE INITIAL TRAINING SET. TODO check it works for multiple reward functions! Then clean up and done!

So yeah, rewards is just a single number in get_batch whereas it's the whole sequence in qp (at the start of the compute_advantages fn). I need to fix the for loop so it only appends once the whole
sequence is done. Additionally, I should run that for loop not just once for each prompt, but multiple times, so the sampling picks up some different things.

qp is still working. in head_shared.py i went back to using an optim just on the value head, 
because i def thought it wasn't working on the whole model. Then from there I went back to using the same value model as qp.py. But it still isn't working: see
how the value loss doesn't actually go down over 100 epochs. Whereas in qp.py it does. So
something is wrong... I tried to make it identical but there is some difference. Step through
the code when batch size = 1 and figure out what!


shared.py seems working, although i need to validate with good chars and also that
with bs=1 the numbers are the same, at least for the first iteration of policy loss.
SEEMS THAT we just do (policy_loss + value_loss).backward(), so one optimizer for the whole
parameters.

Negative rewards are ok. Reward normalization was bad. qp is working, I think, though I haven't
let it run for many epochs. It def worked for whitespace. Seems ok for chars but again, let it run.
Then retry shared/unshared.py with 1 prompt and the new temperateure stuff and the fixed rewrad
stuff (no normalization/temperature/using mean for the value function instead of last logit)

Having the reward positive seems important. Also, the normalize reward function seems to be
what makes the policy loss negative. Large PPO update_epochs tells me whether the loss is going
down or not, or just oscillating (which it is for shared/unshared). So, they are totally broke,
qp.py isn't?? but needs lots of work. value loss is way high, for example. Although it
comes down a lot over repeated epochs of epochs.
ALSO REALLY TRY TO KEEP REWARD POS AND <1.0. LARGE REWARD MAKES THE VALUE LOSS BLOW UP.

Working on unshared.py. It just stopped running because I'm making crazy changes. It clearly doesn't work, except for whitespace.
Convince myself: does qp.py actually really work? With rewards other than +/- whitespace? Is everything fine apart from small
data and sparse rewards?

Working on shared.py. It "works" now in that it runs - with batches. But it doesn't seem to learn any useful thing. 
Although on the other hand, very last thing tonight, it did seem to progress upwards very rapidly over several epochs. I changed
the params to make it learn faster.... they may be a bit crazy. Yea saw that with both + and - whitespace.

* Make it blinking work with batches
* Share the embeddings
* Play with clipping param
* Transpose the signal in the value model to get a single output value rather than just using the final token
* Add a value head to the policy model

* Positional embeddings may lose meaning when its possibly single tokens and possibly whole words!?!?
* Convert the get_batch into a dataloader with pin memory.

* PPO only maybe sorta works UPDATE: it actually doesn't seem to work at all
* Trying to troubleshoot... sadly a bit hit and miss. Start with 4 in the chat (value network update). I think ratio/log_prob/old_log_prob looks ok now.


* PPO doesn't use batches
* PPO doesn't respect multi token prediction
* I don't feel good about the advantage.clone.detach

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