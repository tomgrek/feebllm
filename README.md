# FeebLLM

Single file implementation of a "large language model" with tiny training data. Super quick to experiment hands-on and test different ideas.

Model is trained using the masked next token prediction objective and is a fairly realistic transformer architecture. Training as-is takes under a minute on GPU; early stopping ensures it doesn't overfit (well, somewhat).

Model can then predict via beam search, given a starting prompt.

Next, model is trained with reinforcement learning (PPO) to optimize a simple objective: whatever the prompt, lots of `b c` in the generation maximizes the reward. This is obviously very different from its training distribution but it still gets there. Hyperparams have a big effect. Value model is just a linear head on the transformer (policy) model, which works but doesn't have many degrees of freedom.

## Usage

```python
python main.py
```

Will use CUDA, if available. Batch size is defined in the code but you won't have a VMEM problem.

The other files in this repo are just various iterations working towards `main.py`, plus the lab notes of a madman in `README_old.md`. There are some different ideas and some blind alleys.

## Next Steps

* Interesting effect that beam search is (much) better after a small amount of PPO compared to no beam search, but the effect reverses after lots of PPO (when straight sampling is better and beam search is rubbish).
* Epsilon exploration for PPO is interesting, particularly if you could optimize the model's curiosity.
* A better value model or head.
* Definitively answer whether final reward being "standalone" or "cumulative" is better.
* Continous reward function (ie a model) would be way better, sparse rewards were difficult to get working.
* Turn it from an experimental fun side project into a bit less of a spaghetti code.