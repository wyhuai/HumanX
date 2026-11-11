# Experimental multi-GPU HOI PPO

This prototype shards Isaac Gym HOI environments across GPUs, gathers each rollout to rank 0, performs the usual PPO update there, and broadcasts actor/critic parameters. It aggregates completed-episode statistics before updating the supported curricula. It is **experimental**, not a drop-in replacement for the existing trainer: only the basketball HOI configuration below has been exercised; adaptive motion sampling, training-time motion resampling and BC warmstart are rejected. Main-process PPO and replicated all-gather buffers can limit memory and scaling. Eight GPUs have not been tested.

From `xmimic`, with the repository's Isaac Gym/Python environment installed and the example motion in place:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HUMANX_TRAIN_OUTPUT=/absolute/path/to/a/new/output-directory
export HUMANX_TOTAL_ENVS=16384 HUMANX_ITERATIONS=200
bash experimental/multigpu_hoi/launch.sh checkpoint=/absolute/path/model_10000.pt
```

The total environment count must be divisible by the selected GPU count. Omit the checkpoint override for a fresh run. The trainer saves one final checkpoint on rank 0; stdout contains `HUMANX_TRAIN_METRICS` JSON. Optional `HUMANX_CURRICULUM_SNAPSHOT` points to a trusted upstream `last.pkl`, and `HUMANX_CURRICULUM_AVERAGE` sets an estimated episode-length EMA. Upstream checkpoints omit simulator and RNG state and the EMA, so resumed trajectories cannot exactly reproduce an uninterrupted run. Curriculum update ordering differs from upstream; a short continuation does not establish training-to-convergence or task success equivalence. Standard logging, periodic checkpointing, exact resume and fault recovery are not implemented here. Do not use untrusted pickle files.

For diagnostics, set `HUMANX_VERIFY_MODEL_SYNC=1` and `HUMANX_VERIFY_CURRICULUM=1`; checks introduce overhead. The example launcher uses one physical GPU per process through `CUDA_VISIBLE_DEVICES`. Avoid running other work on those GPUs during timing. CPU test (with the repository's PyTorch dependencies):

```bash
python experimental/multigpu_hoi/test_curriculum.py
```

## Measured continuation

A basketball catch-and-throw teacher checkpoint from iteration 10,000 was continued for 200 iterations using four A800 80 GB GPUs, 4096 environments per GPU, 24 steps per environment, central PPO and global advantage normalization. The single-GPU baseline was an already-running continuation on a different A800 server, using 16384 environments. Both used the same task, checkpoint lineage and nominal total rollout batch. The four-GPU run used this branch's headless/reference changes and its curriculum synchronization; the single-GPU run used upstream code. This is a practical comparison, not a same-host, identical-code A/B test. The resumed experiment restored saved curriculum scalars and approximated the missing episode-length EMA with 177.6272 from the original run's logged iteration-10000 mean. No simulator/RNG state was restored.

| Median over iterations 10181–10200 | Single GPU | Four GPUs |
| --- | ---: | ---: |
| Iteration time | 10.785 s | 4.449 s |
| Mean episodic reward | 10.09 | 10.28 |
| Mean episode length | 178.10 | 180.03 |

Four-GPU time includes synchronization, gathering and PPO update, but excludes initialization, ordinary full trainer logging and checkpoint writes; the baseline is upstream's logged collection-plus-learning iteration time. Timings are from different servers and one run each. The four-GPU reward/length are rank-0 rolling 100-episode statistics; the baseline uses its own rolling 100 episodes. The observed per-iteration time was 58.7% lower (2.42x nominal transition throughput); these figures do not establish equivalent success rates, convergence or end-to-end training time. The four-GPU 198 measured iterations after two warmups had median 4.526 s and nominal 86,871 transitions/s.
