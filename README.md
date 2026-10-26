# ObjectJEPA

**An object-centric world model that predicts the future in latent space and answers "what if" questions about physical scenes.**

Built for the Meta Global AI Developer Hackathon (2026).

ObjectJEPA watches a short video of objects moving and colliding ([CLEVRER](http://clevrer.csail.mit.edu/)) and forecasts what happens next. It does not predict future pixels: it predicts future **representations**, the JEPA way. A frozen [V-JEPA 2](https://github.com/facebookresearch/vjepa2) encoder turns video into latent features, and a small transformer predictor learns to roll those latents forward in time. On top of that, object masks from SAM 3.1 give one latent per object, and an LLM agent (Muse Spark) uses the world model as a tool to answer predictive and counterfactual questions ("what happens if the red cube is removed?").

## How it works

```
  CLEVRER video (128 frames, 480×320)
          │
          ├── context: frames  0–31 (every 2nd → 16 frames) ──┐
          └── target : frames 32–63 (every 2nd → 16 frames) ──┤  encoded SEPARATELY
                                                              ▼  (no future leakage)
                                     frozen V-JEPA 2 ViT-L encoder
                                     tokens: 8 time steps × 16×16 patches × 1024-d
                                                              │
                                              avg-pool to 8×8 grid
                                                              ▼
                       ctx latents [8, 64, 1024]  ──►  JEPA predictor  ──►  predicted tgt latents [8, 64, 1024]
                                                     (transformer,             compared to the real tgt latents
                                                      residual on last          with L1 loss
                                                      context step)
```

- **Encoder** (`facebook/vjepa2-vitl-fpc64-256`) is frozen, so targets cannot collapse.
- **Predictor** is a transformer over context tokens plus learnable future-query tokens. Its output is a residual added to the last observed step, so at initialisation it equals the copy-last baseline and only has to learn the *change*: motion, collisions, objects entering.
- **Evaluation** is on a validation split **by video** (no clip of a validation video is ever seen in training), against three baselines:
  - `copy_last`: the scene freezes at the last context step,
  - `linear`: linear extrapolation of the last two steps,
  - `mean`: the mean of the context steps.

  The predictor has to beat all three. The training log prints `BEATS` when it does.

## Roadmap

| Stage | What | Status |
|---|---|---|
| 1 | Frozen V-JEPA 2 feature extraction (2 GPUs, sharded) | ✅ verified, running on full dataset next |
| 2 | Latent predictor training (DDP) vs. the 3 baselines | ⏳ smoke run next |
| 3 | **Object-centric**: SAM 3.1 masks, so features are pooled per object instead of per grid cell | planned |
| 4 | **Surprise signal**: prediction error spikes at collisions (intuitive-physics probe) | planned |
| 5 | **"What if" agent**: Muse Spark calls the world model (remove or alter an object's latents, roll out, compare) to answer CLEVRER predictive and counterfactual questions | planned |
| 6 | Gradio demo + video | planned |

## Files

| File | Role |
|---|---|
| `extract_features.py` | Stage 1. Reads CLEVRER `.mp4`s, cuts context/target clip pairs, encodes each separately with V-JEPA 2, pools patch tokens to a grid, writes one `.pt` per video. `--shard/--num_shards` runs one process per GPU. |
| `train_predictor.py` | Stage 2. Transformer predictor, L1 loss in normalised latent space, DDP via `torchrun`. Prints baselines every epoch. Saves `predictor_best.pt` and `history.json`. |
| `run.sh` | Download → extract on 2 GPUs → train on 2 GPUs. |

## Setup and running

Requirements: Python 3.10+, PyTorch with CUDA, `transformers>=4.53`, `av`, `numpy`. Tested on 2× RTX A6000 (48 GB).

```bash
pip install torch torchvision "transformers>=4.53" av numpy
bash run.sh 500      # smoke run on 500 videos
bash run.sh 0        # full ~10k training videos
```

To run inside a conda env, use `CONDA_ENV=myenv bash run.sh`. Data goes to `~/objectjepa_data` (override with `DATA=/path bash run.sh`). The CLEVRER train zip is 12.4 GB; plan for ~60 GB including features.

### Notes

- **Full frame, not center crop.** The default V-JEPA 2 processor center-crops to 256×256, which drops ~40% of the width of a 480×320 CLEVRER frame (objects entering from the sides disappear). `extract_features.py --frame squash` (the default) resizes the whole frame to 256×256 instead. `--frame crop` restores the processor default.
- **Knobs worth trying**: bigger predictor (`--width 512 --depth 8`), finer grid (`--pool 16`), bigger encoder (`--model facebook/vjepa2-vitg-fpc64-256`, 1408-d; the trainer reads the dimension from the data).

## Acknowledgements

- V-JEPA 2, SAM 3.1 and Muse Spark by Meta.
- CLEVRER dataset: Yi, Gan, et al., *CLEVRER: CoLlision Events for Video REpresentation and Reasoning*, ICLR 2020.
