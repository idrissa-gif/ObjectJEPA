# ObjectJEPA

**An object-centric world model that predicts the future in latent space and answers "what if" questions about physical scenes.**

Built for the Meta Global AI Developer Hackathon (2026), Indie Developer track, "Push the Frontier".

ObjectJEPA watches a short video of objects moving and colliding ([CLEVRER](http://clevrer.csail.mit.edu/)) and forecasts what happens next. It does not predict future pixels: it predicts future **representations**, the JEPA way. A frozen [V-JEPA 2](https://github.com/facebookresearch/vjepa2) encoder turns video into latent features, and a small transformer predictor learns to roll those latents forward in time. On top of that, object masks from SAM 3 give one latent per object, and an LLM agent (Muse Spark) uses the world model as a tool to answer predictive and counterfactual questions ("what happens if the red cube is removed?").

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

## Results

All numbers are on held-out **validation videos** (split by video; 1,000 of the 10,000 CLEVRER training videos, never seen in training). L1 is in normalised latent space, lower is better.

**Stage 2: frame-level latent prediction** (2,000 val clip pairs, 8×8 grid, ViT-L)

| model | val L1 | cosine |
|---|---|---|
| **predictor** (width 512, depth 8, 60 epochs) | **0.537** | **0.721** |
| predictor (width 384, depth 6, 30 epochs) | 0.563 | 0.692 |
| mean of context | 0.707 | |
| copy_clip (repeat the whole context clip) | 0.766 | |
| copy_last | 0.921 | 0.322 |
| linear extrapolation | 4.350 | |

The predictor beats every baseline from epoch 1 on, and train and val L1 are equal (no overfitting).

**Stage 3: object-centric prediction** (one latent per object, from object masks)

| | val L1 | object position error |
|---|---|---|
| **object predictor** (CLEVRER masks) | **0.581** | **4.6 px** |
| same model on **SAM 3** masks (200 val videos) | 0.589 | 4.3 px |
| mean / constant velocity | 0.683 | 16.3 px |

SAM 3 masks match the CLEVRER ground-truth masks with recall 0.991, precision 0.972 and mean IoU 0.932. The predictor is trained on CLEVRER masks only and transfers to SAM 3 masks with almost no loss.

**Stage 4: "what if" questions** (1,858 CLEVRER counterfactual questions on the val videos, "which event happens if the cyan sphere is removed?")

| method | per option | per question |
|---|---|---|
| **world model + collision head + learned update** (remove the object's latents, roll out, read the change with the learned update) | **0.844** | **0.564** |
| world model + collision head, delta rule (apply the change between the two rollouts to the observed events) | 0.837 | 0.530 |
| no change (answer with the events of the real video, read by the head) | 0.830 | 0.522 |
| raw rollout (collisions of the counterfactual rollout alone) | 0.775 | 0.377 |
| first version: distance rule on positions, delta rule | 0.791 | 0.438 |
| first version: no change (distance rule) | 0.784 | 0.433 |

**Full agent on SAM 3 masks, no ground truth at all** (373 questions of 200 val videos; per option / per question)

| agent | distance rule | + collision head | + learned update |
|---|---|---|---|
| offline parser (rule-based, no LLM) | 0.745 / 0.357 | 0.778 / 0.410 | 0.786 / 0.453 |
| **Muse Spark** (`muse-spark-1.3` calling the tools) | 0.771 / 0.397 | 0.802 / 0.456 | **0.808 / 0.491** |

Both agents use the same tools and the same world model; only the model that reads the question and calls the tools differs. With the full pipeline, Muse Spark beats the offline parser by 2.2 points per option and 3.8 per question. Each 373-question Muse Spark run costs about $5.50 (≈1.36 M input / 0.9 M output tokens).

**Collision head** (`collision_head.py`). The first version read collisions out of predicted positions with a distance rule, and its gain over "no change" was small (p = 0.0025): a better forecaster barely moved the score, so the bottleneck was the read-out, not the latent prediction. The head is a small temporal convolution over pairs of object latents plus geometry, trained on CLEVRER's annotated collisions on both real and predicted sequences (8,000 train videos; 1,000 other train videos pick the epoch and threshold). On predicted future steps it raises event F1 from 0.762 (distance rule) to 0.898.

**Learned update** (`whatif_update.py`). A small gradient-boosted classifier replaces the delta rule. Per object pair it sees only world-model signals: the head's scores on the real video, the factual rollout and the counterfactual rollout. It is fitted on the counterfactual questions of the 1,000 selection videos and beats the delta rule on val (paired sign test on the options where they disagree, p = 0.04). Not used, on purpose: cues about the removed object's own contacts in the real video. A classifier on those alone reaches 0.88 on the selection videos (a CLEVRER shortcut: "if the removed object hit A first, A's next collision changes"), and the rollouts add nothing on top of it, so it would not measure the world model.

## Roadmap

| Stage | What | Status |
|---|---|---|
| 1 | Frozen V-JEPA 2 feature extraction (sharded) | ✅ 10,000 videos |
| 2 | Latent predictor vs. the baselines | ✅ beats all baselines |
| 3 | **Object-centric**: SAM 3 masks, features pooled per object | ✅ |
| 4 | **"What if" tools + agent**: remove an object's latents, roll out, compare collisions | ✅ collision head, learned update, offline parser and Muse Spark (`agent.py --backend chat`, `demo_live.py`) |
| 5 | Demo videos (observed / predicted / what-if) | ✅ `visualize.py` |

## Files

| File | Role |
|---|---|
| `extract_features.py` | Stage 1. Reads CLEVRER `.mp4`s, cuts context/target clip pairs, encodes each separately with V-JEPA 2, pools patch tokens to a grid, writes one `.pt` per video. `--shard/--num_shards` runs one process per GPU. |
| `train_predictor.py` | Stage 2. Transformer predictor, L1 loss in normalised latent space, DDP via `torchrun`. Prints baselines every epoch. Saves `predictor_best.pt` and `history.json`. |
| `run.sh` | Download → extract on 2 GPUs → train on 2 GPUs. |
| `sam_masks.py` | Stage 3. SAM 3 object masks and tracks for each video. |
| `extract_object_features.py` | Stage 3. Pools V-JEPA 2 patch tokens inside each object mask → one latent per object (CLEVRER or SAM masks). |
| `train_object_predictor.py` | Stage 3. Transformer predictor over object latents and positions. |
| `eval_motion.py` | Evaluation on moving vs. static patches. |
| `whatif.py` | Stage 4. Counterfactual rollouts and the CLEVRER counterfactual-question evaluation (`--head --head_fact --update` for the learned read-out). |
| `collision_head.py` | Stage 4. Learned collision head: per pair and step contact logits from object latents + geometry. |
| `whatif_update.py` | Stage 4. Learned what-if update over the head's scores on the real video and both rollouts. |
| `agent_tools.py` | Stage 4. Local tools for the agent: `describe_objects`, `observed_collisions`, `what_if_remove`. |
| `agent.py` | Stage 4. Agent loop. `--backend offline` (rule-based, no API) or `--backend chat` (any OpenAI-compatible endpoint, e.g. Muse Spark). |
| `demo_live.py` | Live demo: asks the agent one question and prints each tool call and result as it happens, then checks the answer against CLEVRER. `--offline` for no API call. |
| `visualize.py` | Demo videos: observed, predicted and what-if, with object labels and the answer. |

## Setup and running

Requirements: Python 3.10+, PyTorch with CUDA, `transformers>=4.53`, `av`, `numpy` (and `scikit-learn` for the learned update). Tested on 2× RTX A6000 (48 GB).

```bash
pip install torch torchvision "transformers>=4.53" av numpy
bash run.sh 500      # smoke run on 500 videos
bash run.sh 0        # full ~10k training videos
```

To run inside a conda env, use `CONDA_ENV=myenv bash run.sh`. Data goes to `~/objectjepa_data` (override with `DATA=/path bash run.sh`). The CLEVRER train zip is 12.4 GB; plan for ~60 GB including features.

### Notes

- **Full frame, not center crop.** The default V-JEPA 2 processor center-crops to 256×256, which drops ~40% of the width of a 480×320 CLEVRER frame (objects entering from the sides disappear). `extract_features.py --frame squash` (the default) resizes the whole frame to 256×256 instead. `--frame crop` restores the processor default.
- **Agent with an LLM**: `export META_API_KEY=...` (never commit it), then `python agent.py --backend chat --base-url <endpoint>/v1 --model <model> --features ... --ckpt ... --videos val_videos.txt --limit 10`. It prints the token count and an estimated cost, and stops once `--max-usd` (default 1) is reached.
- **Collision head and learned update**: `python collision_head.py --features obj_features_clevrer --ckpt ckpt_obj --val_videos val_videos.txt --out ckpt_collision`, then `python whatif_update.py --features obj_features_clevrer --ckpt ckpt_obj --head ckpt_collision --videos ckpt_collision/dev_videos.txt --out ckpt_update` (the head's selection videos, written by `collision_head.py`; never val). Pass `--head ckpt_collision --update ckpt_update` to `agent.py` and `demo_live.py` (and `--head_fact` to `whatif.py`).
- **Knobs worth trying**: bigger predictor (`--width 512 --depth 8`), finer grid (`--pool 16`), bigger encoder (`--model facebook/vjepa2-vitg-fpc64-256`, 1408-d; the trainer reads the dimension from the data).

## What was built during the hackathon (Oct 26 – Nov 4, 2026)

All code in this repository was written during the event (first commit Oct 27, 2026):

- the feature extraction pipeline (separate context/target encoding, full-frame resize, pooling, sharding) and the frame-level latent predictor with its baselines;
- the object stage: SAM 3 mask extraction (`sam_masks.py`), per-object V-JEPA 2 latents and the object predictor;
- the "what if" rollout and collision read-out (`whatif.py`), the learned collision head (`collision_head.py`) and what-if update (`whatif_update.py`), the agent tools (colour, shape and material from pixels) and the agent loop with the Muse Spark backend;
- the evaluation (baselines, sign test, SAM vs ground-truth masks) and the demo renderer (`visualize.py`).

Used as-is, not built here: the pretrained V-JEPA 2 and SAM 3 weights (loaded through Hugging Face `transformers`), the Muse Spark model through the Meta Model API, and the CLEVRER dataset.

## Acknowledgements

- V-JEPA 2, SAM 3 and Muse Spark by Meta.
- CLEVRER dataset: Yi, Gan, et al., *CLEVRER: CoLlision Events for Video REpresentation and Reasoning*, ICLR 2020.
  Released under CC0 ([README](http://data.csail.mit.edu/clevrer/README.txt)).

## License

Code: MIT (see `LICENSE`). The pretrained models and the CLEVRER dataset keep their own licences.
