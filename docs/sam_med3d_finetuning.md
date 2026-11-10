# Fine-tuning SAM-Med3D on fish: an experiment

**This was an experiment, not a real fine-tuning of the model.** Goal was to
answer if fine-tuning worth pursuing with the available data. It used three
fish, trained only the decoder, and ran one fixed recipe without searching for
better settings. The tuned weights it produced are for comparison, not for use.

**Result:** a decoder tuned on a fish segments that same fish well, but not a
fish it has not seen. There it is worse than the published model in 5 of 6
cases. The published weights (`weights: standard`, the default) remain the
version to use.

## Approach

- **What was trained:**
  - only the prompt encoder and the mask decoder (7.6 M parameters);
  - the image encoder (93 M) stayed frozen, so each training cube was encoded
    once and cached.
- **Model, weights and loss:** SAM-Med3D's own model code and published
  weights (`sam_med3d_turbo.pth`), with its loss (Dice + cross-entropy). The
  training loop is our own.
- **Samples:** 256 cubes per fish, cut as `sam_med3d` cuts them at inference.
  - The box varies (grown 10-50%, shifted up to 10%).
  - Each cube is flipped, rotated and scaled, and its brightness and noise
    vary.
  - Every step picks a new click inside the bladder.
- **Recipe**, fixed before any test result was seen:
  - AdamW, lr 8e-5, weight decay 0.1;
  - 2000 steps of 4 cubes;
  - no early stopping.
- **Synthetic bladders** (arm B only):
  - the real bladder's shape, deformed, placed elsewhere in the same fish's
    tissue;
  - one third of the training cubes;
  - only from Gadus and the tuna, because none fit in Oreo's small scan.
- **Test:** 9 clicks per fish, with a pass when:
  - the median Dice is ≥ 0.70;
  - at least 7 of the 9 clicks give a Dice ≥ 0.50;
  - the median HD95 is ≤ 10% of the bladder's length.
- **Human control:** one human CT (AMOS `amos_0013`, 15 organs), to see what
  tuning on fish costs the model on its own domain.

## Results

**Stage 1, trained and tested on the same fish** (a sanity check). Median Dice,
with the published weights → after tuning on that fish:

| Oreo | Gadus | Bluefin tuna |
|---|---|---|
| 0.664 → **0.967** | 0.486 → **0.909** | 0.567 → **0.830** |

The tuned results are memorisation: the 9 clicks give almost the same Dice
wherever they are.

**Stage 2, trained on two fish and tested on the third ("lofo": leave one fish
out).** Median Dice, with the clicks reaching ≥ 0.50:

| Left-out fish | Published | Tuned: real only | Tuned: + synthetic |
|---|---|---|---|---|
| Oreo | 0.664, 9/9 | 0.537, 9/9 | 0.757, 7/9 |
| Gadus | 0.486, 0/9 | 0.286, 0/9 | 0.225, 0/9 |
| Bluefin tuna | 0.567, 9/9 | 0.132, 0/9 | 0.208, 0/9 |

- **No tuned model passes, and none improves** on the published one. An
  improvement needed a median gain of at least 0.05, with no fewer clicks
  reaching 0.50.
- **The one rise (Oreo, synthetic)** is not an improvement: 2 clicks fell to
  about 0.27, and its HD95 failed.
- **The tuna:** trained on two dark (gas-filled) bladders, the model loses the
  tuna's bright one.
- **On the human CT,** the median Dice falls from 0.586 to 0.10-0.36.
- **Tuning the image encoder too** was estimated and not run. With no gain from
  the decoder, it would most likely memorise even more.

## Would better data help?

Possibly. Stage 1's gains are not evidence for it, because they were measured
on the training fish.

**What stage 2 points to:** each left-out fish was also a different kind of
scan:

- mass density at 3 mm (Oreo);
- micro-CT at 0.41 mm (Gadus);
- medical-style CT (tuna).

So the model was never tested on a scan type it had trained on. The synthetic
bladders gave the only gains, which suggests that variety helps.

**A real fine-tuning would need:**

- a dozen or more fish of the same scan type, each with a reference mask;
- both dark and bright bladders, if both are expected;
- whole fish held out for testing.

## How to run it

The script is `scripts/finetune_sam_med3d.py`. It needs SAM-Med3D installed as
in the README, and a torch build for the GPU. Each fish needs an entry in
`configs/experiments/sam_med3d_fish_plan.json`:

- for training: its scan, format and reference mask;
- for evaluation: also its box, its clicks and its HD95 limit.

```bash
# --device: xpu or cuda, depending on the GPU of the machine it runs on (cpu works, but slowly)
# 1. encode the training cubes, once per fish (synthetic: only for --synthetic training)
python scripts/finetune_sam_med3d.py encode --fish oreo gadus tuna --kind real --device xpu
python scripts/finetune_sam_med3d.py encode --fish gadus tuna --kind synthetic --device xpu
# 2. train a named set, here on Oreo and Gadus with synthetic bladders
python scripts/finetune_sam_med3d.py train --name lofo-test-tuna-synthetic --fish oreo gadus --synthetic --device xpu
# 3. evaluate it on the left-out fish (--human adds the human control; --weights standard for the baseline)
python scripts/finetune_sam_med3d.py evaluate --weights lofo-test-tuna-synthetic --fish tuna --human
```

**Where things go:**

- each set: `sam-med3d-assets/finetuned/<name>/`, holding `decoder.pth` and a
  `manifest.json` with the training fish, the recipe, the loss and the
  checksums;
- the training log: `results/sam_med3d_finetune/training/<name>.log`;
- the scores: `results/sam_med3d_finetune/<weights>/summary.json`.

`--input NAME=PATH` replaces a fish's scan. Until the DICOM loader keeps the
slices in order, the tuna needs a NIfTI copy in slice order.

**The sets from this experiment:**

- `s1-<fish>`: trained on that fish;
- `lofo-test-<fish>`: trained on the other two;
- `lofo-test-<fish>-synthetic`: the same, with synthetic bladders.

## Using a tuned set

`sam_med3d` takes `weights: standard` (the default) or a set's name.

- **CLI:** in the experiment config:

  ```yaml
  algorithm:
    name: "sam_med3d"
    parameters:
      weights: "lofo-test-oreo"
  ```

- **Web app:** on `/segment`, choose SAM-Med3D and type the name in the
  "Weights" field. The result then carries a warning that it is not the
  published model.

The report records the weights' name and SHA-256. An unknown name, a set made
from other base weights, or a changed file gives an error.

Only test a set on a fish it was not trained on: `lofo-test-<fish>` on the fish
in its name.
