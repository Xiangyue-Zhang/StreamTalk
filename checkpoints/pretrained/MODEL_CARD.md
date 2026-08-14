---
license: mit
library_name: pytorch
tags:
  - co-speech-gesture-generation
  - motion-generation
  - smpl-x
  - streamtalk
datasets:
  - H-Liu1997/BEAT2
---

# StreamTalk retrained CFG checkpoints

These are hardware-agnostic FP32 generator state dictionaries for
`DiffusionDITNetPartsFixedExpressions2PostNormInteraction2`. They were retrained
on BEAT2 English with classifier-free condition dropout and selected with an
accelerated H200 implementation of the StreamTalk generation procedure plus
the official EMAGE/PantoMatrix AESK FGD metric.

| File | Intended role | Epoch | CFG | Speaker2 FGD | All FGD |
| --- | --- | ---: | ---: | ---: | ---: |
| `streamtalk_speaker2_combined_e0946_cfg3.pt` | Speaker2 and combined | 946 | 3 | 0.378879 | 0.250287 |
| `streamtalk_speaker_all_e0940_cfg3.pt` | All speakers | 940 | 3 | 0.424477 | 0.217672 |

The combined selection score is
`max(Speaker2 FGD / 0.383, All FGD / 0.293)`.

The checkpoints contain 407 CPU FP32 tensors and 71,167,501 parameters. Both
load strictly into the originally released StreamTalk model schema. H200,
BF16, CUDA Graphs, and four-GPU execution were training/evaluation
infrastructure choices and are not encoded in the checkpoint files.

The reported values are accelerated EMAGE/PantoMatrix AESK FGD measurements,
not paper-exact oracle measurements. A 1e-3 absolute FGD reporting tolerance
was selected for engineering comparisons, but the final CFG=3 values have not
been accepted against a direct B=1/M=1/full-window oracle. BC and DIV have not
been re-evaluated for these retrained checkpoints.

The published inference NPZ bytes were independently re-scored with the
released scorer (metric batch 16): Speaker 2 `0.3788789702354345`; all speakers
`0.21764184426140076`. These differ from the selection-time values by about
`7e-15` and `2.97e-5`, respectively.

Verify downloads before loading:

```bash
python tools/verify_pretrained.py --weights-dir checkpoints/pretrained
```

PyTorch checkpoints use pickle internally. Only load files downloaded from the
linked StreamTalk release and matching the published SHA256 values.

End-to-end inference also requires WavLM Large, the bundled
`SimpleSpeechModel`, SMPL-X neutral, and the BEAT2 retrieval database. Those
runtime assets are independent of the generator checkpoint format.
