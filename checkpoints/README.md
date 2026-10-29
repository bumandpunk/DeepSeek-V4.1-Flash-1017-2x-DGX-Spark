# Optional E320 checkpoint transform

E320 was selected in an earlier experiment using code-prefill routing-weight mass. It is an expert-pruning transform, not a new quantization or training run. The provided map retains 320/384 experts in each of 40 backbone layers and leaves top-6 activation unchanged. Retained packed EXL3 tensor bytes are copied exactly; router rows are sliced and expert IDs remapped. DSpark, shared experts, attention and Engram stay unchanged.

Use the exact compatible 1017 EXL3 2.9 bpw source checkpoint and a new destination with at least 180 GiB of free disk. Run the export only while inference is stopped: it streams and verifies the complete model.

```bash
python tools/export_checkpoint.py --source /absolute/original/checkpoint --destination /absolute/new/e320 --selection checkpoints/e320-selection.json
```

This exports tensors and metadata. The Engram files remain separate and must be supplied to the engine. The model family adapter reads the new expert count from config.json. Do not give this map to a different architecture or checkpoint without a fresh router calibration and quality evaluation.

The first development test scored 37/50 SciCode substeps for both A0 and E320, with one gained and one lost case; the gold reference passed only 48/50 in the pinned executor. These results do not prove lossless model quality or an official Artificial Analysis score. Pruning preserves retained bytes, not the original model function.
