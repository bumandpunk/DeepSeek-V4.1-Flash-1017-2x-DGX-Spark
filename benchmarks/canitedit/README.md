# Frozen CanItEdit subset

This additional check uses 12 of the 105 public [CanItEdit](https://huggingface.co/datasets/nuprl/CanItEdit) cases. IDs were selected by sorting SHA256(`spark1017-eval-20261027:` + decimal ID), taking the first 12, **before running inference**. No case is replaced after seeing results. `manifest.json` pins the dataset revision, original Parquet hash and selected IDs; `selected.json` retains source, instructions, reference answers and unchanged public tests under the included MIT license.

The sample contains five corrective, three perfective and four adaptive edits. It is a small supplementary regression, not a full CanItEdit result. It uses descriptive instructions, the upstream zero-shot chat prompt, greedy decoding, thinking off, seed `1017 + case ID`, 8,192 output tokens, and two repetitions. It does not reproduce the paper's sampling protocol. Test cases and reference answers are never sent to the inference endpoint.

Run the gold answers and unmodified input programs first. Keep both controls, every failed generation and any timeout. The executor is a separate CPU-only Python image with NumPy, Pandas, PyTorch and autograd; its exact image and dependency versions are recorded with results. It differs from CanItEdit's original evaluator image. Results are comparable between these paired arms, not directly to the paper's scores.

```bash
docker build --platform linux/amd64 -t spark1017-canitedit benchmarks/canitedit
python benchmarks/canitedit/run.py --mode gold --label gold --executor-image spark1017-canitedit --output results/canitedit-gold
python benchmarks/canitedit/run.py --mode before --label before --executor-image spark1017-canitedit --output results/canitedit-before
python benchmarks/canitedit/run.py --label baseline --base http://127.0.0.1:19890 --key-file /absolute/key-file --executor-image spark1017-canitedit --output results/canitedit-baseline
```

Use the same request, weights, prefill chunk and executor for the candidate. Compare complete text/solution hashes, token counts, correctness, decode rate and full request duration. Do not infer speed gains from a failed, truncated or shorter answer alone.

Generated code and the unchanged tests run in Docker with no network, a read-only filesystem, a non-root user, dropped capabilities, CPU/memory/process limits and a 60-second timeout. Random generators are seeded to 1017 inside the container. Success requires reaching a final sentinel after the tests. Client output files are limited to 1 MiB each and the container's log driver is disabled. The benchmark client never imports generated code. `--sudo-docker` is available for Linux hosts whose Docker CLI requires it.

Prompt source: [nuprl/CanItEdit, benchmark/generate_completions.py](https://github.com/nuprl/CanItEdit/blob/main/benchmark/generate_completions.py), Git blob `b3eec611edf8ac71e47aa9e8124bf63756ecf9eb`. The client adapts HTTP streaming, extraction of an optional Markdown code fence, bounded isolated execution and result logging. Citation: Cassano et al., *Can It Edit? Evaluating the Ability of Large Language Models to Follow Code Editing Instructions*, COLM 2024.

The portable Dockerfile pins the measured CPU dependency versions. It has not been rebuilt as a new image in this experiment; `executor-provenance.json` records the actual image and package inventory. Run both controls after rebuilding.
