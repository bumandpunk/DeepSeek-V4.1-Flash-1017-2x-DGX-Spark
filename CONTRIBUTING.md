# Contributing

Start with [the experiment rules](docs/contributing.md). Keep each performance
change independently switchable and disabled until its paired validation is available.

On Linux with Python 3.11+ and a C++17 compiler:

```bash
bash scripts/ci.sh
python3 results/release-v0.2.0-alpha.1/check.py
```

This runs CPU invariants, lifecycle failure tests, recorded-result consistency,
source checks and an ASan/UBSan native cache test. It does not download weights,
run inference, access SSH hosts or execute recorded model-generated programs.
The Linux AIO integration test is separate and requires PyTorch and a suitable
local filesystem. GPU and RDMA tests require an idle, explicitly assigned pair.
The release checker ties the current runtime/build inputs to their measured image.
For a subsequent release, archive this release's source with its evidence before
replacing those inputs and recording new validation. See [publishing](docs/publishing.md).

For a performance pull request, include the baseline and candidate source IDs,
hardware, identical weight identity, request hashes, cache state, complete response
hashes, correctness, end-to-end timings and every failed measurement. Include a
small correctness/failure test for changed behavior. Do not use evaluation tasks to
calibrate the controller or present changed prompts as a paired comparison.

Use `config/cluster.local.json` for local configuration. Never commit that file,
API keys, model weights, rank caches, private snapshots or unredacted host logs.

README structure is informed by the public MiaAI-Lab and Elevatormusic two-Spark
recipes; text and implementation here are specific to this project. Attribute
borrowed code and retain its license/notice. A contribution must explain which
part is original, adapted, or unchanged upstream work.
