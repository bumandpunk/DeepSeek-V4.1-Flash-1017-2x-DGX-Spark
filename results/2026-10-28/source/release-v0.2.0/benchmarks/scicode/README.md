# Frozen SciCode development regression

This is a **50-substep, one-repeat development regression**, with 15 parent problems. It is not the Artificial Analysis leaderboard evaluation or a full SciCode test-set result. Use it to detect regressions, not to claim a coding capability improvement or a statistically established absence of degradation.

The protocol uses the public validation data at `SciCode1/SciCode` revision `4510f6a6aa27c43fad7b43da2c59602a86e88480`, background context, sequential generated solutions to previous steps, temperature 1, top-p 1, a deterministic per-step seed, high reasoning, and 16,384 output tokens. The included development JSONL and prompt template are unchanged. `provenance.json` records hashes, source URLs, and the limitation of the numeric-data mirror verification.

The measured script ran generation through an authenticated endpoint, and ran every generated solution inside an isolated Docker container. `run.py` adapts only paths, endpoint, key and executor arguments. Its prompt construction and code extraction were checked against all 50 saved baseline cases. `portability.json` identifies both script revisions. It needs Python 3.11 or newer on the client.

Build the executor and obtain the approximately 1 GB numeric file from the source in `provenance.json`. The file must match SHA256 `48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890`. The original official Drive file was not fully hash-verified independently of this mirror. The new executor Dockerfile is a portable recipe; the measured image ID is recorded with results.

```bash
docker build -t spark1017-scicode benchmarks/scicode
python benchmarks/scicode/run.py gold --executor-image spark1017-scicode --h5-file /absolute/test_data.h5 --output results/gold
python benchmarks/scicode/run.py generate --base http://127.0.0.1:19890 --key-file /absolute/key-file --output results/candidate
python benchmarks/scicode/run.py grade --executor-image spark1017-scicode --h5-file /absolute/test_data.h5 --output results/candidate
```

The Docker CLI must be available to the caller. Grading uses a non-root user, no network, a read-only filesystem, dropped capabilities, resource limits, and a 300-second timeout. Do not execute `solution.py` or `run.py` from a result directory directly on the host.

To regrade published responses without querying a model, first materialize a new directory with `replay.py --records <recorded-jsonl> --output <new-directory>`, then run the same isolated `run.py grade` command against that directory. The materializer writes data only; it does not import or execute any generated program. Compare the new `grade.json` against `recorded-grade.json` per case.

Keep all failed and truncated generations. The reference-code control passes **48/50** in the frozen environment; steps **78.3** and **70.8** have reference/harness issues. Publish the raw denominator of 50 and the control result together. Do not silently remove the two cases or change assertions to improve a score. Parent-level results are separate from substep pass rates. The development set is never used to train this project's inference extensions.

SciCode code and data retain Apache-2.0 licensing. `scicode/` and the prompt template derive from [scicode-bench/SciCode](https://github.com/scicode-bench/SciCode/tree/e3158ea011d4235245a547460d3688d7ccbf9900). `scicode_helpers.py` extracts the upstream HDF5 reading functions with standalone imports and a container-local default path; its implementation is preserved from the measured harness.
