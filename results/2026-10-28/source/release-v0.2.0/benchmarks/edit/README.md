# Complete-module edit fixtures

Three engineering fixtures use CPython v3.12.9 `bisect.py`, `graphlib.py`, and `textwrap.py`, downloaded from [python/cpython](https://github.com/python/cpython/tree/v3.12.9/Lib). The PSF license is preserved in `sources/LICENSE`. Original file SHA256 values are recorded in every run manifest.

Each prompt asks for a small new function/method while returning the complete updated source. These tasks deliberately measure source-preserving edits with repeated context; they are not a representative general coding leaderboard. Tests check the new function and selected existing behavior, not every CPython regression. Independent reference edits pass all 418 assertions (204 + 126 + 88).

```bash
python benchmarks/edit/run.py --label baseline --base http://127.0.0.1:19890 --key-file /absolute/key-file --executor-image python:3.12-slim --output results/edit-baseline
```

Pin the executor image digest for a published comparison. The measured executor was the existing pinned SciCode Python image; the harness needs only the Python standard library. Docker runs generated programs with no network, a read-only filesystem, a non-root user, dropped capabilities and CPU/memory/process limits. No generated code executes in the benchmark-client process.

Run baseline and candidate with the same prompts, seed, model bytes, context, slots and output budget. Compare request SHA256, full text SHA256, token count, executable assertions, decode rate, and total request duration. Decode streaming chunks can contain multiple tokens; complete-request wall time provides a complementary metric. Preserve failed and truncated generations.
