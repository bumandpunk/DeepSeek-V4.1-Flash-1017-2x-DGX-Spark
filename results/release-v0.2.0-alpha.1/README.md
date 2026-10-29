# v0.2.0-alpha.1 validation

This is a source alpha with a known repeated-128K-input failure. The full
clean-image regression did not pass. Original services have been restored.

| Check | Result |
|---|---|
| No-cache application build and matching image on both GB10s | Passed |
| Native compilation from empty caches | Passed |
| Engram CUDA exactness/workspace checks | 9 cases passed on each GB10 |
| Linux CPU/lifecycle/native sanitizer CI | Passed |
| GitHub Python 3.11/3.12 source CI | Passed; GPU qualification reported separately |
| Existing-rank-cache diagnostic, three profiles | 58 complete matched requests per profile |
| Diagnostic full-file edits | 6 generations, 836/836 assertions |
| Final image with new rank caches | 55 requests completed; second 128K prefill failed |
| Full SciCode/CanItEdit on that cold run | Not reached |
| Original deployment restoration | 50 file hashes, 4 containers, API 5/5, Web 200 |

The successful three-profile diagnostic and the cold image have identical
installed runtime source hashes and Python versions. Their cache states differ.
This does not resolve the failed cold-deployment test.

- [Build](build.json) · [Installed source hashes](installed-image.json) · [CUDA checks](engram-cuda.json)
- [Three-profile diagnostic](tiled-workspace-probe/README.md) · [Runtime equivalence](tiled-workspace-probe/runtime-equivalence.json)
- [Cold-run failure](failed-cold-tiled/README.md) · [Known issue](../../docs/known-issues.md)
- [Original-service restoration](restoration.json)
- [Earlier quality reference](pre-workspace-fix-quality/README.md) · [Initial hosted CI](github-source-preview-ci.json)

Verify these records without GPU inference or executing generated code:

```bash
python3 results/release-v0.2.0-alpha.1/check_alpha.py
```

`check.py` and `scripts/package_release.py` require a completed clean-deployment
quality certificate. That certificate is not supplied for this failed run.
The GitHub prerelease provides the source snapshot and its checksums.
