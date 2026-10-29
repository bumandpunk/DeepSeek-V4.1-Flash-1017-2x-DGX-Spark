# v0.2.0-alpha.1 validation status

The source preview is public. The final clean-image programming regression and
end-of-run restoration are still running. No completed release-quality result
is claimed yet. The current snapshot is in [validation-status.json](validation-status.json).

Completed evidence:

- [Final image build](build.json), [installed sources and dependencies](installed-image.json), and [both-node CUDA checks](engram-cuda.json).
- [Clean Linux CPU CI](ci-environment.json).
- [All-profile tiled diagnostic](tiled-workspace-probe/README.md): 58 requests per profile, matched outputs, 836 edit assertions and verified restoration. This used a layered diagnostic image and reused rank caches, so it is not the final empty-cache run.
- [Earlier complete quality reference](pre-workspace-fix-quality/README.md), with its distinct image and context capacity disclosed.

The running full pass starts with empty native/rank caches and tests the final
image on the same two GB10s. Baseline/edit quality, startup-failure recovery and
the frozen recovery service will be added when completed. Failed attempts are
retained in their separate folders. Historical Mia comparisons remain separate.

GitHub CPU checks can pass while this hardware run is pending; the workflow
reports that distinction. The release-asset workflow and local source packager
still require complete release evidence.
