# Earlier A0 quality reference

This complete quality suite predates the allocator-budget refresh and Engram
sum/cast change. It used image and installed-source hashes recorded here, the
same A0/Engram input metadata and read-only input mounts, and the same public
quality-harness bytes. The early lifecycle run encountered an environment-order
verification issue after measurement; that did not invalidate the saved requests,
responses or isolated grades.

The earlier shared window was 512K and the final release window is 256K. This
reference is used only to compare identical quality requests, complete messages,
extracted code and verdicts; no performance comparison across those capacities
is claimed. All failed and budget-exhausted samples remain in the denominator.
The release checker compares these records with the final baseline as well as
comparing the final baseline with the edit profile.

`model-before.py` is the exact installed pre-change model source (hash checked
against the image manifest), derived from the pinned Apache-2.0 TensorFold
source and the published overlays. The repository LICENSES/ and NOTICE apply.
