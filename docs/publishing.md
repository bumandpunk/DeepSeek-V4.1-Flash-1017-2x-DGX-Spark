# Source alpha and validated bundles

`v0.2.0-alpha.1` is a source prerelease with a known cold-deployment 128K-input
failure. Its source archive contains the recorded diagnostics and failed run.

Check the source-alpha records with:

```bash
bash scripts/ci.sh
python3 results/release-v0.2.0-alpha.1/check_alpha.py
```

The source ZIP can be reproduced from the immutable tag:

```bash
git archive --format=zip --prefix=Spark1017-v0.2.0-alpha.1/ \
  -o Spark1017-v0.2.0-alpha.1-source.zip v0.2.0-alpha.1
```

The stricter `results/release-v0.2.0-alpha.1/check.py`,
`scripts/package_release.py`, and manual **Release candidate assets** workflow
require a completed clean-deployment quality certificate. They intentionally
do not accept this incomplete cold run. Source publication does not mark that
hardware qualification as passed.

For a subsequent version, retain these records, fix the known issue, and repeat
the affected deployment and quality checks. Keep the previous tag immutable.
