# Publishing the developer preview

The prepared version is Git tag `v0.2.0-alpha.1`, Python version `0.2.0a1`.
Use a **prerelease**, with the developer-preview scope in the release notes.

The repository source can be public while the final hardware regression runs.
The commands below intentionally require completed release evidence before
creating the versioned archive and tag; pending checks are listed in the
[validation status](../results/release-v0.2.0-alpha.1/README.md).

From the final source checkout:

```bash
bash scripts/ci.sh
python3 results/release-v0.2.0-alpha.1/check.py
python3 scripts/package_release.py
```

The archive contains public source, licenses and reproducibility evidence. It
does not contain model weights, keys, original-deployment snapshots, Docker
images or rank caches. `SHA256SUMS` identifies the ZIP. The ZIP also contains a
per-file hash manifest. The image build remains a separate operation on a Spark.

After selecting and configuring the GitHub repository, push the prepared main
branch and version tag. Check the repository's actual **CPU and release checks**
run before creating a release. Local Linux validation is recorded, but it does
not substitute for a hosted GitHub Actions run.

```bash
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
git push origin v0.2.0-alpha.1
gh release create v0.2.0-alpha.1 \
  dist/Spark1017-v0.2.0-alpha.1.zip dist/SHA256SUMS \
  --title 'Spark1017 v0.2.0-alpha.1' \
  --notes-file docs/releases/v0.2.0-alpha.1.md \
  --prerelease --draft
```

The manual **Release candidate assets** workflow performs the same checks and
uploads source artifacts to the workflow run. It has read-only repository
permissions and does not create a GitHub release. Review the draft's scope and
attachments before publishing it.

Do not change runtime/build inputs under an already validated tag. For a later
release, preserve this evidence with its source snapshot, update the version,
and record the new deployment and quality results. Keep failed experiments and
the documented limitations; a developer preview is not a production guarantee.
