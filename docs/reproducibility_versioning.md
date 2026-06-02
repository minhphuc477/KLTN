# Reproducibility Versioning

The package version is intentionally conservative. Experimental runs must be
identified by the repository commit hash and resolved configuration snapshot,
not only by the static Python package version.

Before a benchmark or training run, record:

```powershell
git rev-parse HEAD
git status --short
```

Archive the generated `resolved_config.yaml` or `resolved_config.json`, the
checkpoint metadata sidecar, and the metrics JSON with the output artifacts.
Use a clean commit for thesis evidence runs so the recorded hash identifies the
executed code exactly.
