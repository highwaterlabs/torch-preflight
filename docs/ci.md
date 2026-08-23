# CI integration

## GitHub Action

```yaml
- uses: highwaterlabs/torch-preflight@v0
  with:
    paths: src/
    format: github        # inline PR annotations
```

`fail-on` defaults to `error`. For a repository whose product is training runs, use
`fail-on: warning` — it adds the retained-graph, device-sync and unseeded-run findings to the
gate, while `DataLoader` tuning notes stay out of it. [What the levels
mean](rules.md#what-the-levels-mean).

Or with code scanning:

```yaml
- uses: highwaterlabs/torch-preflight@v0
  with:
    paths: src/
    format: sarif
    output: torch-preflight.sarif
- uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: torch-preflight.sarif
```

## Pre-commit

```yaml
repos:
  - repo: https://github.com/highwaterlabs/torch-preflight
    rev: v0.5.0
    hooks:
      - id: torch-preflight
```

