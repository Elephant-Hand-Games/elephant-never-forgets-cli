# Cache Workflows

Query embeddings are keyed by normalized query and active profile. Repeating a
cached query should avoid loading the model again.

Strict offline cache mode:

```sh
enf search "cache hit behavior" --cached-query-only
```

If the query vector is absent, strict mode should fail with a clear remediation.
Keyword mode can still run with `--cached-query-only` because it does not need a
query vector.

