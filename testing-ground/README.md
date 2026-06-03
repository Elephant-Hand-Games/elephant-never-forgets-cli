# Elephant Never Forgets Test Repo

Small local fixture for trying `enf` by hand.

Suggested flow:

```sh
cd /Users/turnercore/Projects/CLI/elephant-never-forgets-cli/testing-ground
enf init --db --model-cache project
enf status
enf index .
enf search "where are agent rules documented?"
enf search "strict offline cache" --mode keyword --cached-query-only
enf retrieve "provider override warning" --json
enf doctor
enf ci --no-embed
```

To try native vectors, use:

```sh
enf models install
enf index . --install-models
enf search "cache hit behavior" --mode hybrid --level both --json
```

Generated `.enf/` state and `.enf.toml` are ignored here so repeated manual tests
do not dirty the main repo.
