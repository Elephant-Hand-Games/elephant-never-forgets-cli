# Provider Notes

The default provider is native fastembed with the quantized Nomic profile.

Ollama is useful for local HTTP embedding tests:

```sh
enf search "provider override warning" --provider ollama --model nomic-embed-text --json
```

OpenAI and OpenAI-compatible providers should read API keys from environment
variables and must not print secrets in errors or diagnostics.

If a provider override points at a profile that has not been indexed yet, search
should warn that the active profile has no indexed embeddings and suggest an index
command that installs or populates vectors.

