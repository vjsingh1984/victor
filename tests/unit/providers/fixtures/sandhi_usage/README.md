# Vendored sandhi usage-parser fixture corpus

Copied from `anvai-labs/sandhi` `crates/sandhi-providers/tests/fixtures/` at commit
`3102dd8` (2026-07-20). `expected_usage.json` is the corpus ground truth
(`{tokens_in, tokens_out, cache_creation_tokens, cache_read_tokens}`) that sandhi's own
recorded-fixture replay + differential oracle (Sandhi TD-0001 W1/W2) verify its parsers
against. Victor's differential tests replay the same files through
`victor.providers.usage_parsing` to prove the victor-convention mapping.
Do not hand-edit; re-vendor from sandhi when the corpus changes.
