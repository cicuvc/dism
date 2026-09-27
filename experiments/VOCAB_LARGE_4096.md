# Larger vocabulary-use evaluation while training continues

User selected baseline and width384_72m only,2026-09-11.4096 packed sequences
of2048 tokens each,8,388,608 tokens per head/side/model,16x prior256-sequence
evaluation. Same held-out prefix and tokenizer,repeat=False; compare final
packed-input SHA256. Pure-hard inference from each completed step3000 checkpoint.
No CE evaluation here; measures actual CUDA embedding hard labels.

Service dism-vocab-large-4096-20260911; experiments/run_vocab_large.py serializes
width384_72m then baseline. Batch2,0.2s sleep after each synchronized batch,
2 CPU threads; current GDN training is neither stopped nor reconfigured.
Concurrent timing is not a standalone throughput benchmark. Initial progress:
384 completed128 sequences in30.79s; concurrent trainer step1410,0.766s/update.

Outputs under activation-study-3k-20260910/vocab-large-4096-20260911:
- status.json and per-model .log files;
- counts.pt / counts.csv: per layer/head/side frequency, unused/rare counts,
  entropy-effective vocabulary, top1/top8 mass,90%-mass coverage, Gini;
- growth.json: cumulative statistics every256 sequences for sample-size convergence;
- sequence_presence.pt and CSV columns: number/fraction of packed sequences
  containing each codeword, not independent-document occurrence;
- causal_coverage_by_position.pt: per layer/head/position count of queries whose
  label has at least one equal K label at j<=i within that packed sequence;
  report.json includes mean coverage and coverage excluding first64 positions;
- marginal Q/K overlap,JS divergence and Q mass on globally unseen/rare K labels;
- label_load.png and ranked_load.png, with correct head counts for both widths.

Causal coverage is label-match opportunity only, not attention mass, useful
retrieval or successful prediction. Marginal concentration is not proof of
collapse; rare/unseen are relative to this finite sample. Tests verify causal
coverage against brute-force prefixes. First-batch traced vs untraced model
outputs must match bitwise, and total label counts must match sampled tokens.
No full attention matrix or full label corpus saved. Initial status evaluation,
not completed; no automatic retries on failure.
