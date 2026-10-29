# Preserved memory-guard failure

The pre-fix candidate completed 53 requests and lost the 54th during the third 32K prefill case. Rank 1 crossed the 2.5 GiB host guard at 2.4798 GiB. No programming grading had started. The recovery procedure restored and checked the original service.

The controller had allowed the engine to retain memory down to a 2 GiB protected host floor. The final controller instead derives that budget from the configured external guard plus a 0.5 GiB margin. The guard itself is unchanged. The archived controller hash matches the failed image build record.

Raw requests and the partial response remain in `speed/requests.jsonl`; logs and `summary.json` preserve the failure. They are separate from the subsequent complete paired runs, not discarded or counted as successful generations.
