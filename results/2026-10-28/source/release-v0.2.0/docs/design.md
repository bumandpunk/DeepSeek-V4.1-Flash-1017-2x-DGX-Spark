# Design and invariants

## Convert freed unified memory into fewer reads

E320 removes 64 of 384 routed experts in each of 40 layers while preserving the retained EXL3 bytes and top-6 routing. Fewer stored experts do not imply fewer active experts per token. The project therefore spends a bounded portion of the released host/GPU shared memory on exact Engram reads.

Each tensor-parallel rank owns a contiguous set of Engram hash columns. Their prime-sized row ranges identify exactly which scale rows that rank can request. After model warm-up, the planner admits all of those scale ranges only if the measured host budget allows it. This is approximately 2.86 GiB per rank for the tested checkpoint. It retains at least 12 GiB of host headroom at initialization, caps new storage at 4 GiB, and recomputes the engine allocator ceiling after reservation. This is an admission calculation, not a guarantee against unrelated future processes allocating memory.

The scale bytes are immutable anonymous resident copies. Direct and buffered file descriptors are aliases only after device/inode validation. Reads use bounds-checked memcpy. The buffered prompt path skips scale read jobs when all rows are resident. Weight misses use the upstream O_DIRECT AIO path. A bounded four-way set-associative cache holds exact row bytes; duplicate misses within one batch share a physical read and fan out to destinations only after successful completion. Failed/short reads are never cached. Scale residency, hot cache and coalescing may be ablated separately.

All descriptors stay open, are never reused, and refer to read-only immutable files for the engine lifetime. Resident ranges are sealed before serving. All demand-cache mutation occurs under the existing AIO mutex. The hot path allocates no cache entries dynamically; duplicate bookkeeping remains per batch. No weight reconstruction, floating-point conversion, approximate lookup or speculative result bypass is introduced.

## Calibrate acceptance without diverging ranks

The draft confidence head estimates conditional prefix acceptance. For each depth/confidence bin, the controller stores integer counts of actually reached positions. Accepted prefix positions are successes; the first rejected position is a failure; the unreached suffix is censored. A prior shrinks low-sample cells toward the raw confidence. Integer decay bounds adaptation history.

The controller computes expected accepted-prefix curves, then solves the joint verification-row allocation exactly with dynamic programming. The objective is expected emitted target tokens divided by a frozen hardware cost. Complexity is O(streams × row_budget × max_depth), avoiding an exponential enumeration of stream-depth combinations. Every emitted token still passes the unchanged target verifier and sampler.

Both ranks see identical logits and target outcomes, update counts in identical stream order, and use the identical frozen cost file. No local wall-clock observation enters a serving decision. Offline calibration records complete round latency; these tables have draft_ms=0 because drafting is already included. Interpolated/extrapolated row sizes are labeled explicitly. A curve measured at one concurrency level may not generalize to another; the included curve is experimental.

## Quality and attribution boundaries

Exact cache semantics can be tested byte for byte. Changing verification batch shapes can still affect floating-point numerical paths; theoretical verifier preservation does not replace output and executable-code checks. Small development suites cannot establish universal quality preservation. E320 quality differences are a separate weight-transformation issue from engine acceleration.

TensorFold supplies the inference engine, kernels, RDMA, quantized loading and DSpark drafter. MiaAI supplies the separately benchmarked native-engine patches. The independent contribution here is the bounded partition-aware read design, its implementation/invariants, and acceptance-calibrated row-budget controller. No claim of first-ever cache, speculative decoding, or global state-of-the-art is made.

## Spend a verification round on more repeated code

The GPU trace motivates a longer verified window for source-preserving edits. `lookup.py` maintains an 8192-token ring and an incremental exact 8-token index with at most four positions per key. It requires a 16–32-token suffix match and proposes up to 15 subsequent tokens from earlier committed history. The continuation excludes the current query suffix. Lookup and state are bounded, expire with the ring, and belong to one stream; there is no cross-user content cache.

A deterministic arbiter estimates reached-prefix acceptance separately for three match-length bands. It compares expected target tokens per frozen cost against the observed DSpark accepted-token rate (using the same frozen cost model). Unreached suffix positions remain censored. Unproductive proposals cause a return to DSpark. No local timer enters the decision. In this version, lookup is active only with one live stream; concurrent work retains the upstream DSpark/joint schedule. Target verification, grammar masks, sampling and drafter absorption remain in place. The target window is capped by its existing 16-row limit.

Prompt lookup and n-gram speculation are established techniques, described in [vLLM's documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/) and [its speculative decoding overview](https://docs.vllm.ai/en/latest/features/speculative_decoding/index.html). This project does not claim their invention. The implemented integration adds bounded incremental state, accepted-prefix cost arbitration, and DSpark fallback for this two-rank EXL3 engine. The implementation must be judged on matched-output measurements, including workloads with little repetition.

## Related work and contribution scope

| Prior work | Relevant mechanism | Scope of this implementation |
|---|---|---|
| [Prompt Lookup Decoding, Saxena 2023](https://github.com/apoorvumang/prompt-lookup-decoding) | Propose continuations found in earlier context, then target-verify them. | Bounded incremental index and deterministic acceptance/cost arbitration integrated into the two-rank 1017 EXL3 engine. The lookup idea itself is prior art. |
| [Blazedit, Daita et al. 2025](https://github.com/ise-uiuc/blazedit) | Multi-layer speculation uses prompt lookup to accelerate a draft model before target verification. | Here lookup directly proposes into the target window or falls back to the existing DSpark path; no separate small draft model is trained or deployed. |
| [EfficientEdit, Wang et al. 2025](https://arxiv.org/abs/2506.02780v2) | Reuses source segments, locates edits, and uses edit-oriented drafts with dynamic verification. | This release does not implement a learned edit-location model or claim that combining reuse and neural drafting is new. Its measured contribution is a concrete constrained-hardware implementation. |
| [Cursor speculative edits, 2024](https://cursor.com/blog/instant-apply) | Uses code-edit structure to propose deterministic continuations for full-file rewriting. | Full-file edit acceleration has established precedent. No headline rate from another model/hardware is treated as a same-condition comparison. |

No code from these four projects was copied into the new lookup implementation. Their mechanisms are cited as related work. Original engineering implementation and useful measurements do not by themselves establish a new research algorithm or global novelty.
