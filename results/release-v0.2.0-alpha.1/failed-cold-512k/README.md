# Cold 512K startup rejected at request admission

This candidate used fresh native and rank caches, a 524288-token shared window,
a 3.0 GiB engine host reserve, and an unchanged 2.5 GiB external guard. The model
loaded but a minimal readiness request was rejected: the head had approximately
3.61 GiB available while the guard required 3.0 GiB plus 0.75 GiB workspace.
Ordinary CUDA allocator cleanup had already run. Graph-private reserved memory
could not safely be assumed reusable. No admission guard was bypassed and the
external guard did not trip. The public controller restored the original service
and its five API checks passed.

The final release default is a 262144-token shared window. All three release
profiles use it, and the long-input workload remains 131072 input tokens plus
512 output tokens. The earlier 512K Mia comparisons remain historical data;
they are not mixed with the new 256K release measurements.
