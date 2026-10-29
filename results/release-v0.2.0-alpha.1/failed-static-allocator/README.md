# Failed allocator-cap diagnostic

The sequence included five API health checks, three structured API checks,
ordinary generation and concurrency, then failed on its first 8192-token
prefill. Of 49 recorded requests, 48 completed and the failed request returned
no generated tokens. The server raised CUDA allocation failure while cloning
a 200 MiB FP32 Engram reduction accumulator. This was not an external memory
guard trip. The 256K context window, 3.0 GiB engine host reserve and 2.5 GiB
external guard were unchanged. These failed records are excluded from release
performance and quality counts. The original service was recovered and its
five API checks passed.

This was the clean image before dynamic allocator-budget refresh.
