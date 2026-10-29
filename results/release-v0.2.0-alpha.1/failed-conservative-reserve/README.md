# Conservative reserve rejected at readiness

The candidate image in `build.json` used a 3.5 GiB engine host reserve above the
unchanged 2.5 GiB external guard. At initial readiness the head had approximately
4.14 GiB available and the request needed 0.75 GiB. Admission therefore returned
`NoRoom`: the reserve plus request did not fit. This was a failed deployment,
not a successful benchmark. The external guard did not trip. The original
service was restored, and the logs and exact controller are retained here.

The final controller uses a 3.0 GiB host reserve (external guard plus 0.5 GiB).
The separate operating-budget probe and the full release validation must pass
before that configuration is described as validated. Neither changing the
guard threshold nor deleting system caches was used to make this attempt pass.
