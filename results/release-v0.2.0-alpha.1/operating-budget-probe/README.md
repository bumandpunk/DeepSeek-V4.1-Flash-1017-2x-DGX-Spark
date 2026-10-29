# Operating-budget probe

Ten fixed long-input requests passed with a 3.0 GiB engine host reserve and the
unchanged 2.5 GiB external guard: three each at 8K, 32K and 128K, followed by 128K
plus 512 generated tokens. The original service was restored afterwards.

This diagnostic reused the prior image with the updated host controller and
existing native/rank caches. It is not clean-deployment evidence and does not
enter the final 58-request/profile comparison. The final candidate is rebuilt
from the published source and tested separately with new caches.

This preliminary probe used a 512K window with existing caches. A subsequent
fresh 512K start failed admission; see [the cold-start record](../failed-cold-512k/README.md).
The final release uses a 256K shared window.
