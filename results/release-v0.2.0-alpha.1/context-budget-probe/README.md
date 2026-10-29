# 256K operating-budget probe

The full 58-request speed suite passed with a 262144-token shared window,
3.0 GiB engine host reserve and unchanged 2.5 GiB external guard. This includes
ordinary generation, two/four HTTP streams over two engine slots, 8K/32K/128K
inputs and 128K input plus 512 generated tokens. The original service was
restored afterwards.

This diagnostic reused the previous image and native/rank caches, with the new
context setting. It is not the final clean deployment and is excluded from
all final comparison counts. The final candidate is rebuilt and tested with a
new native/rank-cache namespace on both nodes.
