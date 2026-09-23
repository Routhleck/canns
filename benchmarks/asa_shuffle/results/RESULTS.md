# ASA shuffle benchmark — grid_1

Compared with the original Python multiprocessing workflow, the new complete
ASA pipeline reduces peak process-tree memory (PSS) by **66–68%** and reduces
single-worker batch time by **16% for H1 and 27% for H2**.

| Case | Workers | Old time (s) | New time (s) | Old → new peak PSS (GiB) |
|---|---:|---:|---:|---:|
| H1, 1,200 points | 1 | 68.38 | 57.29 | 9.86 → 3.38 |
| H1, 1,200 points | 2 | 39.88 | 45.14 | 18.51 → 6.10 |
| H2, 400 points | 1 | 83.51 | 60.97 | 10.26 → 3.38 |
| H2, 400 points | 2 | 47.04 | 45.93 | 18.95 → 6.02 |

Two-worker H1 takes **13% longer**; two-worker H2 is about **2% faster**.
All 36 test jobs passed, with identical complete persistence diagrams and cocycles.

Measured on one 16-vCPU Linux VM: median of three cold batches, four fixed
shifts per batch. The new configuration enables Rust sampling and finite PH;
both workflows already use Rust PH. With the same finite threshold on both
sides, the new single-worker workflow is about **1.19× faster** for H1 and H2.
