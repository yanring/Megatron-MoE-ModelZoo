## Performance Results

Experiment Setup:
- 256 H100 GPUS
- NVLink 4th Generation
- InfiniBand 8x50 GB/s


| Model | Dispatcher | System | Precision | #GPUs | SEQ LEN | TP | CP | EP | PP | EDP | ETP | VPP | DP | FSDP | MBS | GBS | GA | recompute | Step time (sec) | Per GPU TF | MFU | Mem | Notes | Code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cross-node EP strategy | | | | | | | | | | | | | | | | | | | | | | | | |
| 235B | DeepEP | H100 | BF16 | 256 | 4096 | 2 | 1 | 32 | 8 | 1 | 1 | 4 | 16 | / | 1 | 2048 | 128 | norm, act | 19.7 | 245 |  |  |  | moe_dev(f3e6c5) |
| 235B | DeepEP | H100 | BF16 | 256 | 4096 | 2 | 1 | 32 | 8 | 1 | 1 | 4 | 16 | / | 1 | 2048 | 128 | norm, act | 17.6 | 276 |  |  | +1f1b overlap | moe_dev(f3e6c5) |
| Intra-node EP strategy |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 235B | DeepEP | H100 | BF16 | 256 | 4096 | 2 | 1 | 8 | 1 | 32 | 1 | 1 | 128 | On | 4 | 2048 | 4 | full | 17.6 | 276 |  |  | disable log para norm | main(1e057005) |
| 235B | A2A | H100 | BF16 | 256 | 4096 | 2 | 1 | 8 | 32 | 1 | 1 | 3 | 4 | / | 1 | 2048 | 512 | norm, act | 24.2 | 200 |  |  |  | main(1e057005) |
| 235B | DeepEP | H100 | BF16 | 256 | 4096 | 4 | 1 | 8 | 8 | 4 | 1 | 4 | 8 | / | 1 | 2048 | 256 |  | 29.0 | 167 |  |  |  | main(1e057005) |
