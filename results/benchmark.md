Synthetic warehouse, 200 random orders per size (seed 42).

| Items | Method | Mean route length | Shorter than terminal order | Gap to optimal | Time / order |
|---:|---|---:|---:|---:|---:|
| 5 | Terminal order | 138.7 | 0.0% | 31.8% | 0.02 ms |
| 5 | Random order | 140.7 | -1.5% | 33.7% | 0.14 ms |
| 5 | Nearest neighbour | 109.6 | 21.0% | 4.1% | 0.03 ms |
| 5 | NN + 2-opt | 105.5 | 23.9% | 0.3% | 0.08 ms |
| 5 | Q-learning | 105.2 | 24.1% | 0.0% | 1361.21 ms |
| 5 | Held-Karp (optimal) | 105.2 | 24.1% | 0.0% | 0.23 ms |
| 8 | Terminal order | 209.0 | 0.0% | 62.3% | 0.01 ms |
| 8 | Random order | 210.1 | -0.5% | 63.1% | 0.09 ms |
| 8 | Nearest neighbour | 137.7 | 34.1% | 6.9% | 0.05 ms |
| 8 | NN + 2-opt | 129.9 | 37.9% | 0.8% | 0.20 ms |
| 8 | Q-learning | 128.8 | 38.4% | 0.0% | 2401.15 ms |
| 8 | Held-Karp (optimal) | 128.8 | 38.4% | 0.0% | 3.23 ms |
| 12 | Terminal order | 301.0 | 0.0% | 90.0% | 0.02 ms |
| 12 | Random order | 301.6 | -0.2% | 90.4% | 0.10 ms |
| 12 | Nearest neighbour | 170.5 | 43.4% | 7.6% | 0.09 ms |
| 12 | NN + 2-opt | 160.7 | 46.6% | 1.5% | 0.47 ms |
| 12 | Q-learning | 248.7 | 17.4% | 57.0% | 4035.70 ms |
| 12 | Held-Karp (optimal) | 158.4 | 47.4% | 0.0% | 111.81 ms |
