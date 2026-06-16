# Benchmark Results (Spike sizes: 100, 1000, 50000)

| Model       | Optimizer     | Spike_Size | Final_Loss | Accuracy(%) |
|-------------|---------------|-----------:|-----------:|-----------:|
| MLP         | SGD           |      100.0 | 2.284101   | 12.5      |
| MLP         | Adam          |      100.0 | 1.032236   | 90.625    |
| MLP         | AdamW         |      100.0 | 1.032796   | 90.625    |
| MLP         | Adaptive GMR  |      100.0 | 0.939321   | 90.625    |
| MLP         | SGD           |     1000.0 | 2.284101   | 12.5      |
| MLP         | Adam          |     1000.0 | 1.032239   | 90.625    |
| MLP         | AdamW         |     1000.0 | 1.032799   | 90.625    |
| MLP         | Adaptive GMR  |     1000.0 | 0.939319   | 90.625    |
| MLP         | SGD           |    50000.0 | 2.284101   | 12.5      |
| MLP         | Adam          |    50000.0 | 1.032239   | 90.625    |
| MLP         | AdamW         |    50000.0 | 1.032799   | 90.625    |
| MLP         | Adaptive GMR  |    50000.0 | 0.939319   | 90.625    |
| CNN         | SGD           |      100.0 | 2.319270   | 6.25      |
| CNN         | Adam          |      100.0 | 1.288799   | 96.875    |
| CNN         | AdamW         |      100.0 | 1.289086   | 96.875    |
| CNN         | Adaptive GMR  |      100.0 | 1.344023   | 90.625    |
| CNN         | SGD           |     1000.0 | 2.319271   | 6.25      |
| CNN         | Adam          |     1000.0 | 1.288803   | 96.875    |
| CNN         | AdamW         |     1000.0 | 1.289091   | 96.875    |
| CNN         | Adaptive GMR  |     1000.0 | 1.344024   | 90.625    |
| CNN         | SGD           |    50000.0 | 2.319271   | 6.25      |
| CNN         | Adam          |    50000.0 | 1.288804   | 96.875    |
| CNN         | AdamW         |    50000.0 | 1.289091   | 96.875    |
| CNN         | Adaptive GMR  |    50000.0 | 1.344024   | 90.625    |
| Transformer | SGD           |      100.0 | 2.282357   | 9.375     |
| Transformer | Adam          |      100.0 | 0.711300   | 100.0     |
| Transformer | AdamW         |      100.0 | 0.712188   | 100.0     |
| Transformer | Adaptive GMR  |      100.0 | 0.732469   | 96.875    |
| Transformer | SGD           |     1000.0 | 2.282357   | 9.375     |
| Transformer | Adam          |     1000.0 | 0.711296   | 100.0     |
| Transformer | AdamW         |     1000.0 | 0.712189   | 100.0     |
| Transformer | Adaptive GMR  |     1000.0 | 0.732468   | 96.875    |
| Transformer | SGD           |    50000.0 | 2.282357   | 9.375     |
| Transformer | Adam          |    50000.0 | 0.711303   | 100.0     |
| Transformer | AdamW         |    50000.0 | 0.712190   | 100.0     |
| Transformer | Adaptive GMR  |    50000.0 | 0.732468   | 96.875    |
