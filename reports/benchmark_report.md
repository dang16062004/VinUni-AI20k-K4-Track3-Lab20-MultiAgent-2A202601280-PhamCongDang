# Benchmark Report

| Run                                                     | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
| ------------------------------------------------------- | ----------: | ---------: | ------: | ------------: | -----------: | ----- |
| baseline · Research GraphRAG state-of-the-art and w    |        9.55 |     0.0004 |         |            0% |           0% |       |
| multi-agent · Research GraphRAG state-of-the-art and w |       23.12 |     0.0011 |         |           28% |           0% |       |
| baseline · Compare single-agent and multi-agent wor    |        6.17 |     0.0002 |         |            0% |           0% |       |
| multi-agent · Compare single-agent and multi-agent wor |       14.03 |     0.0006 |         |           50% |           0% |       |
| baseline · Summarize production guardrails for LLM     |        4.60 |     0.0001 |         |            0% |           0% |       |
| multi-agent · Summarize production guardrails for LLM  |       13.85 |     0.0006 |         |           29% |           0% |       |

## Summary

- **baseline** (3 runs): avg latency 6.77s, avg cost $0.00027, avg citation coverage 0%, 0 failed........
- **multi-agent** (3 runs): avg latency 17.00s, avg cost $0.00080, avg citation coverage 36%, 0 failed.
