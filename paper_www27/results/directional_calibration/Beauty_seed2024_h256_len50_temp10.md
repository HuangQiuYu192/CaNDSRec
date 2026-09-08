# Directional calibration diagnostics

The counterfactual rows compare dot-product and cosine rankings within the same checkpoint. They therefore isolate scoring geometry, not independently trained-model differences.

## Item-norm / popularity coupling

| model | group | n | mean item norm | Spearman(norm, popularity) |
| --- | --- | ---: | ---: | ---: |
| SASRec | all | 12042 | 0.9414 | -0.7820 |
| SASRec | head | 4014 | 0.8327 | -0.7820 |
| SASRec | mid | 4014 | 0.9506 | -0.7820 |
| SASRec | tail | 4014 | 1.0408 | -0.7820 |
| CANDSSASRec | all | 12042 | 0.8817 | -0.2909 |
| CANDSSASRec | head | 4014 | 0.8674 | -0.2909 |
| CANDSSASRec | mid | 4014 | 0.8841 | -0.2909 |
| CANDSSASRec | tail | 4014 | 0.8936 | -0.2909 |

## Within-checkpoint dot-versus-cosine counterfactual

| model | target group | n | dot R@10 | cosine R@10 | Δ R@10 | dot NDCG@10 | cosine NDCG@10 | angular-win / radial-displacement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SASRec | all | 22363 | 0.0770 | 0.0739 | -0.0030 | 0.0392 | 0.0369 | 0.0127 / 0.0080 |
| SASRec | head | 12707 | 0.1115 | 0.1146 | +0.0031 | 0.0565 | 0.0575 | 0.0209 / 0.0137 |
| SASRec | mid | 4183 | 0.0347 | 0.0232 | -0.0115 | 0.0182 | 0.0119 | 0.0026 / 0.0007 |
| SASRec | tail | 5277 | 0.0301 | 0.0190 | -0.0112 | 0.0155 | 0.0082 | 0.0011 / 0.0002 |
| CANDSSASRec | all | 22363 | 0.0826 | 0.0877 | +0.0051 | 0.0398 | 0.0450 | 0.0215 / 0.0158 |
| CANDSSASRec | head | 12707 | 0.1216 | 0.1306 | +0.0091 | 0.0596 | 0.0684 | 0.0360 / 0.0263 |
| CANDSSASRec | mid | 4183 | 0.0414 | 0.0392 | -0.0022 | 0.0188 | 0.0190 | 0.0038 / 0.0036 |
| CANDSSASRec | tail | 5277 | 0.0246 | 0.0262 | +0.0015 | 0.0104 | 0.0109 | 0.0013 / 0.0009 |

## Interpretation boundary

- A positive cosine-minus-dot difference in the SASRec counterfactual shows that removing radial factors changes rankings favorably for that fixed representation; it is not, by itself, evidence that retraining with cosine will improve every dataset.
- `radial_displacement_rate` counts examples where the target is directionally closer than SASRec's strongest dot-product competitor but loses to that competitor under the dot product. It operationalizes the proposed rank-reversal mechanism.
- The trained CaNDS rows remain a complementary endpoint comparison. Causal performance claims require matched seeds and datasets, which the accompanying training grid supplies.
