# Directional calibration diagnostics

The counterfactual rows compare dot-product and cosine rankings within the same checkpoint. They therefore isolate scoring geometry, not independently trained-model differences.

## Item-norm / popularity coupling

| model | group | n | mean item norm | Spearman(norm, popularity) |
| --- | --- | ---: | ---: | ---: |
| SASRec | all | 18229 | 0.8898 | -0.8425 |
| SASRec | head | 6077 | 0.7789 | -0.8425 |
| SASRec | mid | 6076 | 0.8946 | -0.8425 |
| SASRec | tail | 6076 | 0.9959 | -0.8425 |
| CANDSSASRec | all | 18229 | 1.3188 | -0.1590 |
| CANDSSASRec | head | 6077 | 1.3010 | -0.1590 |
| CANDSSASRec | mid | 6076 | 1.3268 | -0.1590 |
| CANDSSASRec | tail | 6076 | 1.3286 | -0.1590 |

## Within-checkpoint dot-versus-cosine counterfactual

| model | target group | n | dot R@10 | cosine R@10 | Δ R@10 | dot NDCG@10 | cosine NDCG@10 | angular-win / radial-displacement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SASRec | all | 35598 | 0.0438 | 0.0424 | -0.0014 | 0.0201 | 0.0197 | 0.0032 / 0.0026 |
| SASRec | head | 19825 | 0.0678 | 0.0714 | +0.0036 | 0.0310 | 0.0332 | 0.0056 / 0.0045 |
| SASRec | mid | 6875 | 0.0166 | 0.0096 | -0.0070 | 0.0082 | 0.0044 | 0.0003 / 0.0001 |
| SASRec | tail | 8484 | 0.0117 | 0.0031 | -0.0086 | 0.0053 | 0.0014 | 0.0000 / 0.0000 |
| CANDSSASRec | all | 35598 | 0.0497 | 0.0551 | +0.0054 | 0.0233 | 0.0273 | 0.0099 / 0.0076 |
| CANDSSASRec | head | 19825 | 0.0771 | 0.0861 | +0.0090 | 0.0366 | 0.0431 | 0.0166 / 0.0127 |
| CANDSSASRec | mid | 6875 | 0.0237 | 0.0255 | +0.0017 | 0.0105 | 0.0121 | 0.0025 / 0.0025 |
| CANDSSASRec | tail | 8484 | 0.0091 | 0.0093 | +0.0002 | 0.0040 | 0.0041 | 0.0005 / 0.0005 |

## Interpretation boundary

- A positive cosine-minus-dot difference in the SASRec counterfactual shows that removing radial factors changes rankings favorably for that fixed representation; it is not, by itself, evidence that retraining with cosine will improve every dataset.
- `radial_displacement_rate` counts examples where the target is directionally closer than SASRec's strongest dot-product competitor but loses to that competitor under the dot product. It operationalizes the proposed rank-reversal mechanism.
- The trained CaNDS rows remain a complementary endpoint comparison. Causal performance claims require matched seeds and datasets, which the accompanying training grid supplies.
