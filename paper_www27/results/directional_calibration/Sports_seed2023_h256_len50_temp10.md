# Directional calibration diagnostics

The counterfactual rows compare dot-product and cosine rankings within the same checkpoint. They therefore isolate scoring geometry, not independently trained-model differences.

## Item-norm / popularity coupling

| model | group | n | mean item norm | Spearman(norm, popularity) |
| --- | --- | ---: | ---: | ---: |
| SASRec | all | 18229 | 0.9269 | -0.8430 |
| SASRec | head | 6077 | 0.8102 | -0.8430 |
| SASRec | mid | 6076 | 0.9340 | -0.8430 |
| SASRec | tail | 6076 | 1.0367 | -0.8430 |
| CANDSSASRec | all | 18229 | 1.3503 | -0.1626 |
| CANDSSASRec | head | 6077 | 1.3314 | -0.1626 |
| CANDSSASRec | mid | 6076 | 1.3589 | -0.1626 |
| CANDSSASRec | tail | 6076 | 1.3606 | -0.1626 |

## Within-checkpoint dot-versus-cosine counterfactual

| model | target group | n | dot R@10 | cosine R@10 | Δ R@10 | dot NDCG@10 | cosine NDCG@10 | angular-win / radial-displacement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SASRec | all | 35598 | 0.0427 | 0.0430 | +0.0003 | 0.0199 | 0.0201 | 0.0036 / 0.0027 |
| SASRec | head | 19825 | 0.0646 | 0.0716 | +0.0070 | 0.0300 | 0.0336 | 0.0060 / 0.0047 |
| SASRec | mid | 6875 | 0.0198 | 0.0119 | -0.0079 | 0.0099 | 0.0056 | 0.0010 / 0.0004 |
| SASRec | tail | 8484 | 0.0121 | 0.0034 | -0.0087 | 0.0054 | 0.0015 | 0.0001 / 0.0000 |
| CANDSSASRec | all | 35598 | 0.0498 | 0.0545 | +0.0047 | 0.0236 | 0.0269 | 0.0092 / 0.0067 |
| CANDSSASRec | head | 19825 | 0.0767 | 0.0857 | +0.0090 | 0.0368 | 0.0427 | 0.0159 / 0.0114 |
| CANDSSASRec | mid | 6875 | 0.0253 | 0.0240 | -0.0013 | 0.0116 | 0.0114 | 0.0016 / 0.0015 |
| CANDSSASRec | tail | 8484 | 0.0094 | 0.0091 | -0.0004 | 0.0037 | 0.0038 | 0.0001 / 0.0001 |

## Interpretation boundary

- A positive cosine-minus-dot difference in the SASRec counterfactual shows that removing radial factors changes rankings favorably for that fixed representation; it is not, by itself, evidence that retraining with cosine will improve every dataset.
- `radial_displacement_rate` counts examples where the target is directionally closer than SASRec's strongest dot-product competitor but loses to that competitor under the dot product. It operationalizes the proposed rank-reversal mechanism.
- The trained CaNDS rows remain a complementary endpoint comparison. Causal performance claims require matched seeds and datasets, which the accompanying training grid supplies.
