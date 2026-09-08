# Directional calibration diagnostics

The counterfactual rows compare dot-product and cosine rankings within the same checkpoint. They therefore isolate scoring geometry, not independently trained-model differences.

## Item-norm / popularity coupling

| model | group | n | mean item norm | Spearman(norm, popularity) |
| --- | --- | ---: | ---: | ---: |
| SASRec | all | 12042 | 0.9361 | -0.7788 |
| SASRec | head | 4014 | 0.8300 | -0.7788 |
| SASRec | mid | 4014 | 0.9456 | -0.7788 |
| SASRec | tail | 4014 | 1.0327 | -0.7788 |
| CANDSSASRec | all | 12042 | 0.9499 | -0.2647 |
| CANDSSASRec | head | 4014 | 0.9349 | -0.2647 |
| CANDSSASRec | mid | 4014 | 0.9528 | -0.2647 |
| CANDSSASRec | tail | 4014 | 0.9621 | -0.2647 |

## Within-checkpoint dot-versus-cosine counterfactual

| model | target group | n | dot R@10 | cosine R@10 | Δ R@10 | dot NDCG@10 | cosine NDCG@10 | angular-win / radial-displacement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SASRec | all | 22363 | 0.0757 | 0.0748 | -0.0009 | 0.0391 | 0.0371 | 0.0146 / 0.0090 |
| SASRec | head | 12707 | 0.1089 | 0.1165 | +0.0076 | 0.0567 | 0.0583 | 0.0247 / 0.0153 |
| SASRec | mid | 4183 | 0.0373 | 0.0232 | -0.0141 | 0.0180 | 0.0113 | 0.0022 / 0.0014 |
| SASRec | tail | 5277 | 0.0290 | 0.0182 | -0.0108 | 0.0147 | 0.0080 | 0.0006 / 0.0002 |
| CANDSSASRec | all | 22363 | 0.0859 | 0.0899 | +0.0041 | 0.0406 | 0.0460 | 0.0234 / 0.0178 |
| CANDSSASRec | head | 12707 | 0.1260 | 0.1319 | +0.0059 | 0.0607 | 0.0692 | 0.0393 / 0.0297 |
| CANDSSASRec | mid | 4183 | 0.0421 | 0.0440 | +0.0019 | 0.0188 | 0.0209 | 0.0043 / 0.0038 |
| CANDSSASRec | tail | 5277 | 0.0271 | 0.0286 | +0.0015 | 0.0111 | 0.0118 | 0.0011 / 0.0011 |

## Interpretation boundary

- A positive cosine-minus-dot difference in the SASRec counterfactual shows that removing radial factors changes rankings favorably for that fixed representation; it is not, by itself, evidence that retraining with cosine will improve every dataset.
- `radial_displacement_rate` counts examples where the target is directionally closer than SASRec's strongest dot-product competitor but loses to that competitor under the dot product. It operationalizes the proposed rank-reversal mechanism.
- The trained CaNDS rows remain a complementary endpoint comparison. Causal performance claims require matched seeds and datasets, which the accompanying training grid supplies.
