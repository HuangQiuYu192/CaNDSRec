# Directional calibration diagnostics

The counterfactual rows compare dot-product and cosine rankings within the same checkpoint. They therefore isolate scoring geometry, not independently trained-model differences.

## Item-norm / popularity coupling

| model | group | n | mean item norm | Spearman(norm, popularity) |
| --- | --- | ---: | ---: | ---: |
| SASRec | all | 11846 | 1.1248 | -0.7037 |
| SASRec | head | 3949 | 1.0028 | -0.7037 |
| SASRec | mid | 3949 | 1.1460 | -0.7037 |
| SASRec | tail | 3948 | 1.2256 | -0.7037 |
| CANDSSASRec | all | 11846 | 1.1022 | -0.1725 |
| CANDSSASRec | head | 3949 | 1.0868 | -0.1725 |
| CANDSSASRec | mid | 3949 | 1.1065 | -0.1725 |
| CANDSSASRec | tail | 3948 | 1.1133 | -0.1725 |

## Within-checkpoint dot-versus-cosine counterfactual

| model | target group | n | dot R@10 | cosine R@10 | Δ R@10 | dot NDCG@10 | cosine NDCG@10 | angular-win / radial-displacement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SASRec | all | 19412 | 0.0723 | 0.0703 | -0.0020 | 0.0408 | 0.0397 | 0.0220 / 0.0095 |
| SASRec | head | 9927 | 0.0908 | 0.1005 | +0.0098 | 0.0514 | 0.0576 | 0.0341 / 0.0163 |
| SASRec | mid | 3913 | 0.0703 | 0.0555 | -0.0148 | 0.0372 | 0.0286 | 0.0125 / 0.0043 |
| SASRec | tail | 5227 | 0.0436 | 0.0287 | -0.0149 | 0.0259 | 0.0165 | 0.0075 / 0.0010 |
| CANDSSASRec | all | 19412 | 0.0917 | 0.0977 | +0.0060 | 0.0441 | 0.0495 | 0.0195 / 0.0145 |
| CANDSSASRec | head | 9927 | 0.1169 | 0.1270 | +0.0102 | 0.0578 | 0.0668 | 0.0302 / 0.0220 |
| CANDSSASRec | mid | 3913 | 0.0894 | 0.0915 | +0.0020 | 0.0424 | 0.0450 | 0.0164 / 0.0130 |
| CANDSSASRec | tail | 5227 | 0.0518 | 0.0532 | +0.0013 | 0.0222 | 0.0235 | 0.0029 / 0.0025 |

## Interpretation boundary

- A positive cosine-minus-dot difference in the SASRec counterfactual shows that removing radial factors changes rankings favorably for that fixed representation; it is not, by itself, evidence that retraining with cosine will improve every dataset.
- `radial_displacement_rate` counts examples where the target is directionally closer than SASRec's strongest dot-product competitor but loses to that competitor under the dot product. It operationalizes the proposed rank-reversal mechanism.
- The trained CaNDS rows remain a complementary endpoint comparison. Causal performance claims require matched seeds and datasets, which the accompanying training grid supplies.
