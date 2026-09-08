# Directional calibration diagnostics

The counterfactual rows compare dot-product and cosine rankings within the same checkpoint. They therefore isolate scoring geometry, not independently trained-model differences.

## Item-norm / popularity coupling

| model | group | n | mean item norm | Spearman(norm, popularity) |
| --- | --- | ---: | ---: | ---: |
| SASRec | all | 11846 | 1.1644 | -0.7015 |
| SASRec | head | 3949 | 1.0371 | -0.7015 |
| SASRec | mid | 3949 | 1.1886 | -0.7015 |
| SASRec | tail | 3948 | 1.2676 | -0.7015 |
| CANDSSASRec | all | 11846 | 1.8951 | -0.1637 |
| CANDSSASRec | head | 3949 | 1.8580 | -0.1637 |
| CANDSSASRec | mid | 3949 | 1.9054 | -0.1637 |
| CANDSSASRec | tail | 3948 | 1.9219 | -0.1637 |

## Within-checkpoint dot-versus-cosine counterfactual

| model | target group | n | dot R@10 | cosine R@10 | Δ R@10 | dot NDCG@10 | cosine NDCG@10 | angular-win / radial-displacement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SASRec | all | 19412 | 0.0720 | 0.0699 | -0.0022 | 0.0414 | 0.0409 | 0.0241 / 0.0103 |
| SASRec | head | 9927 | 0.0896 | 0.0984 | +0.0089 | 0.0515 | 0.0585 | 0.0376 / 0.0187 |
| SASRec | mid | 3913 | 0.0708 | 0.0544 | -0.0164 | 0.0388 | 0.0300 | 0.0138 / 0.0031 |
| SASRec | tail | 5227 | 0.0444 | 0.0318 | -0.0126 | 0.0269 | 0.0183 | 0.0078 / 0.0004 |
| CANDSSASRec | all | 19412 | 0.0838 | 0.0929 | +0.0091 | 0.0423 | 0.0508 | 0.0338 / 0.0252 |
| CANDSSASRec | head | 9927 | 0.1055 | 0.1207 | +0.0152 | 0.0553 | 0.0682 | 0.0485 / 0.0350 |
| CANDSSASRec | mid | 3913 | 0.0851 | 0.0846 | -0.0005 | 0.0409 | 0.0444 | 0.0307 / 0.0240 |
| CANDSSASRec | tail | 5227 | 0.0473 | 0.0526 | +0.0054 | 0.0214 | 0.0262 | 0.0107 / 0.0092 |

## Interpretation boundary

- A positive cosine-minus-dot difference in the SASRec counterfactual shows that removing radial factors changes rankings favorably for that fixed representation; it is not, by itself, evidence that retraining with cosine will improve every dataset.
- `radial_displacement_rate` counts examples where the target is directionally closer than SASRec's strongest dot-product competitor but loses to that competitor under the dot product. It operationalizes the proposed rank-reversal mechanism.
- The trained CaNDS rows remain a complementary endpoint comparison. Causal performance claims require matched seeds and datasets, which the accompanying training grid supplies.
