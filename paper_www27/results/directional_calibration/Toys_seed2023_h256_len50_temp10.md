# Directional calibration diagnostics

The counterfactual rows compare dot-product and cosine rankings within the same checkpoint. They therefore isolate scoring geometry, not independently trained-model differences.

## Item-norm / popularity coupling

| model | group | n | mean item norm | Spearman(norm, popularity) |
| --- | --- | ---: | ---: | ---: |
| SASRec | all | 11846 | 0.8724 | -0.7211 |
| SASRec | head | 3949 | 0.7862 | -0.7211 |
| SASRec | mid | 3949 | 0.8794 | -0.7211 |
| SASRec | tail | 3948 | 0.9516 | -0.7211 |
| CANDSSASRec | all | 11846 | 0.9652 | -0.1885 |
| CANDSSASRec | head | 3949 | 0.9533 | -0.1885 |
| CANDSSASRec | mid | 3949 | 0.9678 | -0.1885 |
| CANDSSASRec | tail | 3948 | 0.9744 | -0.1885 |

## Within-checkpoint dot-versus-cosine counterfactual

| model | target group | n | dot R@10 | cosine R@10 | Δ R@10 | dot NDCG@10 | cosine NDCG@10 | angular-win / radial-displacement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SASRec | all | 19412 | 0.0804 | 0.0789 | -0.0015 | 0.0412 | 0.0400 | 0.0108 / 0.0054 |
| SASRec | head | 9927 | 0.1014 | 0.1118 | +0.0104 | 0.0517 | 0.0574 | 0.0176 / 0.0095 |
| SASRec | mid | 3913 | 0.0744 | 0.0613 | -0.0130 | 0.0376 | 0.0301 | 0.0069 / 0.0026 |
| SASRec | tail | 5227 | 0.0503 | 0.0348 | -0.0155 | 0.0269 | 0.0168 | 0.0015 / 0.0000 |
| CANDSSASRec | all | 19412 | 0.0931 | 0.0959 | +0.0027 | 0.0447 | 0.0482 | 0.0163 / 0.0107 |
| CANDSSASRec | head | 9927 | 0.1204 | 0.1255 | +0.0051 | 0.0600 | 0.0655 | 0.0249 / 0.0156 |
| CANDSSASRec | mid | 3913 | 0.0907 | 0.0892 | -0.0015 | 0.0423 | 0.0439 | 0.0153 / 0.0115 |
| CANDSSASRec | tail | 5227 | 0.0494 | 0.0509 | +0.0015 | 0.0206 | 0.0219 | 0.0019 / 0.0013 |

## Interpretation boundary

- A positive cosine-minus-dot difference in the SASRec counterfactual shows that removing radial factors changes rankings favorably for that fixed representation; it is not, by itself, evidence that retraining with cosine will improve every dataset.
- `radial_displacement_rate` counts examples where the target is directionally closer than SASRec's strongest dot-product competitor but loses to that competitor under the dot product. It operationalizes the proposed rank-reversal mechanism.
- The trained CaNDS rows remain a complementary endpoint comparison. Causal performance claims require matched seeds and datasets, which the accompanying training grid supplies.
