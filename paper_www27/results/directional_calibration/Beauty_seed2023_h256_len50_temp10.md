# Directional calibration diagnostics

The counterfactual rows compare dot-product and cosine rankings within the same checkpoint. They therefore isolate scoring geometry, not independently trained-model differences.

## Item-norm / popularity coupling

| model | group | n | mean item norm | Spearman(norm, popularity) |
| --- | --- | ---: | ---: | ---: |
| SASRec | all | 12042 | 0.7282 | -0.6931 |
| SASRec | head | 4014 | 0.6680 | -0.6931 |
| SASRec | mid | 4014 | 0.7259 | -0.6931 |
| SASRec | tail | 4014 | 0.7907 | -0.6931 |
| CANDSSASRec | all | 12042 | 1.5121 | -0.2183 |
| CANDSSASRec | head | 4014 | 1.4805 | -0.2183 |
| CANDSSASRec | mid | 4014 | 1.5185 | -0.2183 |
| CANDSSASRec | tail | 4014 | 1.5371 | -0.2183 |

## Within-checkpoint dot-versus-cosine counterfactual

| model | target group | n | dot R@10 | cosine R@10 | Δ R@10 | dot NDCG@10 | cosine NDCG@10 | angular-win / radial-displacement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SASRec | all | 22363 | 0.0818 | 0.0772 | -0.0046 | 0.0399 | 0.0372 | 0.0078 / 0.0052 |
| SASRec | head | 12707 | 0.1217 | 0.1225 | +0.0009 | 0.0597 | 0.0591 | 0.0134 / 0.0088 |
| SASRec | mid | 4183 | 0.0330 | 0.0241 | -0.0088 | 0.0163 | 0.0122 | 0.0010 / 0.0010 |
| SASRec | tail | 5277 | 0.0277 | 0.0131 | -0.0146 | 0.0123 | 0.0055 | 0.0002 / 0.0002 |
| CANDSSASRec | all | 22363 | 0.0777 | 0.0885 | +0.0108 | 0.0373 | 0.0471 | 0.0343 / 0.0287 |
| CANDSSASRec | head | 12707 | 0.1129 | 0.1295 | +0.0167 | 0.0548 | 0.0705 | 0.0565 / 0.0471 |
| CANDSSASRec | mid | 4183 | 0.0414 | 0.0437 | +0.0024 | 0.0194 | 0.0214 | 0.0072 / 0.0060 |
| CANDSSASRec | tail | 5277 | 0.0248 | 0.0286 | +0.0038 | 0.0107 | 0.0127 | 0.0036 / 0.0032 |

## Interpretation boundary

- A positive cosine-minus-dot difference in the SASRec counterfactual shows that removing radial factors changes rankings favorably for that fixed representation; it is not, by itself, evidence that retraining with cosine will improve every dataset.
- `radial_displacement_rate` counts examples where the target is directionally closer than SASRec's strongest dot-product competitor but loses to that competitor under the dot product. It operationalizes the proposed rank-reversal mechanism.
- The trained CaNDS rows remain a complementary endpoint comparison. Causal performance claims require matched seeds and datasets, which the accompanying training grid supplies.
