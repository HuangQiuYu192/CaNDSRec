# Directional calibration diagnostics

The counterfactual rows compare dot-product and cosine rankings within the same checkpoint. They therefore isolate scoring geometry, not independently trained-model differences.

## Item-norm / popularity coupling

| model | group | n | mean item norm | Spearman(norm, popularity) |
| --- | --- | ---: | ---: | ---: |
| SASRec | all | 18229 | 0.8985 | -0.8430 |
| SASRec | head | 6077 | 0.7825 | -0.8430 |
| SASRec | mid | 6076 | 0.9016 | -0.8430 |
| SASRec | tail | 6076 | 1.0113 | -0.8430 |
| CANDSSASRec | all | 18229 | 1.2540 | -0.1491 |
| CANDSSASRec | head | 6077 | 1.2386 | -0.1491 |
| CANDSSASRec | mid | 6076 | 1.2617 | -0.1491 |
| CANDSSASRec | tail | 6076 | 1.2616 | -0.1491 |

## Within-checkpoint dot-versus-cosine counterfactual

| model | target group | n | dot R@10 | cosine R@10 | Δ R@10 | dot NDCG@10 | cosine NDCG@10 | angular-win / radial-displacement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SASRec | all | 35598 | 0.0437 | 0.0429 | -0.0008 | 0.0203 | 0.0195 | 0.0038 / 0.0028 |
| SASRec | head | 19825 | 0.0684 | 0.0723 | +0.0039 | 0.0317 | 0.0330 | 0.0064 / 0.0046 |
| SASRec | mid | 6875 | 0.0164 | 0.0099 | -0.0065 | 0.0079 | 0.0045 | 0.0009 / 0.0007 |
| SASRec | tail | 8484 | 0.0103 | 0.0028 | -0.0074 | 0.0047 | 0.0012 | 0.0002 / 0.0001 |
| CANDSSASRec | all | 35598 | 0.0493 | 0.0547 | +0.0054 | 0.0233 | 0.0271 | 0.0091 / 0.0069 |
| CANDSSASRec | head | 19825 | 0.0761 | 0.0859 | +0.0098 | 0.0363 | 0.0430 | 0.0153 / 0.0115 |
| CANDSSASRec | mid | 6875 | 0.0256 | 0.0244 | -0.0012 | 0.0115 | 0.0117 | 0.0026 / 0.0022 |
| CANDSSASRec | tail | 8484 | 0.0083 | 0.0091 | +0.0008 | 0.0034 | 0.0039 | 0.0002 / 0.0002 |

## Interpretation boundary

- A positive cosine-minus-dot difference in the SASRec counterfactual shows that removing radial factors changes rankings favorably for that fixed representation; it is not, by itself, evidence that retraining with cosine will improve every dataset.
- `radial_displacement_rate` counts examples where the target is directionally closer than SASRec's strongest dot-product competitor but loses to that competitor under the dot product. It operationalizes the proposed rank-reversal mechanism.
- The trained CaNDS rows remain a complementary endpoint comparison. Causal performance claims require matched seeds and datasets, which the accompanying training grid supplies.
