**Logistic regression (primary)**, mean +/- std over 5 seeds

| Setting | Accuracy | Recall | False-positive rate | AUC |
| --- | --- | --- | --- | --- |
| Random-split CV (families seen) | 0.977 +/- 0.009 | 0.972 +/- 0.015 | 0.020 +/- 0.007 | 0.997 +/- 0.001 |
| Family-wise OOF (unseen families) | 0.849 +/- 0.064 | 0.942 +/- 0.051 | 0.213 +/- 0.071 | 0.928 +/- 0.051 |
| Held-out test families | 0.821 +/- 0.146 | 0.927 +/- 0.053 | 0.211 +/- 0.229 | 0.935 +/- 0.083 |

Test files per seed: 302 to 479; inference latency 1.95 ms per file (excludes feature extraction)

**Random Forest (comparison)**, mean +/- std over 5 seeds

| Setting | Accuracy | Recall | False-positive rate | AUC |
| --- | --- | --- | --- | --- |
| Random-split CV (families seen) | 0.970 +/- 0.007 | 0.971 +/- 0.012 | 0.032 +/- 0.011 | 0.996 +/- 0.002 |
| Family-wise OOF (unseen families) | 0.752 +/- 0.091 | 0.930 +/- 0.051 | 0.368 +/- 0.120 | 0.843 +/- 0.108 |
| Held-out test families | 0.787 +/- 0.167 | 0.951 +/- 0.029 | 0.268 +/- 0.245 | 0.862 +/- 0.160 |

Test files per seed: 302 to 479; inference latency 21.55 ms per file (excludes feature extraction)

