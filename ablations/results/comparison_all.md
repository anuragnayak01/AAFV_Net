# Complete comparison (calibration split, development numbers)

Ours = `full`. Every metric is listed; nothing is filtered. 'better' = clearly higher for ours; 'worse' = clearly lower.

## `full` vs `arcface_only`
better: 3 | no clear difference: 19 | worse: 14

### better
- decision table (false-reject target 0.005), D_hard_rej: ours 0.966 vs 0.000
- decision table (false-reject target 0.01), D_hard_rej: ours 0.982 vs 0.000
- decision table (false-reject target 0.02), D_hard_rej: ours 0.989 vs 0.739

### no clear difference
- AUC, gap bin G2: ours 0.833 vs 0.817
- AUC, gap bin G3: ours 0.839 vs 0.796
- AUC, gap bin G4: ours 0.501 vs 0.580
- AUC, gap bin G5: ours 0.639 vs 0.656
- TPR@FPR0.001, gap bin G1: ours 0.992 vs 0.990
- TPR@FPR0.001, gap bin G2: ours 0.668 vs 0.668
- TPR@FPR0.001, gap bin G3: ours 0.649 vs 0.649
- TPR@FPR0.001, gap bin G4: ours 0.023 vs 0.023
- TPR@FPR0.001, gap bin G5: ours 0.091 vs 0.091
- TPR@FPR0.001, gap bin G6: ours 0.143 vs 0.143
- TPR@FPR0.01, gap bin G1: ours 0.994 vs 0.995
- TPR@FPR0.01, gap bin G2: ours 0.668 vs 0.680
- TPR@FPR0.01, gap bin G3: ours 0.649 vs 0.655
- TPR@FPR0.01, gap bin G4: ours 0.045 vs 0.023
- TPR@FPR0.01, gap bin G5: ours 0.117 vs 0.091
- TPR@FPR0.01, gap bin G6: ours 0.143 vs 0.143
- decision table (false-reject target 0.005), same_acc_G4G6: ours 0.064 vs 0.083
- decision table (false-reject target 0.01), same_acc_G4G6: ours 0.064 vs 0.083
- decision table (false-reject target 0.02), same_acc_G4G6: ours 0.064 vs 0.083

### worse
- AUC, gap bin G1: ours 0.996 vs 0.999
- AUC, gap bin G6: ours 0.528 vs 0.818
- decision table (false-reject target 0.005), same_acc: ours 0.733 vs 0.877
- decision table (false-reject target 0.005), D_random_rej: ours 0.124 vs 0.413
- decision table (false-reject target 0.005), D_all_random_rej: ours 0.129 vs 0.400
- decision table (false-reject target 0.005), automation_balanced: ours 0.433 vs 0.641
- decision table (false-reject target 0.01), same_acc: ours 0.733 vs 0.877
- decision table (false-reject target 0.01), D_random_rej: ours 0.368 vs 0.643
- decision table (false-reject target 0.01), D_all_random_rej: ours 0.361 vs 0.627
- decision table (false-reject target 0.01), automation_balanced: ours 0.552 vs 0.757
- decision table (false-reject target 0.02), same_acc: ours 0.733 vs 0.877
- decision table (false-reject target 0.02), D_random_rej: ours 0.859 vs 0.999
- decision table (false-reject target 0.02), D_all_random_rej: ours 0.869 vs 1.000
- decision table (false-reject target 0.02), automation_balanced: ours 0.811 vs 0.948

## `full` vs `direct5_logreg`
better: 3 | no clear difference: 19 | worse: 14

### better
- decision table (false-reject target 0.005), D_hard_rej: ours 0.966 vs 0.000
- decision table (false-reject target 0.01), D_hard_rej: ours 0.982 vs 0.000
- decision table (false-reject target 0.02), D_hard_rej: ours 0.989 vs 0.804

### no clear difference
- AUC, gap bin G2: ours 0.833 vs 0.816
- AUC, gap bin G3: ours 0.839 vs 0.793
- AUC, gap bin G4: ours 0.501 vs 0.590
- AUC, gap bin G5: ours 0.639 vs 0.676
- TPR@FPR0.001, gap bin G1: ours 0.992 vs 0.991
- TPR@FPR0.001, gap bin G2: ours 0.668 vs 0.670
- TPR@FPR0.001, gap bin G3: ours 0.649 vs 0.649
- TPR@FPR0.001, gap bin G4: ours 0.023 vs 0.023
- TPR@FPR0.001, gap bin G5: ours 0.091 vs 0.091
- TPR@FPR0.001, gap bin G6: ours 0.143 vs 0.143
- TPR@FPR0.01, gap bin G1: ours 0.994 vs 0.995
- TPR@FPR0.01, gap bin G2: ours 0.668 vs 0.682
- TPR@FPR0.01, gap bin G3: ours 0.649 vs 0.655
- TPR@FPR0.01, gap bin G4: ours 0.045 vs 0.023
- TPR@FPR0.01, gap bin G5: ours 0.117 vs 0.091
- TPR@FPR0.01, gap bin G6: ours 0.143 vs 0.143
- decision table (false-reject target 0.005), same_acc_G4G6: ours 0.064 vs 0.064
- decision table (false-reject target 0.01), same_acc_G4G6: ours 0.064 vs 0.064
- decision table (false-reject target 0.02), same_acc_G4G6: ours 0.064 vs 0.064

### worse
- AUC, gap bin G1: ours 0.996 vs 0.999
- AUC, gap bin G6: ours 0.528 vs 0.838
- decision table (false-reject target 0.005), same_acc: ours 0.733 vs 0.849
- decision table (false-reject target 0.005), D_random_rej: ours 0.124 vs 0.395
- decision table (false-reject target 0.005), D_all_random_rej: ours 0.129 vs 0.384
- decision table (false-reject target 0.005), automation_balanced: ours 0.433 vs 0.619
- decision table (false-reject target 0.01), same_acc: ours 0.733 vs 0.849
- decision table (false-reject target 0.01), D_random_rej: ours 0.368 vs 0.609
- decision table (false-reject target 0.01), D_all_random_rej: ours 0.361 vs 0.587
- decision table (false-reject target 0.01), automation_balanced: ours 0.552 vs 0.723
- decision table (false-reject target 0.02), same_acc: ours 0.733 vs 0.849
- decision table (false-reject target 0.02), D_random_rej: ours 0.859 vs 1.000
- decision table (false-reject target 0.02), D_all_random_rej: ours 0.869 vs 1.000
- decision table (false-reject target 0.02), automation_balanced: ours 0.811 vs 0.934

## `full` vs `direct_only_mlp`
better: 13 | no clear difference: 21 | worse: 2

### better
- decision table (false-reject target 0.005), same_acc: ours 0.733 vs 0.000
- decision table (false-reject target 0.005), same_acc_G4G6: ours 0.064 vs 0.000
- decision table (false-reject target 0.005), D_random_rej: ours 0.124 vs 0.083
- decision table (false-reject target 0.005), D_all_random_rej: ours 0.129 vs 0.079
- decision table (false-reject target 0.005), automation_balanced: ours 0.433 vs 0.042
- decision table (false-reject target 0.01), same_acc: ours 0.733 vs 0.000
- decision table (false-reject target 0.01), same_acc_G4G6: ours 0.064 vs 0.000
- decision table (false-reject target 0.01), D_random_rej: ours 0.368 vs 0.308
- decision table (false-reject target 0.01), D_all_random_rej: ours 0.361 vs 0.310
- decision table (false-reject target 0.01), automation_balanced: ours 0.552 vs 0.160
- decision table (false-reject target 0.02), same_acc: ours 0.733 vs 0.000
- decision table (false-reject target 0.02), same_acc_G4G6: ours 0.064 vs 0.000
- decision table (false-reject target 0.02), automation_balanced: ours 0.811 vs 0.456

### no clear difference
- AUC, gap bin G1: ours 0.996 vs 0.996
- AUC, gap bin G2: ours 0.833 vs 0.863
- AUC, gap bin G3: ours 0.839 vs 0.850
- AUC, gap bin G4: ours 0.501 vs 0.432
- AUC, gap bin G5: ours 0.639 vs 0.534
- AUC, gap bin G6: ours 0.528 vs 0.337
- TPR@FPR0.001, gap bin G1: ours 0.992 vs 0.991
- TPR@FPR0.001, gap bin G2: ours 0.668 vs 0.667
- TPR@FPR0.001, gap bin G3: ours 0.649 vs 0.655
- TPR@FPR0.001, gap bin G4: ours 0.023 vs 0.023
- TPR@FPR0.001, gap bin G5: ours 0.091 vs 0.091
- TPR@FPR0.001, gap bin G6: ours 0.143 vs 0.143
- TPR@FPR0.01, gap bin G1: ours 0.994 vs 0.994
- TPR@FPR0.01, gap bin G2: ours 0.668 vs 0.675
- TPR@FPR0.01, gap bin G3: ours 0.649 vs 0.672
- TPR@FPR0.01, gap bin G4: ours 0.045 vs 0.023
- TPR@FPR0.01, gap bin G5: ours 0.117 vs 0.091
- TPR@FPR0.01, gap bin G6: ours 0.143 vs 0.143
- decision table (false-reject target 0.005), D_hard_rej: ours 0.966 vs 0.951
- decision table (false-reject target 0.01), D_hard_rej: ours 0.982 vs 0.979
- decision table (false-reject target 0.02), D_hard_rej: ours 0.989 vs 0.987

### worse
- decision table (false-reject target 0.02), D_random_rej: ours 0.859 vs 0.891
- decision table (false-reject target 0.02), D_all_random_rej: ours 0.869 vs 0.892
