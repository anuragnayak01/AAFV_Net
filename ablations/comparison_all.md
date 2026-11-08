# Complete comparison (calibration split, development numbers)

Ours = `full`. Every metric is listed; nothing is filtered. 'better' = clearly higher for ours; 'worse' = clearly lower.

## `full` vs `arcface_only`
better: 3 | no clear difference: 0 | worse: 15

### better
- decision table (false-reject target 0.005), D_hard_rej: ours 0.955 vs 0.000
- decision table (false-reject target 0.01), D_hard_rej: ours 0.982 vs 0.000
- decision table (false-reject target 0.02), D_hard_rej: ours 0.991 vs 0.000

### no clear difference

### worse
- decision table (false-reject target 0.005), same_acc: ours 0.741 vs 0.872
- decision table (false-reject target 0.005), same_acc_G4G6: ours 0.000 vs 0.143
- decision table (false-reject target 0.005), D_random_rej: ours 0.088 vs 0.310
- decision table (false-reject target 0.005), D_all_random_rej: ours 0.102 vs 0.294
- decision table (false-reject target 0.005), automation_balanced: ours 0.424 vs 0.586
- decision table (false-reject target 0.01), same_acc: ours 0.741 vs 0.872
- decision table (false-reject target 0.01), same_acc_G4G6: ours 0.000 vs 0.143
- decision table (false-reject target 0.01), D_random_rej: ours 0.262 vs 0.479
- decision table (false-reject target 0.01), D_all_random_rej: ours 0.280 vs 0.459
- decision table (false-reject target 0.01), automation_balanced: ours 0.516 vs 0.670
- decision table (false-reject target 0.02), same_acc: ours 0.741 vs 0.872
- decision table (false-reject target 0.02), same_acc_G4G6: ours 0.000 vs 0.143
- decision table (false-reject target 0.02), D_random_rej: ours 0.644 vs 0.928
- decision table (false-reject target 0.02), D_all_random_rej: ours 0.672 vs 0.922
- decision table (false-reject target 0.02), automation_balanced: ours 0.717 vs 0.907

## `full` vs `direct5_logreg`
better: 3 | no clear difference: 0 | worse: 15

### better
- decision table (false-reject target 0.005), D_hard_rej: ours 0.955 vs 0.000
- decision table (false-reject target 0.01), D_hard_rej: ours 0.982 vs 0.000
- decision table (false-reject target 0.02), D_hard_rej: ours 0.991 vs 0.001

### no clear difference

### worse
- decision table (false-reject target 0.005), same_acc: ours 0.741 vs 0.843
- decision table (false-reject target 0.005), same_acc_G4G6: ours 0.000 vs 0.143
- decision table (false-reject target 0.005), D_random_rej: ours 0.088 vs 0.322
- decision table (false-reject target 0.005), D_all_random_rej: ours 0.102 vs 0.315
- decision table (false-reject target 0.005), automation_balanced: ours 0.424 vs 0.582
- decision table (false-reject target 0.01), same_acc: ours 0.741 vs 0.843
- decision table (false-reject target 0.01), same_acc_G4G6: ours 0.000 vs 0.143
- decision table (false-reject target 0.01), D_random_rej: ours 0.262 vs 0.573
- decision table (false-reject target 0.01), D_all_random_rej: ours 0.280 vs 0.552
- decision table (false-reject target 0.01), automation_balanced: ours 0.516 vs 0.702
- decision table (false-reject target 0.02), same_acc: ours 0.741 vs 0.843
- decision table (false-reject target 0.02), same_acc_G4G6: ours 0.000 vs 0.143
- decision table (false-reject target 0.02), D_random_rej: ours 0.644 vs 0.932
- decision table (false-reject target 0.02), D_all_random_rej: ours 0.672 vs 0.927
- decision table (false-reject target 0.02), automation_balanced: ours 0.717 vs 0.895

## `full` vs `direct_only_mlp`
better: 6 | no clear difference: 10 | worse: 2

### better
- decision table (false-reject target 0.005), same_acc: ours 0.741 vs 0.000
- decision table (false-reject target 0.005), automation_balanced: ours 0.424 vs 0.048
- decision table (false-reject target 0.01), same_acc: ours 0.741 vs 0.000
- decision table (false-reject target 0.01), automation_balanced: ours 0.516 vs 0.139
- decision table (false-reject target 0.02), same_acc: ours 0.741 vs 0.000
- decision table (false-reject target 0.02), automation_balanced: ours 0.717 vs 0.416

### no clear difference
- decision table (false-reject target 0.005), same_acc_G4G6: ours 0.000 vs 0.000
- decision table (false-reject target 0.005), D_random_rej: ours 0.088 vs 0.086
- decision table (false-reject target 0.005), D_all_random_rej: ours 0.102 vs 0.090
- decision table (false-reject target 0.005), D_hard_rej: ours 0.955 vs 0.965
- decision table (false-reject target 0.01), same_acc_G4G6: ours 0.000 vs 0.000
- decision table (false-reject target 0.01), D_random_rej: ours 0.262 vs 0.267
- decision table (false-reject target 0.01), D_all_random_rej: ours 0.280 vs 0.267
- decision table (false-reject target 0.01), D_hard_rej: ours 0.982 vs 0.985
- decision table (false-reject target 0.02), same_acc_G4G6: ours 0.000 vs 0.000
- decision table (false-reject target 0.02), D_hard_rej: ours 0.991 vs 0.991

### worse
- decision table (false-reject target 0.02), D_random_rej: ours 0.644 vs 0.795
- decision table (false-reject target 0.02), D_all_random_rej: ours 0.672 vs 0.813

## `full` vs `baseline_siamese_cnn`
better: 9 | no clear difference: 3 | worse: 6

### better
- decision table (false-reject target 0.005), same_acc: ours 0.741 vs 0.003
- decision table (false-reject target 0.005), D_hard_rej: ours 0.955 vs 0.049
- decision table (false-reject target 0.005), automation_balanced: ours 0.424 vs 0.174
- decision table (false-reject target 0.01), same_acc: ours 0.741 vs 0.003
- decision table (false-reject target 0.01), D_hard_rej: ours 0.982 vs 0.109
- decision table (false-reject target 0.01), automation_balanced: ours 0.516 vs 0.265
- decision table (false-reject target 0.02), same_acc: ours 0.741 vs 0.003
- decision table (false-reject target 0.02), D_hard_rej: ours 0.991 vs 0.342
- decision table (false-reject target 0.02), automation_balanced: ours 0.717 vs 0.414

### no clear difference
- decision table (false-reject target 0.005), same_acc_G4G6: ours 0.000 vs 0.000
- decision table (false-reject target 0.01), same_acc_G4G6: ours 0.000 vs 0.000
- decision table (false-reject target 0.02), same_acc_G4G6: ours 0.000 vs 0.000

### worse
- decision table (false-reject target 0.005), D_random_rej: ours 0.088 vs 0.340
- decision table (false-reject target 0.005), D_all_random_rej: ours 0.102 vs 0.340
- decision table (false-reject target 0.01), D_random_rej: ours 0.262 vs 0.497
- decision table (false-reject target 0.01), D_all_random_rej: ours 0.280 vs 0.518
- decision table (false-reject target 0.02), D_random_rej: ours 0.644 vs 0.809
- decision table (false-reject target 0.02), D_all_random_rej: ours 0.672 vs 0.805

## `full` vs `baseline_siamese_vit`
better: 9 | no clear difference: 3 | worse: 6

### better
- decision table (false-reject target 0.005), same_acc: ours 0.741 vs 0.000
- decision table (false-reject target 0.005), D_hard_rej: ours 0.955 vs 0.041
- decision table (false-reject target 0.005), automation_balanced: ours 0.424 vs 0.162
- decision table (false-reject target 0.01), same_acc: ours 0.741 vs 0.000
- decision table (false-reject target 0.01), D_hard_rej: ours 0.982 vs 0.122
- decision table (false-reject target 0.01), automation_balanced: ours 0.516 vs 0.261
- decision table (false-reject target 0.02), same_acc: ours 0.741 vs 0.000
- decision table (false-reject target 0.02), D_hard_rej: ours 0.991 vs 0.345
- decision table (false-reject target 0.02), automation_balanced: ours 0.717 vs 0.420

### no clear difference
- decision table (false-reject target 0.005), same_acc_G4G6: ours 0.000 vs 0.000
- decision table (false-reject target 0.01), same_acc_G4G6: ours 0.000 vs 0.000
- decision table (false-reject target 0.02), same_acc_G4G6: ours 0.000 vs 0.000

### worse
- decision table (false-reject target 0.005), D_random_rej: ours 0.088 vs 0.304
- decision table (false-reject target 0.005), D_all_random_rej: ours 0.102 vs 0.319
- decision table (false-reject target 0.01), D_random_rej: ours 0.262 vs 0.520
- decision table (false-reject target 0.01), D_all_random_rej: ours 0.280 vs 0.513
- decision table (false-reject target 0.02), D_random_rej: ours 0.644 vs 0.794
- decision table (false-reject target 0.02), D_all_random_rej: ours 0.672 vs 0.821

## `full` vs `baseline_vlm`
better: 9 | no clear difference: 3 | worse: 6

### better
- decision table (false-reject target 0.005), same_acc: ours 0.741 vs 0.000
- decision table (false-reject target 0.005), D_hard_rej: ours 0.955 vs 0.052
- decision table (false-reject target 0.005), automation_balanced: ours 0.424 vs 0.149
- decision table (false-reject target 0.01), same_acc: ours 0.741 vs 0.000
- decision table (false-reject target 0.01), D_hard_rej: ours 0.982 vs 0.102
- decision table (false-reject target 0.01), automation_balanced: ours 0.516 vs 0.240
- decision table (false-reject target 0.02), same_acc: ours 0.741 vs 0.000
- decision table (false-reject target 0.02), D_hard_rej: ours 0.991 vs 0.233
- decision table (false-reject target 0.02), automation_balanced: ours 0.717 vs 0.380

### no clear difference
- decision table (false-reject target 0.005), same_acc_G4G6: ours 0.000 vs 0.000
- decision table (false-reject target 0.01), same_acc_G4G6: ours 0.000 vs 0.000
- decision table (false-reject target 0.02), same_acc_G4G6: ours 0.000 vs 0.000

### worse
- decision table (false-reject target 0.005), D_random_rej: ours 0.088 vs 0.292
- decision table (false-reject target 0.005), D_all_random_rej: ours 0.102 vs 0.292
- decision table (false-reject target 0.01), D_random_rej: ours 0.262 vs 0.470
- decision table (false-reject target 0.01), D_all_random_rej: ours 0.280 vs 0.471
- decision table (false-reject target 0.02), D_random_rej: ours 0.644 vs 0.726
- decision table (false-reject target 0.02), D_all_random_rej: ours 0.672 vs 0.741
