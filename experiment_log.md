Exp01 - baseline_basic
- overall_acc: 0.9694
- cats_acc: 0.9763
- dogs_acc: 0.9625
- wrong_count: 62
- notes: baseline

Exp02 - baseline_rot
- overall_acc: 0.9733
- cats_acc: 0.9802
- dogs_acc: 0.9664
- wrong_count: 54
- notes: best overall and most balanced; recommended as current strongest baseline

Exp03 - baseline_color
- overall_acc: 0.9723
- cats_acc: 0.9693
- dogs_acc: 0.9753
- wrong_count: 56
- notes: improves dogs significantly but hurts cats; less stable than rot
Exp04 - se_rot
- best_epoch: 14
- best_val_acc: 0.9728
- overall_acc: 0.9728
- cats_acc: 0.9713
- dogs_acc: 0.9743
- wrong_count: 55
- notes: improves dogs, hurts cats, more balanced but not better overall than rot baseline
