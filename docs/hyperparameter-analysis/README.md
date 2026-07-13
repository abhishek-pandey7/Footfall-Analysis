# Hyperparameter Sensitivity Analysis (Stage 1 Training & Pure SVM Baseline for Age Classification)

This folder documents the following:
1. [Gender Classifier Hyperparameter Sensitivity Analysis](gender-classifier-hyperparameter-sensitivity-analysis/README.md): Experiments conducted during the Stage 1 base attribute training of the **ViT-B/16** model on the **PETA dataset**.
2. [Age Classifier Hyperparameter Sensitivity Analysis](age-classifier-hyperparameter-sensitivity-analysis/README.md): Experiments conducted for the **Pure SVM** age classification baseline, which replicates the original PETA benchmark methodology (region-based color + texture histograms with one intersection/RBF-kernel SVM per attribute).

The purpose of these experiments was to determine the configuration that offers optimal classification accuracy, smooth convergence, and generalization stability, while preventing overfitting collapse. Secondly, determine the feature-extraction and training configuration that offers the highest screening accuracy (average mA) on a 30% subsample of the training and validation data, before locking in the final configuration used for full-scale training.

## Gender Classifier Hyperparameter Summary

Below is a summary of the hyperparameters evaluated, the values tested, and the selected optimal configuration for the Gender Classifier:

| Hyperparameter | Tested Values | Optimal Value | Conclusion & Impact |
| :--- | :--- | :--- | :--- |
| **[Backbone Learning Rate](gender-classifier-hyperparameter-sensitivity-analysis/learning_rate.md)** | `1e-4`, `1e-5`, `1e-7` | **`1e-5`** | Avoids underfitting seen at `1e-7` and offers a smoother, safer convergence curve than `1e-4` on longer runs. |
| **[Linear Probing Duration](gender-classifier-hyperparameter-sensitivity-analysis/linear_probing.md)** | `0` to `4` epochs | **`4` epochs** | Some degree of linear probing meaningfully aids generalization, with 4 epochs yielding the highest final validation accuracy (97.8%). |
| **[Weight Decay](gender-classifier-hyperparameter-sensitivity-analysis/weight_decay.md)** | `0.0`, `0.001`, `0.01`, `0.05`, `0.1`, `0.3`, `0.5` | **`0.3`** | A weight decay of 0.3 achieved the lowest generalization gap and highest final validation accuracy, demonstrating clear benefit from regularization. |
| **[Warmup Ratio](gender-classifier-hyperparameter-sensitivity-analysis/warmup_ratio.md)** | `0.0`, `0.05`, `0.1`, `0.2`, `0.3`, `0.4` | **`0.2`** | A 20% warmup ratio achieved the highest validation accuracy (97.68%), confirming that warmup stabilizes and improves final training performance. |

---

## Age Classifier Hyperparameter Summary

Below is a summary of the feature extraction parameters evaluated, the values tested, and the selected optimal configuration for the Age Classifier SVM Baseline:

| Hyperparameter | Tested Values | Optimal Value | Conclusion & Impact |
| :--- | :--- | :--- | :--- |
| **[Number of Body Regions](age-classifier-hyperparameter-sensitivity-analysis/n_regions.md)** | `2, 3, 4, 5, 6, 7` | **`4`** | Accuracy peaks at 4 regions (0.7457); finer slicing beyond this point fragments regions into strips too thin for reliable histograms, adding noise instead of detail. |
| **[Color Histogram Bin Count](age-classifier-hyperparameter-sensitivity-analysis/color_bins.md)** | `8, 16, 24, 32` | **`16`** | 16 bins achieves the highest accuracy (0.7457); finer binning beyond 16 sparsifies histograms as region size and bin count both shrink relative to it. |
| **[LBP Sampling Points](age-classifier-hyperparameter-sensitivity-analysis/lbp_points.md)** | `8, 16, 24` | **`16`** | 16 points achieves the highest accuracy (0.7493); 8 points misses discriminative detail, while 24 points shows diminishing returns from added noise. |
| **[Multi-scale LBP Radii](age-classifier-hyperparameter-sensitivity-analysis/lbp_radii.md)** | `[1], [1,2], [1,2,3], [1,2,3,4], [2,3]` | **`[1, 2, 3]`** | All configurations score very similarly, indicating low sensitivity to the specific radii combination; `[1,2,3]` scores marginally highest. |
| **[Class Weighting](age-classifier-hyperparameter-sensitivity-analysis/class_weight.md)** | `none`, `balanced` | **`balanced`** | Balanced weighting achieves a clearly higher accuracy (0.7493 vs. 0.7432), confirming the benefit of compensating for age-bucket imbalance during SVM training. |

## Detailed Analyses

To explore the findings, metrics, and visualization charts for each parameter, refer to the individual parameter files in their respective folders:

### Gender Classifier Analysis
1. **[Backbone Learning Rate Analysis](gender-classifier-hyperparameter-sensitivity-analysis/learning_rate.md)** - Details the validation accuracy convergence charts for learning rates.
2. **[Linear Probing Duration Analysis](gender-classifier-hyperparameter-sensitivity-analysis/linear_probing.md)** - Explains the training loss comparison between skipping and incorporating probing epochs.
3. **[Weight Decay Analysis](gender-classifier-hyperparameter-sensitivity-analysis/weight_decay.md)** - Discusses the generalization gap findings over short training horizons.
4. **[Warmup Ratio Analysis](gender-classifier-hyperparameter-sensitivity-analysis/warmup_ratio.md)** - Examines the step-by-step training loss trends for different warmup schedules.

### Age Classifier Analysis
1. **[Number of Body Regions Analysis](age-classifier-hyperparameter-sensitivity-analysis/n_regions.md)** - Details the accuracy trend across region counts from 2 to 7.
2. **[Color Histogram Bin Count Analysis](age-classifier-hyperparameter-sensitivity-analysis/color_bins.md)** - Examines the effect of finer vs. coarser color binning.
3. **[LBP Sampling Points Analysis](age-classifier-hyperparameter-sensitivity-analysis/lbp_points.md)** - Discusses the impact of texture descriptor resolution.
4. **[Multi-scale LBP Radii Analysis](age-classifier-hyperparameter-sensitivity-analysis/lbp_radii.md)** - Compares different multi-scale radius combinations for texture extraction.
5. **[Class Weighting Analysis](age-classifier-hyperparameter-sensitivity-analysis/class_weight.md)** - Evaluates the benefit of balanced class weighting for imbalanced age buckets.
