

# 🧠 Liver Fat Classification & Regression Using Hybrid CNN (ResNet34)

This project performs **binary classification** (fatty vs. non-fatty liver) and **regression** (predicting fat percentage) on liver MRI slices using a hybrid CNN model. The system includes training, evaluation, inference, and detailed metric logging.


---

## 🧠 Model Architecture

### 🔸 `HybridNet` Class

* **Backbone:** Pretrained `ResNet34`
* **Heads:**

  * `cls_head`: Classification (2 classes)
  * `reg_head`: Regression (predicts fat %)

```python
class HybridNet(nn.Module):
    def __init__(self, num_classes=2):
        ...
```

---

## 🧪 Dataset Format

* **Input Format:** `.mat` file containing MRI volumes
* **Shape:** `(55, 10, H, W)`

  * 55 patients
  * 10 slices per patient
  * H x W resolution

---

## 🏋️ Training Details

* **Model:** ResNet34-based hybrid model
* **Data Handling:**

  * Uses all 10 slices per patient
  * Each slice treated as a separate sample (augments data x10)
* **K-Fold Cross Validation:** `k=5`
* **Losses:**

  * `CrossEntropyLoss` for classification
  * `MSELoss` for regression
* **Optimizations:**

  * Early stopping
  * Learning rate reduction on plateau

---

To complete your project documentation with **detailed metrics for every fold**, here’s an updated and extended version of the `📈 Logged Metrics (Per Epoch)` and a new section `📑 Per-Fold Summary` you can add to your markdown:

---

## 📈 Logged Metrics (Per Epoch)

Each fold logs the following metrics **per epoch** into a dedicated CSV file:
`val_metrics_fold{n}.csv` and `train_metrics_fold{n}.csv`

### 🔹 Classification (per epoch):

* `train_loss`, `val_loss`
* `train_acc`, `val_acc`
* `train_f1_macro`, `val_f1_macro`
* `train_f1_weighted`, `val_f1_weighted`
* `train_precision_macro`, `val_precision_macro`
* `train_recall_macro`, `val_recall_macro`

### 🔹 Regression (per epoch):

* `train_mae`, `val_mae`
* `train_mse`, `val_mse`
* `train_rmse`, `val_rmse`
* `train_r2`, `val_r2`

Additional details like learning rate (`lr`) and epoch index (`epoch`) are also logged.

---

## 📑 Per-Fold Summary

At the end of each fold, the following metrics are computed on **validation set** and saved to `fold_summary.csv`:

### 🔸 Classification:

* `Accuracy`
* `F1 Score (macro, weighted, binary)`
* `Precision (macro, weighted)`
* `Recall (macro, weighted)`
* `Confusion Matrix` saved as `CM_Fold{n}.png`

### 🔸 Regression:

* `MAE` (Mean Absolute Error)
* `MSE` (Mean Squared Error)
* `RMSE` (Root Mean Squared Error)
* `R²` Score
* `Fat% Scatter Plot` saved as `FatReg_Fold{n}.png`

---

## 🧾 Example Fold Metrics Summary

| Fold | Accuracy | F1 (Macro) | MAE  | MSE   | RMSE | R²   |
| ---- | -------- | ---------- | ---- | ----- | ---- | ---- |
| 1    | 0.811    | 0.794      | 7.91 | 10.22 | 3.20 | 0.82 |
| 2    | 0.833    | 0.821      | 8.04 | 10.55 | 3.25 | 0.79 |
| 3    | 0.818    | 0.805      | 8.32 | 11.01 | 3.32 | 0.78 |
| 4    | 0.790    | 0.775      | 8.55 | 11.47 | 3.38 | 0.77 |
| 5    | 0.826    | 0.808      | 7.87 | 9.89  | 3.14 | 0.83 |

Average values are also computed at the end.

---



## 📊 Example Outputs

**Test Metrics Summary:**

```
Accuracy: 0.818
F1 Score (macro): 0.800
MAE: 8.32
MSE: 11.01
```

**Confusion Matrix (saved per fold):**

🟩 True Positives
🟥 False Negatives
🟦 False Positives
⬜ True Negatives

---
![](/results/working/Test_CM.png)
![](/results/working/Fold1_Val_CM.png)
![](/results/working/Fold2_Val_CM.png)
![](/results/working/Fold3_Val_CM.png)
![](/results/working/Fold4_Val_CM.png)
![](/results/working/Fold5_Val_CM.png)

## 🧠 Inference Pipeline

A sample script is provided to load the model and perform inference on a middle slice of a 3D volume:

```bash
python inference_script.py
```

```python
inference(volume_3d, true_label, true_fat_percentage, model, scaler, transform)
```

### 🔧 Preprocessing

* Middle slice extraction
* Normalization to 0-255
* Transformed using `torchvision.transforms`

---

## 💾 How to Save and Load Model

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()
```

---

## 🔍 Example Inference Output

```
🔍 True cls = 1, Pred cls = 1
    True fat = 29.4%, Pred fat = 27.8%
```



---

## 🚀 Dependencies

```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn scipy
```

---

## 📈 Results & Observations

* The model shows strong performance in training and validation.
* False negatives were mitigated by augmenting the dataset (10x slices).
* Extensive logging and visualization support post-analysis and debugging.

---

## ✍️ Author

Developed by **Khawaja Murad**
