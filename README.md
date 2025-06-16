

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

## 📈 Logged Metrics (Per Epoch)

Saved in CSVs (e.g. `val_metrics_fold2.csv`, `test_metrics.csv`):

### Classification:

* Accuracy
* Precision (macro, weighted)
* Recall (macro, weighted)
* F1 Score (macro, binary, weighted)
* Confusion Matrix (saved as PNG)

### Regression:

* MAE
* MSE
* RMSE
* MAPE

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
![Slice Image with Predictions](/results/working/Test_CM.png)
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
