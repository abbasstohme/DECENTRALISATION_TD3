# DECENTRALISATION_TD3

## **1. Introduction**
This report explores a **decentralized machine learning approach**, aggregating predictions from multiple models:  
- **Logistic Regression** (Fast and simple)  
- **Random Forest** (Robust for complex datasets)  
- **SVM** (Effective for well-defined decision boundaries)  
- **KNN** (Uses proximity-based classification)  

It also introduces a **Proof-of-Stake (PoS) mechanism with slashing**, ensuring that underperforming models are penalized while reliable ones maintain influence.

---

## **2. Model Performance Evaluation**

The models were trained and evaluated on the **Iris dataset**, using a **test set of 30 samples**.  

### **Evaluation Results:**

| Model | Accuracy | Confusion Matrix | F1-score |
|-----------------|------------|---------------------------------|------------|
| **Logistic Regression** | 1.00 | `[[10, 0, 0], [0, 9, 0], [0, 0, 11]]` | 1.00 |
| **Random Forest** | 1.00 | `[[10, 0, 0], [0, 9, 0], [0, 0, 11]]` | 1.00 |
| **SVM** | 1.00 | `[[10, 0, 0], [0, 9, 0], [0, 0, 11]]` | 1.00 |
| **KNN** | 1.00 | `[[10, 0, 0], [0, 9, 0], [0, 0, 11]]` | 1.00 |

### **Analysis:**
- **All models achieved 100% accuracy**, correctly classifying all test samples.
- The **confusion matrices** show that **no misclassifications occurred**.
- **F1-scores** confirm **perfect precision and recall** across all classes.

⚠️ **However**, this dataset is simple. In real-world applications, these models may perform differently.

---

## **3. Aggregated Model Using Majority Voting**
The **aggregated model** combines the predictions from all four classifiers using a **majority vote** approach.

### **Key Benefits of Model Aggregation**
✅ **Robustness** – If a model makes a wrong prediction, the majority vote corrects it.  
✅ **Higher Performance** – Reduces errors from individual models.

### **Aggregated Model Accuracy**
✅ **Final Accuracy: 100%** on the Iris dataset.

📌 **Limitation:**  
- Since the Iris dataset is simple, the benefits of aggregation **may not be visible**.
- On more complex datasets, this approach would be more useful.

---

## **4. Weighted Aggregation with Dynamic Model Weights**
Instead of **equal voting**, a **dynamic weighting system** improves accuracy by **giving more importance to reliable models**.

### **How It Works:**
1. **Each model starts with equal weight (1.0).**  
2. **Weight adjustment rules:**
   - ✅ **+0.1 increase** if the model aligns with the consensus.
   - ❌ **-0.1 penalty** if the model disagrees.
   - 🔄 **Weights remain between [0.0, 1.0]**.

### **Results of Dynamic Weighting**
- Models that consistently agree with the consensus **gain more influence**.
- Poor-performing models **lose influence** over time.

🚀 **Future Applications:**  
- This method is useful in **real-world scenarios** where models have varying reliability.

---

## **5. Proof-of-Stake with Slashing Mechanism**
To ensure accountability, a **PoS-based system** with **slashing penalties** is introduced.

### **How It Works:**
1. **Each model deposits 1000 euros** to participate.
2. If a model **disagrees with the consensus**, it is **fined 50 euros**.
3. **When balance reaches 0**, the model is **removed from the system**.

### **Effects of Slashing**
- Prevents unreliable models from influencing results.
- Ensures **only high-performing models contribute**.

### **Final Results**
✅ **No penalties applied on the Iris dataset** (all models performed perfectly).  
⚠️ **On more complex datasets, models with frequent errors would be excluded over time.**

---

## **6. Running the Experiment**
### **Step 1: Install Dependencies**
```bash
pip install flask scikit-learn requests
python Code_1.py  # Logistic Regression
python Code_2.py  # Random Forest
python Code_3.py  # SVM
python Code_4.py  # KNN
python Aggregated_Model.py
python Proof_of_Stake_Slashing.py
http://127.0.0.1:5000/predict?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2