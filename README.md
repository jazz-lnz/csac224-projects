# CSAC 224: Machine Learning Projects

A collection of Supervised Learning, Unsupervised Clustering, Statistical Anomaly Detection, and Regression Analysis projects completed for **CSAC 224 (Machine Learning)**. This repository explores classification on imbalanced sensor data, operational segmentation, probability-based outlier detection, and empirical bias-variance tradeoff modeling using **Scikit-Learn**, **Imbalanced-Learn (SMOTE)**, **SciPy**, **Yellowbrick**, **Seaborn**, and **Plotly**.

> **Note on Datasets & Documentation:** Raw dataset files (`.csv`, `.txt`) and formal academic PDF reports are not committed to this repository for storage and privacy reasons. Direct links to the original datasets from the **UCI Machine Learning Repository** (as well as generation parameters for the synthetic datasets) are provided inside each respective Jupyter Notebook and in the sections below. All methodologies, experimental setups, and analytical findings from the formal reports are documented directly in this README.

---

## Repository Structure

```text
csac224-projects/
├── README.md                                  # Comprehensive project documentation & results
├── PS5_OccupancyDetection_SVM.ipynb           # Problem Set 5: Support Vector Machine & SMOTE
├── PS6_GasTurbine_KMeansClustering.ipynb      # Problem Set 6: K-Means Clustering & Elbow Method
├── PS7_OnlineShoppers_AnomalyDetection.ipynb  # Problem Set 7: Gaussian & K-Means Anomaly Detection
└── FinalProject_BiasVariance_Regression.ipynb # Final Project: Bias-Variance Tradeoff in Regression
```

---

## 1. Problem Set 5: Room Occupancy Detection with Support Vector Machines

### Overview & Dataset
This project applies a **Linear Support Vector Machine (SVM)** to predict room occupancy based on environmental sensor measurements using the **Occupancy Detection Dataset** from the UCI Machine Learning Repository (ID: 357).
- **Dataset Scope:** Combines three experimental data files (`datatest.txt`, `datatest2.txt`, and `datatraining.txt`) into a unified dataset of **20,560 samples**, with ground-truth occupancy obtained from time-stamped pictures taken every minute.
- **Features & Target:** 5 continuous environmental features (after dropping the `date` timestamp column): `Temperature` (in Celsius), `Humidity` (relative humidity in %), `Light` (in Lux), `CO2` (in ppm), and `HumidityRatio` (derived from temperature and relative humidity). The binary target `Occupancy` indicates whether the room is unoccupied (`0`, 15,810 instances / 76.90%) or occupied (`1`, 4,750 instances / 23.10%).
- **Dataset Access:** Original data is available at the [UCI Machine Learning Repository (ID: 357)](https://archive.ics.uci.edu/dataset/357/occupancy+detection).

### Methodology
1. **Exploratory Data Analysis (EDA) & Class Balancing:** Inspected feature distributions and class proportions using histograms and Seaborn pairplots. Because the original dataset exhibited a steep 76.90% vs. 23.10% class imbalance, **SMOTE (Synthetic Minority Oversampling Technique)** (`random_state=42`) was applied to balance the dataset to **31,620 samples** (15,810 per class).
2. **Model Training:** Split the data into 80% training (25,296 samples) and 20% testing (6,324 samples) sets and trained a **Linear SVM** classifier.
3. **Baseline Comparison (Imbalanced Data):** To verify that the model's strong generalization was not an artifact of synthetic oversampling, a separate Linear SVM was trained and evaluated on the original imbalanced dataset (16,448 training and 4,112 testing samples).

### Results & Findings

| Dataset Configuration | Split / Support | Overall Accuracy | Class 0 (Unoccupied) Precision / Recall / F1 | Class 1 (Occupied) Precision / Recall / F1 | Macro Avg F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SMOTE-Balanced SVM (Test)** | 6,324 samples | **99%** | 1.00 / 0.99 / 0.99 | **0.99 / 1.00 / 0.99** | **0.99** |
| **SMOTE-Balanced SVM (Train)** | 25,296 samples | **99%** | 1.00 / 0.99 / 0.99 | **0.99 / 1.00 / 0.99** | **0.99** |
| **Original Imbalanced SVM (Test)** | 4,112 samples | 99% | 1.00 / 0.99 / 0.99 | 0.95 / 0.99 / 0.97 | 0.98 |
| **Original Imbalanced SVM (Train)** | 16,448 samples | 99% | 1.00 / 0.99 / 0.99 | 0.96 / 1.00 / 0.98 | 0.99 |

- **Confusion Matrix Analysis:** On the SMOTE-balanced testing set (6,324 instances), the Linear SVM correctly classified **3,116 unoccupied rooms (TN)** and **3,157 occupied rooms (TP)**, with only **46 False Positives** and **5 False Negatives**.
- **Impact of SMOTE:** Comparing the balanced model against the baseline imbalanced model confirmed that applying SMOTE improved minority-class (`Class 1`) test precision from **0.95 to 0.99** and F1-score from **0.97 to 0.99**, addressing slight overfitting on the minority class.

---

## 2. Problem Set 6: Gas Turbine Emission & Performance Clustering

### Overview & Dataset
This project explores unsupervised learning using **K-Means Clustering** on the **Gas Turbine CO and NOx Emission Dataset** from the UCI Machine Learning Repository (ID: 551) to uncover operational efficiency patterns across varying ambient temperatures.
- **Dataset Scope:** Concatenates five annual sensor datasets (`gt_2011.csv` through `gt_2015.csv`) totaling **36,733 instances** of hourly aggregated sensor measurements from a gas turbine located in Turkey.
- **Features:** The dataset contains 11 continuous features spanning ambient conditions (`AT`, `AP`, `AH`), operating parameters (`AFDP`, `GTEP`, `TIT`, `TAT`, `CDP`), energy performance (`TEY`), and flue gas emissions (`CO`, `NOX`).
- **Selected Clustering Features:** Focused on **Ambient Temperature (`AT`)** and **Turbine Energy Yield (`TEY`)** to segment turbine performance under different thermal environments.
- **Dataset Access:** Original data is available at the [UCI Machine Learning Repository (ID: 551)](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set).

### Methodology
1. **Preprocessing:** Extracted `AT` and `TEY` into the feature matrix and standardized the variables using `StandardScaler` so distance calculations were not skewed by scale differences.
2. **Optimal Cluster Identification:** Evaluated K-Means models across $k \in [2, 10]$ using the **Elbow Method** (via Yellowbrick's `KElbowVisualizer`) and confirmed the result with **Silhouette Score** analysis. The rate of distortion decrease slowed significantly at **$k = 4$** (distortion score = `19,021.601`), establishing 4 as the optimal number of clusters.
3. **Cluster Segmentation:** Fitted the final `KMeans(n_clusters=4)` model and inverse-transformed the cluster centroids to interpret real-world operational regimes.

### Results & Findings
The 4 identified clusters reveal distinct operational regimes between ambient temperature and energy yield:
- **Cluster 0 (Green) — High AT, Low TEY:** Turbines operating in high-temperature environments exhibit reduced energy yield, pointing to thermal efficiency loss.
- **Cluster 1 (Blue) — Low AT, High TEY:** Cooler ambient temperatures correlate with increased turbine efficiency and optimal performance.
- **Cluster 2 (Yellow) — Moderate AT, Stable/High TEY:** Turbines maintain consistently high and stable energy yield under low-to-mid-range ambient temperatures.
- **Cluster 3 (Black) — Extreme AT Variations, Fluctuating TEY:** Mid-to-high ambient temperatures produce wide fluctuations in energy yield (low to above-mid-range), indicating the influence of additional operational factors beyond ambient temperature alone. These segments provide actionable baselines for predictive maintenance and thermal conditioning.

---

## 3. Problem Set 7: Online Shoppers Purchasing Intention Anomaly Detection

### Overview & Dataset
This project implements statistical and clustering-based **Anomaly Detection** on the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository (ID: 468) to identify irregular e-commerce browsing behaviors.
- **Dataset Scope:** Contains **12,330 user sessions** across 18 features derived from URL information, Google Analytics metrics, and e-commerce dynamics (84.5% / 10,422 negative sessions without a purchase, and 15.5% / 1,908 positive sessions ending in a purchase).
- **Selected Features:** Analyzed **`BounceRates`** (percentage of visitors entering and leaving a site from the same single page) as the primary behavioral metric, paired with **`ProductRelated_Duration`** (total time spent on product-related pages) as a contextual feature for visual interpretation.
- **Dataset Access:** Original data is available at the [UCI Machine Learning Repository (ID: 468)](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

### Methodology
1. **Data Cleaning & Distribution Fitting:** Replaced empty values with `NaN` and dropped incomplete rows. Plotted distribution histograms against a fitted normal probability density function (`scipy.stats.norm.fit`, $\mu = 0.02$, $\sigma = 0.05$) and modeled the feature using a **Gaussian Distribution** (`multivariate_normal`, mean $\mu \approx 0.022191$, covariance $\sigma \approx 0.002351$) to compute probability density values ($p$) for all sessions.
2. **Approach 1 — Empirical Thresholding:** Calculated the variance of the probability densities ($\text{Var}(p) \approx 3.9168$) and empirically flagged data points as anomalies where $p < 2 \times \text{Var}(p)$.
3. **Approach 2 — Automated Thresholding via K-Means Clustering:** Clustered the probability densities $p$ across $k \in [2, 150]$ and computed Silhouette Scores (mean score = `0.81`), selecting **$k = 69$** as the optimal cluster count closest to the mean silhouette score. Sorted the resulting cluster centroids to categorize anomalies into two severity tiers:
   - **Grade 1 Outliers (Serious Anomalies):** $p \le \text{centroid}_0$ ($\approx 0.01027$).
   - **Grade 2 Outliers (Moderate Anomalies):** $\text{centroid}_0 < p \le \text{centroid}_1$ ($\approx 0.13385$).

### Results & Behavioral Interpretation
Comparing the two anomaly detection approaches across `ProductRelated_Duration` (X-axis) and `BounceRates` (Y-axis) revealed two primary behavioral regimes:
- **Low `ProductRelated_Duration` & High `BounceRates`:** Represents very brief page visits followed by immediate exits. This behavior typically stems from users encountering irrelevant/misleading page content or technical navigation issues (e.g., page crashes) and is flagged as anomalous by **both** the empirical and K-Means approaches.
- **High `ProductRelated_Duration` & Low `BounceRates`:** Represents highly engaged users spending extended time browsing product pages without bouncing. While the strict empirical threshold flags these long-duration sessions, the K-Means clustering approach is more lenient and does not flag them, better accommodating borderline high-engagement shopping behaviors.

---

## 4. Final Project: Theoretical Analysis of the Bias-Variance Tradeoff in Regression Models

### Overview & Datasets
This study conducts a theoretical and empirical analysis of the **Bias-Variance Tradeoff** by evaluating how increasing regression model complexity impacts prediction error, explanatory power, and overfitting risk across synthetic datasets of varying underlying complexity.
- **Synthetic Dataset Generation:** Generated four univariate regression datasets (`degree1_dataset.csv`, `degree2_dataset.csv`, `degree5_dataset.csv`, `degree10_dataset.csv`) using NumPy (`np.random.seed(42)`), each consisting of **10,000 samples** with feature values $X \sim \mathcal{N}(0, 1)$ and a 20% Gaussian noise term ($\epsilon$) to simulate real-world data imperfections:
  - **Dataset 1 (Linear, Degree 1):** $y = 3x + \epsilon$
  - **Dataset 2 (Quadratic, Degree 2):** $y = -2x^2 + 3x + \epsilon$
  - **Dataset 3 (Polynomial, Degree 5):** $y = 5x^5 - 3x^3 + 2x + \epsilon$
  - **Dataset 4 (Polynomial, Degree 10):** $y = 0.1x^{10} - 0.5x^5 + 0.3x^3 - 0.2x + \epsilon$ *(coefficients scaled down to prevent extreme values)*

### Methodology
1. **Train-Test Partitioning:** Each of the four synthetic datasets was split 80/20 into training (8,000 samples) and testing (2,000 samples) sets (`random_state=1`).
2. **Model Complexity Progression:** Constructed four unregularized regression models of increasing flexibility using Scikit-Learn's `LinearRegression` and `PolynomialFeatures` pipelines: **Linear Regression (Degree 1)**, **Polynomial Regression (Degree 2)**, **Polynomial Regression (Degree 5)**, and **Polynomial Regression (Degree 10)**.
3. **Evaluation Metrics:** Trained and evaluated all 16 model-dataset combinations using **Training MSE** (to assess bias), **Testing MSE** (to assess generalization and variance via the train-test gap), **Train & Test $R^2$ Scores** (to measure explained variance), and **5-Fold Cross-Validated MSE** (`cross_val_score`).

### Results & Model Comparison

| Dataset | Model Complexity | Train MSE | Test MSE | 5-Fold CV MSE | Train $R^2$ | Test $R^2$ | Fit Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dataset 1**<br>*(Degree 1)* | **Degree 1 (Linear)** | **3.9945** | **4.0088** | **3.9967** | **0.7469** | **0.7435** | **Optimal Fit** |
| | Degree 2 | 3.9943 | 4.0095 | 3.9982 | 0.7470 | 0.7435 | Redundant Complexity |
| | Degree 5 | 3.9918 | 4.0091 | 3.9971 | 0.7471 | 0.7435 | Redundant Complexity |
| | Degree 10 | 3.9898 | 4.0093 | 4.0049 | 0.7472 | 0.7435 | Redundant Complexity |
| **Dataset 2**<br>*(Degree 2)* | Degree 1 (Linear) | 9.7643 | 9.4694 | 9.7705 | 0.5527 | 0.5579 | High Bias (Underfit) |
| | **Degree 2** | **3.9943** | **4.0095** | **3.9982** | **0.8170** | **0.8128** | **Optimal Fit** |
| | Degree 5 | 3.9918 | 4.0091 | 3.9971 | 0.8172 | 0.8128 | Redundant Complexity |
| | Degree 10 | 3.9898 | 4.0093 | 4.0049 | 0.8172 | 0.8128 | Redundant Complexity |
| **Dataset 3**<br>*(Degree 5)* | Degree 1 (Linear) | 575.8171 | 531.8750 | 575.9762 | 0.6606 | 0.6657 | Severe High Bias |
| | Degree 2 | 575.5476 | 532.2921 | 575.7392 | 0.6607 | 0.6655 | Severe High Bias |
| | **Degree 5** | **3.9918** | **4.0091** | **3.9971** | **0.9976** | **0.9975** | **Optimal Fit** |
| | Degree 10 | 3.9898 | 4.0093 | 4.0049 | 0.9976 | 0.9975 | Redundant Complexity |
| **Dataset 4**<br>*(Degree 10)* | Degree 1 (Linear) | 418.1519 | 364.7743 | 418.2896 | 0.0315 | 0.0283 | Severe High Bias |
| | Degree 2 | 163.8438 | 149.6909 | 163.9589 | 0.6205 | 0.6012 | High Bias (Underfit) |
| | Degree 5 | 28.5725 | 28.0748 | 28.6551 | 0.9338 | 0.9252 | Moderate Bias |
| | **Degree 10** | **3.9898** | **4.0093** | **4.0049** | **0.9908** | **0.9893** | **Optimal Fit** |

### Key Findings
- **Bias Reduction via Model Flexibility:** Lower-degree models exhibit severe underfitting (high bias) when applied to higher-degree datasets—for instance, Linear Regression achieves a Test $R^2$ of only `0.0283` (Test MSE = `364.7743`) on Dataset 4, whereas matching the polynomial degree to the data structure reduces Test MSE to the irreducible noise floor (`~4.0093`, Test $R^2$ = `0.9893`).
- **Variance Stability on Large Sample Sizes:** Because each synthetic dataset contained a large sample size ($N = 10,000$) with a single feature, higher-degree models (Degrees 5 and 10) maintained tight alignment between Training MSE (`3.9898`), Testing MSE (`4.0093`), and 5-Fold CV MSE (`4.0049`) without suffering from severe overfitting.
- **Selecting Optimal Complexity:** Adding polynomial degrees beyond the true underlying data complexity (e.g., using Degree 10 on Dataset 1 or Dataset 2) yields zero improvement in Test MSE or $R^2$ while slightly widening the train-test MSE gap and wasting computational resources.

---

## How to Run the Notebooks

1. Install the required Python dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn scipy statsmodels yellowbrick plotly
   ```
2. Download the respective datasets from the UCI Machine Learning Repository links provided in **Sections 1–3** (or run the NumPy data generation script from the Final Project notebook for **Section 4**) and place the `.txt` / `.csv` files in the root directory.
3. Launch Jupyter Notebook or open the `.ipynb` files in VS Code / Google Colab:
   ```bash
   jupyter notebook
   ```

---

## Author
- **Jessica Mae T. Lanuzo** — BSCS