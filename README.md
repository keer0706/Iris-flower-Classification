# Iris Flower Classification

## Overview

This project uses the Iris dataset to build and compare machine learning classification models for predicting the species of an Iris flower from its sepal and petal measurements.

The notebook covers data loading, data inspection, duplicate removal, exploratory data analysis (EDA), correlation analysis, label encoding, feature scaling, model training, and model evaluation.

## Dataset

The Iris dataset is fetched from the UCI Machine Learning Repository using `ucimlrepo`.

The dataset initially contains **150 observations** and **5 columns**:

- 4 numerical features
- 1 target column (`class`)

### Features

| Feature | Description |
|---|---|
| `sepal length` | Length of the sepal |
| `sepal width` | Width of the sepal |
| `petal length` | Length of the petal |
| `petal width` | Width of the petal |
| `class` | Iris species / target variable |

The three classes are:

- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

## Data Preprocessing

### Duplicate Removal

The dataset initially contained **3 duplicate rows**.

These duplicates were removed using:

```python
df_combined.drop_duplicates(inplace=True)
```

After removal, the dataset contained **147 observations**.

### Missing Values

The notebook checks for missing values using:

```python
df_combined.isnull().sum()
```

No missing values were found after preprocessing.

### Label Encoding

The categorical target variable was converted into numerical labels using `LabelEncoder`.

The resulting classes are encoded as:

- `Iris-setosa` → `0`
- `Iris-versicolor` → `1`
- `Iris-virginica` → `2`

## Exploratory Data Analysis

The notebook includes the following visualizations:

- Sepal length vs. sepal width scatter plot
- Petal length vs. petal width scatter plot
- Sepal length histogram
- Petal length histogram
- Boxplots for sepal length, sepal width, petal length, and petal width
- Feature correlation heatmap

### Correlation Analysis

The correlation analysis shows strong relationships between some of the numerical features.

Notable correlations from the notebook include:

- Petal length and petal width: **0.962**
- Sepal length and petal length: **0.871**
- Sepal length and petal width: **0.817**

The scatter plots also show that petal measurements provide clearer separation between the Iris classes than the sepal measurements.

## Train-Test Split

The dataset is divided into training and testing sets using an **80:20 split**:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Feature Scaling

`StandardScaler` is applied to the features used by:

- Logistic Regression
- K-Nearest Neighbors (KNN)

The Decision Tree Classifier is trained using the original feature values.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Machine Learning Models

Three classification algorithms are trained and compared:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**

## Evaluation Metrics

The models are evaluated using:

- **Accuracy**
- **Weighted Precision**
- **Confusion Matrix**

### Results

The results recorded in the notebook are:

| Model | Accuracy | Weighted Precision |
|---|---:|---:|
| Logistic Regression | 93.33% | 93.33% |
| **Decision Tree Classifier** | **96.67%** | **97.00%** |
| K-Nearest Neighbors | 93.33% | 93.33% |

### Confusion Matrices

**Logistic Regression**

```text
[[11, 0, 0],
 [ 0, 9, 1],
 [ 0, 1, 8]]
```

**Decision Tree Classifier**

```text
[[11, 0, 0],
 [ 0, 9, 1],
 [ 0, 0, 9]]
```

**K-Nearest Neighbors**

```text
[[11, 0, 0],
 [ 0, 9, 1],
 [ 0, 1, 8]]
```

## Best Performing Model

Among the three models tested, the **Decision Tree Classifier** achieved the best performance on the test set.

- **Accuracy:** 96.67%
- **Weighted Precision:** 97.00%

Based on the results recorded in the notebook, the Decision Tree Classifier is the best-performing model for this particular train-test split.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- `ucimlrepo`
- Google Colab / Jupyter Notebook

## Project Structure

```text
Iris-Flower-Classification/
│
├── Iris_Flower_Classification.ipynb
├── README.md
├── scatter_sepal_length_vs_sepal_width.png
├── scatter_petal_length_vs_petal_width.png
├── hist_sepal_length.png
├── hist_petal_length.png
├── boxplot_sepal_width.png
├── boxplot_petal_width.png
├── boxplot_petal_length.png
├── sepal_length_boxplot.png
└── heatmap.png
```

## How to Run

1. Open `Iris_Flower_Classification.ipynb` in Google Colab or Jupyter Notebook.
2. Install the required dataset package if necessary:

```python
%pip install ucimlrepo
```

3. Run the notebook cells from top to bottom.
4. The notebook fetches the Iris dataset from the UCI Machine Learning Repository.
5. The final model comparison table displays the performance of all three classifiers.

## Conclusion

This project demonstrates a basic end-to-end classification workflow using the Iris dataset. After preprocessing and exploratory analysis, three classification models were trained and evaluated. The **Decision Tree Classifier** achieved the highest test accuracy and weighted precision among the models tested, with an accuracy of **96.67%** and weighted precision of **97.00%**.
