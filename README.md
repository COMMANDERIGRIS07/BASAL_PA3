# BASAL_PA3
2ECEC
# README — Programming Assignment 3 (PA3)

This assignment consists of **two separate problems**, each in its own Jupyter Notebook. Below are organized READMEs for each notebook: **Problem 1** and **Problem 2**. Each section explains how to run the notebook and gives a cell-by-cell breakdown of the code.

---

# Problem 1 — `BASAL_PA3 - PROBLEM 1.ipynb`

## Overview

This notebook solves **Problem 1** of PA3. It loads a dataset, performs data cleaning, computes averages, applies filters, and visualizes results using boxplots.

## Requirements

* Python 3.8+
* Jupyter Notebook or JupyterLab
* Libraries:

  ```bash
  pip install pandas numpy matplotlib seaborn openpyxl
  ```

## How to run

1. Place `BASAL_PA3 - PROBLEM 1.ipynb` and the dataset file (e.g., `board2.xlsx`) in the same folder.
2. Open the notebook:

   ```bash
   jupyter notebook "BASAL_PA3 - PROBLEM 1.ipynb"
   ```
3. Run cells top to bottom.

## Code explanation (cell by cell)

### 1. Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

* Loads data handling and visualization libraries.

### 2. Load dataset

```python
df = pd.read_excel("board2.xlsx")
```

* Reads Excel file into a pandas DataFrame.

### 3. Export to CSV

```python
df.to_csv("board2.csv", index=False)
```

* Saves dataset into CSV format for easier use.

### 4. Inspect data

```python
df.head()
df.info()
df.isnull().sum()
```

* Displays sample rows, data types, and missing values.

### 5. Filtering example (Visayas, low Math)

```python
Vis = df[(df['Hometown']=='Visayas') & (df['Math']<70)][['Name','Gender','Track','Math']]
```

* Creates a subset of students from Visayas with Math < 70.

### 6. Filtering example (Instrumentation, Luzon, high Electronics)

```python
Instru = df[(df['Track']=='Instrumentation') & (df['Hometown']=='Luzon') & (df['Electronics']>70)][['Name','GEAS','Electronics']]
```

* Filters students from Luzon in Instrumentation track with Electronics > 70.

### 7. Clean column names

```python
print(df.columns.tolist())
df.columns = df.columns.str.strip()
```

* Strips whitespace from column names.

### 8. Compute averages

```python
grade_cols = ['Math','GEAS','Electronics']
df['Average'] = df[grade_cols].mean(axis=1)
```

* Adds a new column `Average` with row-wise mean of scores.

### 9. Visualizations

```python
sns.boxplot(x='Track', y='Average', data=df)
sns.boxplot(x='Gender', y='Average', data=df)
sns.boxplot(x='Hometown', y='Average', data=df)
```

* Shows boxplots of average grades by Track, Gender, and Hometown.

### 10. Observations (Markdown)

* Explains conclusions: e.g., which group performs better, variability, and outliers.

---

# Problem 2 — `PA3 - PROBLEM 2.ipynb`

## Overview

This notebook solves **Problem 2** of PA3. It defines functions/algorithms (depending on the assignment question), applies them to input data, and outputs results in tabular/visual form.

## Requirements

* Python 3.8+
* Jupyter Notebook or JupyterLab
* Libraries:

  ```bash
  pip install pandas numpy matplotlib seaborn
  ```

## How to run

1. Place `PA3 - PROBLEM 2.ipynb` in the working directory.
2. Open the notebook:

   ```bash
   jupyter notebook "PA3 - PROBLEM 2.ipynb"
   ```
3. Run all cells sequentially.

## Code explanation (cell by cell)

### 1. Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

* Loads pandas, numpy, and plotting libraries.

### 2. Problem input

```python
# Example: input dataset or manually defined list/array
data = [ ... ]
```

* Defines the dataset or input values used in Problem 2.

### 3. Helper function(s)

```python
def compute_average(row):
    return np.mean(row)
```

* Example helper function: computes average.
* Each function should have docstring explaining parameters and outputs.

### 4. Main algorithm

```python
results = some_function(data)
```

* Executes the required algorithm for Problem 2.

### 5. Visualization / table output

```python
plt.plot(results)
plt.show()
```

* Produces plots summarizing results.

### 6. Observations (Markdown)

* Summarizes what the results show in relation to the problem statement.

---

# Troubleshooting

* **KeyError:** Check column names with `df.columns.tolist()` and strip whitespace.
* **TypeError:** Ensure numeric columns are converted with `pd.to_numeric(..., errors='coerce')`.
* **Missing library:** Install required packages with `pip install <library>`.

---

# Notes

* Each notebook is **self-contained** and should be graded separately.
* The explanations above follow a **cell-by-cell structure**: each block of code is paired with a clear explanation.
* Replace placeholder code (e.g., `...`) with the actual logic from your assignment if not already present.

---

*End of README for PA3 — Problems 1 & 2*
