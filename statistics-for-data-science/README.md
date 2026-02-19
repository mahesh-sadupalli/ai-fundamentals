# Statistics for Data Science

> A structured guide to the statistical concepts every data scientist needs to know.

---

## Table of Contents

- [Part I: Foundations](#part-i-foundations)
- [Part II: IQR, Covariance & Correlation](#part-ii-iqr-covariance--correlation)
- [Part III: Permutations, Combinations & Probability](#part-iii-permutations-combinations--probability)
- [Part IV: Gaussian/Normal Distribution](#part-iv-gaussiannormal-distribution)

---

## Part I: Foundations

### What is Statistics?

Statistics is the science of collecting, analyzing, and interpreting data to make informed decisions under uncertainty. It provides the mathematical backbone for everything from A/B testing to training deep neural networks.

### Types of Statistics

**1. Descriptive Statistics**
- Focuses on summarizing and describing datasets
- Uses measures such as mean, median, and variance to capture the shape and spread of data

**2. Inferential Statistics**
- Goes beyond the observed data to draw conclusions about larger populations
- Relies on sampling techniques and probability theory to generalize findings

### Types of Data

**Categorical**
- **Nominal:** Categories with no inherent order (e.g., blood group, country)
- **Ordinal:** Categories with a meaningful ranking (e.g., education level, customer satisfaction rating)

**Numerical**
- **Discrete:** Countable values with gaps between them (e.g., number of students, number of clicks)
- **Continuous:** Values that can take any number within a range (e.g., height, temperature, time)

**Temporal**
- **Time Series:** Sequential observations for a single entity over time (e.g., stock prices, daily temperature)
- **Cross-sectional:** A snapshot of multiple entities at one point in time (e.g., a census, a single survey)
- **Panel:** Multiple entities tracked over multiple time periods (e.g., GDP of several countries measured yearly)

### Scale of Measurement

| Scale | Description | Data Type | Examples |
|-------|-------------|-----------|----------|
| **Nominal** | Named categories, no ordering | Categorical | Name, Blood group |
| **Ordinal** | Named + ordered categories | Categorical | Ranking, Rating |
| **Interval** | Ordered with equal spacing, but no true zero | Numerical | Temperature (°C), GPA |
| **Ratio** | Interval scale with a true zero, allowing ratios | Numerical | Age, Height, Weight |

### Central Tendency

**Mean (μ)**

The arithmetic average — sum of all values divided by the count.

```
μ = (1/n) Σ xᵢ
```

*ML Application:* Mean centering is a prerequisite for PCA. Subtracting the mean from each feature ensures that the principal components capture true variance rather than being skewed by offset.

**Median**

The middle value when data is sorted. For even-sized datasets, it's the average of the two central values.

**Mode**

The most frequently occurring value in a dataset. Particularly useful for categorical data.

*ML Application:* The mean is sensitive to outliers. When extreme values are present, the median provides a more robust measure of central tendency — this is why robust scaling techniques use the median instead of the mean.

### Measures of Dispersion

**Variance**

Quantifies how far individual data points deviate from the mean, on average.

```
Var(X) = (1/n) Σ (xᵢ - μ)²
```

*ML Application:* Variance plays a central role in feature selection and PCA. Features with near-zero variance carry almost no information and can often be dropped. In PCA, the algorithm explicitly seeks directions of maximum variance.

**Standard Deviation (σ)**

The square root of variance, bringing the measure back into the original units of the data.

```
σ = √Var(X)
```

*ML Application:* Standard deviation is the foundation of Z-score normalization (standardization). By dividing by σ, we rescale features to a common scale — this is exactly what `StandardScaler()` does in scikit-learn.

**Interquartile Range (IQR)**

The spread of the middle 50% of the data, calculated as the difference between the 75th percentile (Q3) and the 25th percentile (Q1).

```
IQR = Q3 - Q1
```

**Outlier Detection with IQR Fences:**
- Lower Bound = Q1 − 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR
- Any point falling outside these bounds is flagged as an outlier

*ML Application:* IQR-based fencing is one of the most widely used techniques for outlier detection in data preprocessing. Because it depends only on the central 50% of data, it is naturally resistant to extreme values.

---

## Part II: IQR, Covariance & Correlation

### Covariance

Covariance measures the directional relationship between two variables. It tells us whether two features tend to move together or in opposite directions.

```
Cov(X, Y) = (1/n) Σ (xᵢ - μₓ)(yᵢ - μᵧ)
```

- **Positive covariance:** Both variables tend to increase or decrease together
- **Negative covariance:** As one variable increases, the other tends to decrease

> **Note:** When working with a sample (rather than the full population), use (n−1) in the denominator to obtain an unbiased estimate — this is known as Bessel's correction.

### Covariance Matrix

For a dataset with multiple features, the covariance matrix Σ captures all pairwise relationships:

```
Σ = | Var(X₁)      Cov(X₁,X₂)   ...  |
    | Cov(X₂,X₁)  Var(X₂)       ...  |
    | ...          ...           ...  |
```

- **Diagonal:** Variance of each feature
- **Off-diagonal:** Covariance between each pair of features
- The matrix is always **symmetric** (Cov(X,Y) = Cov(Y,X))

### Correlation (Pearson)

Correlation is the normalized version of covariance — it measures both the strength and direction of a linear relationship, but on a fixed scale from −1 to +1. Unlike covariance, it is **scale independent**.

```
ρ(X,Y) = Cov(X,Y) / (σₓ · σᵧ)
```

| Value | Interpretation |
|-------|----------------|
| **+1** | Perfect positive linear relationship |
| **0** | No linear relationship |
| **-1** | Perfect negative linear relationship |

### Why Covariance & Correlation Matter

**Data Exploration & Insight**
1. Correlation provides a standardized strength measure (−1 to +1), making it the go-to tool for comparing relationships across features with different scales
2. Highly correlated features are redundant — they inflate coefficient variance and can destabilize models (multicollinearity)
3. Understanding feature interactions guides feature engineering decisions

**Model Optimization**
1. The covariance matrix is the mathematical foundation of PCA — eigendecomposition of Σ yields the principal components
2. Identifying and removing correlated features reduces dimensionality
3. Removing redundant information improves model stability and reduces overfitting

---

## Part III: Permutations, Combinations & Probability

### Permutations

Arrangements of objects where **order matters**. Think of it as: how many different sequences can you create?

> ABC ≠ BAC ≠ CAB — each arrangement is distinct

**Without repetition:**
```
P(n, r) = n! / (n - r)!
```

**With repetition (each position has n choices):**
```
P = nʳ
```

*Applications:* Password generation, ranking systems, seating arrangements

### Combinations

Selections of objects where **order does not matter**. Think of it as: how many different groups can you form?

> {A, B, C} = {B, A, C} = {C, B, A} — all the same group

```
C(n, r) = n! / [r! × (n - r)!]
```

*Applications:* Team formation, feature subset selection, sampling

### Key Facts

**Factorial:**
- 0! = 1
- 1! = 1

**Combination Properties:**
- C(n, 0) = 1
- C(n, n) = 1
- C(n, r) = C(n, n − r)

### Probability

Probability quantifies how likely an event is to occur, expressed as a value between 0 (impossible) and 1 (certain).

```
P(E) = Number of favorable outcomes / Total number of possible outcomes
```

*Applications:* Model confidence scores, risk estimation, A/B testing, Bayesian inference

### Probability Rules

| Rule | Formula | When to Use |
|------|---------|-------------|
| **Addition (Union)** | P(A ∪ B) = P(A) + P(B) − P(A ∩ B) | Probability of either event occurring |
| **Multiplication (Independent)** | P(A ∩ B) = P(A) × P(B) | Probability of both events occurring together |
| **Complement** | P(Aᶜ) = 1 − P(A) | Probability of an event NOT occurring |

---

## Part IV: Gaussian/Normal Distribution

### The King of Distributions

The normal distribution is named after the German mathematician **Carl Friedrich Gauss**. In 1809, while studying measurement errors in astronomy and land surveying, he formulated the **Law of Errors** — demonstrating that for symmetric measurement errors, the most probable value is the arithmetic mean.

The term "normal" doesn't mean ordinary. Statistician Francis Galton and others used it to describe the **standard** or **ideal** pattern that naturally occurring data tends to follow. It became the benchmark against which all other distributions were compared.

### Definition

A normal (Gaussian) distribution is a continuous probability distribution for a real-valued random variable, fully defined by just two parameters — the **mean (μ)** and the **variance (σ²)**.

```
X ~ N(μ, σ²)
```

**The Two Parameters:**
- **μ (Mean):** Controls the location — shifting μ moves the entire bell curve left or right
- **σ (Standard Deviation):** Controls the spread — a small σ produces a tall, narrow peak; a large σ produces a flat, wide curve

**Shape Properties:**
- Perfectly symmetric around the mean
- Asymptotic — the tails approach but never touch the x-axis

### The Probability Density Function (PDF)

```
f(x) = (1 / σ√2π) × e^(-½ ((x - μ) / σ)²)
```

Breaking it down:
- **The Exponential Decay (e^(−x²)):** As x moves away from the mean, the exponent becomes increasingly negative, pushing the curve toward zero — this creates the characteristic tails
- **The Square ((...)²):** Whether x is above or below the mean, the result is the same — this is what creates the perfect symmetry
- **The Scaling Factor (1/σ√2π):** This normalization constant ensures the total area under the curve integrates to exactly 1 (100% probability)

### The 68-95-99.7 Rule (Empirical Rule)

The golden rule of the normal distribution — if your data follows a Gaussian, you can predict exactly where values will fall:

| Range | Data Coverage | Interpretation |
|-------|---------------|----------------|
| μ ± 1σ | **68%** | The majority — most observations are average |
| μ ± 2σ | **95%** | Uncommon but still within normal range |
| μ ± 3σ | **99.7%** | Nearly everything — only 0.3% lies beyond |

### Outliers

Anything falling outside **±3σ** belongs to the remaining 0.3%. In data science, these points are typically flagged as **outliers or anomalies** and warrant further investigation — they could be measurement errors, data entry mistakes, or genuinely rare events.

### The Standard Normal Distribution (Z-Score)

When comparing values from different datasets (with different means and spreads), we standardize them by converting to the **Standard Normal Distribution** — a special Gaussian with mean = 0 and standard deviation = 1.

```
Z = (x - μ) / σ
```

Standardization serves two critical purposes:
1. **Fair comparison:** It puts all features on the same scale, letting us compare "apples to oranges"
2. **Algorithm performance:** Gradient-based optimization algorithms converge significantly faster when features are standardized

> This is exactly why `StandardScaler()` is one of the most commonly used preprocessing imports in Python's scikit-learn.

---

## Progress

- [x] Part I: Foundations
- [x] Part II: IQR, Covariance & Correlation
- [x] Part III: Permutations, Combinations & Probability
- [x] Part IV: Gaussian/Normal Distribution
- [ ] Part V: Coming soon...
