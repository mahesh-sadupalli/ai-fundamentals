# Statistics for Data Science

A comprehensive visual guide to statistical foundations essential for data science and machine learning.

> Content created at [Black Forest Labs](https://www.blackforesttlabs.com/)

---

## Table of Contents

- [Part I: Foundations](#part-i-foundations)
- [Part II: IQR, Covariance & Correlation](#part-ii-iqr-covariance--correlation)
- [Part III: Permutations, Combinations & Probability](#part-iii-permutations-combinations--probability)
- [Part IV: Gaussian/Normal Distribution](#part-iv-gaussiannormal-distribution)

---

## Part I: Foundations

<p align="center">
  <img src="images/part1-foundations/Statistics1-1.png" width="400"/>
</p>

### What is Statistics?

Statistics is the science of collecting, analyzing, and interpreting data to make decisions under uncertainty.

<p align="center">
  <img src="images/part1-foundations/Statistics1-2.png" width="400"/>
</p>

### Types of Statistics

| Type | Description |
|------|-------------|
| **Descriptive Statistics** | Summarizes and describes data using measures like mean, median, variance |
| **Inferential Statistics** | Draws conclusions about a population based on sample data and probability |

<p align="center">
  <img src="images/part1-foundations/Statistics1-3.png" width="400"/>
</p>

### Types of Data

**Categorical:**
- **Nominal** — No order (e.g., blood group, name)
- **Ordinal** — Ordered categories (e.g., ranking, rating)

**Numerical:**
- **Discrete** — Countable values (e.g., number of students, clicks)
- **Continuous** — Infinite possible values (e.g., height, temperature, time)

<p align="center">
  <img src="images/part1-foundations/Statistics1-4.png" width="400"/>
</p>

**Temporal Data:**
- **Time Series** — Observations over time for a single entity (stock prices, daily temperature)
- **Cross-sectional** — Multiple entities at a single time point (survey data, census)
- **Panel** — Multiple entities tracked over time (countries' GDP over years)

<p align="center">
  <img src="images/part1-foundations/Statistics1-5.png" width="400"/>
</p>

### Scale of Measurement

| Scale | Properties | Data Type | Examples |
|-------|-----------|-----------|----------|
| **Nominal** | Named categories | Categorical | Name, Blood group |
| **Ordinal** | Named + ordered categories | Categorical | Ranking, Rating |
| **Interval** | No absolute zero, difference exists | Numerical | Temperature, GPA |
| **Ratio** | True zero exists, ratios possible | Numerical | Age, Height, Weight |

<p align="center">
  <img src="images/part1-foundations/Statistics1-6.png" width="400"/>
</p>

### Central Tendency

#### Mean (μ)

The average value of the data.

$$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$$

**Application:** Mean is used to center features before applying PCA. Centering ensures principal components capture variance correctly.

<p align="center">
  <img src="images/part1-foundations/Statistics1-7.png" width="400"/>
</p>

#### Median & Mode

- **Median** — Middle value after sorting the data
- **Mode** — Most frequently occurring value

**Application:** Since the mean is sensitive to outliers, the median is preferred for robust statistics when outliers exist.

<p align="center">
  <img src="images/part1-foundations/Statistics1-8.png" width="400"/>
</p>

### Measures of Dispersion

#### Variance

Measures how far data points spread from the mean.

$$\text{Var}(X) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$$

**Application:** Variance is used in feature selection and helps in finding principal components in PCA.

<p align="center">
  <img src="images/part1-foundations/Statistics1-9.png" width="400"/>
</p>

#### Standard Deviation (σ)

Square root of variance, expressed in original data units.

$$\sigma = \sqrt{\text{Var}(X)}$$

**Application:** Used in feature scaling, specifically standardization (Z-score normalization), to rescale features so they share a common scale.

<p align="center">
  <img src="images/part1-foundations/Statistics1-10.png" width="400"/>
</p>

#### Interquartile Range (IQR)

Spread of the middle 50% of data.

$$\text{IQR} = Q_3 - Q_1$$

**Outlier Detection using IQR Fences:**
- Lower Bound = Q1 − 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR
- Any data point outside these bounds is considered an **outlier**

**Application:** Primary tool for outlier detection and robust data preprocessing because it focuses on the central 50% of data, making it naturally resistant to extreme values.

<p align="center">
  <img src="images/part1-foundations/Statistics1-12.png" width="400"/>
</p>

---

## Part II: IQR, Covariance & Correlation

<p align="center">
  <img src="images/part2-covariance-correlation/Statistics2-1.png" width="400"/>
</p>

### IQR (Recap)

IQR is a measure of statistical dispersion representing the spread of the middle 50% of a data set, calculated as the difference between the upper quartile Q3 and lower quartile Q1, by ordering data from least to greatest.

$$\text{IQR} = Q_3 - Q_1$$

<p align="center">
  <img src="images/part2-covariance-correlation/Statistics2-3.png" width="400"/>
</p>

### Covariance

The covariance measures the relationship between two variables x and y.

$$\text{Cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_X)(y_i - \mu_Y)$$

The sign of the covariance tells a story:
- **Positive values:** Both variables tend to increase or decrease together
- **Negative values:** As one variable increases, the other tends to decrease

> **Note:** Use denominator (n-1) if you are working with a sample rather than an entire population to ensure the estimate is unbiased.

<p align="center">
  <img src="images/part2-covariance-correlation/Statistics2-5.png" width="400"/>
</p>

### Covariance Matrix

$$\Sigma = \begin{bmatrix} \text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots \\ \text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

- **Diagonal elements:** Variances of features
- **Off-diagonal elements:** Covariances between features
- **Matrix is symmetric**

<p align="center">
  <img src="images/part2-covariance-correlation/Statistics2-6.png" width="400"/>
</p>

### Correlation (Pearson)

Correlation measures the strength and direction of a linear relationship between two variables. It is the normalized form of covariance and is **scale independent**.

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

**Interpretation:**
| Value | Meaning |
|-------|---------|
| **+1** | Perfect positive linear relationship |
| **0** | No linear relationship |
| **-1** | Perfect negative linear relationship |

<p align="center">
  <img src="images/part2-covariance-correlation/Statistics2-8.png" width="400"/>
</p>

### Why Covariance & Correlation Matter

**Data Exploration & Insight:**
1. While covariance tells us the direction of a relationship, **correlation** provides a standardized strength (−1 to +1), making it the primary tool for comparing relationships across different scales
2. Spot redundant features (high correlation) that can confuse models and inflate the variance of coefficients
3. Feature interactions

**Model Optimization:**
1. **Covariance matrices** form the mathematical bedrock for PCA, allowing you to transform features into a new coordinate system
2. Dimensionality reduction
3. Enhanced stability by reducing noise and redundancy

<p align="center">
  <img src="images/part2-covariance-correlation/Statistics2-9.png" width="400"/>
</p>

---

## Part III: Permutations, Combinations & Probability

<p align="center">
  <img src="images/part3-permutations-combinations-probability/Statistics3-1.png" width="400"/>
</p>

### Permutations

Arrangements of objects where **order matters**. The number of ways to arrange **r** objects chosen from **n** distinct objects.

> ABC ≠ BAC ≠ CAB

**Formulas:**

| Type | Formula |
|------|---------|
| Without repetition | P(n, r) = n! / (n − r)! |
| With repetition | P = n^r |

**Applications:** Passwords, Rankings, Seating orders

<p align="center">
  <img src="images/part3-permutations-combinations-probability/Statistics3-3.png" width="400"/>
</p>

### Combinations

Selections of objects where **order doesn't matter**. The number of ways to choose **r** objects from **n** distinct objects.

> {A, B, C} = {B, A, C} = {C, B, A}

$$\binom{n}{r} = \frac{n!}{r!(n-r)!}$$

**Applications:** Teams, Feature selection, Sampling

<p align="center">
  <img src="images/part3-permutations-combinations-probability/Statistics3-5.png" width="400"/>
</p>

### Key Facts

**Factorial:**
- 0! = 1
- 1! = 1

**Combination Properties:**
- C(n, 0) = 1
- C(n, n) = 1
- C(n, r) = C(n, n − r)

<p align="center">
  <img src="images/part3-permutations-combinations-probability/Statistics3-8.png" width="400"/>
</p>

### Probability

The measure of how likely an event is to occur, expressed as a number between 0 and 1.

$$P(E) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}$$

**Applications:** Model confidence, Risk estimation, A/B testing

<p align="center">
  <img src="images/part3-permutations-combinations-probability/Statistics3-6.png" width="400"/>
</p>

### Probability Rules

| Rule | Formula |
|------|---------|
| **Addition (Union)** | P(A ∪ B) = P(A) + P(B) − P(A ∩ B) |
| **Multiplication (Independent)** | P(A ∩ B) = P(A) × P(B) |
| **Complement** | P(A^c) = 1 − P(A) |

<p align="center">
  <img src="images/part3-permutations-combinations-probability/Statistics3-7.png" width="400"/>
</p>

---

## Part IV: Gaussian/Normal Distribution

<p align="center">
  <img src="images/part4-gaussian-distribution/Statistics4-1.png" width="400"/>
</p>

### The King of Distributions

Normal distribution is named after German mathematician **Carl Friedrich Gauss**. In 1809, while studying measurement errors in astronomy and land surveying, he formulated the **"Law of Errors"**, demonstrating that for symmetric measurement errors, the most probable value in complex measurements is the **arithmetic mean**.

It wasn't named "normal" because it's ordinary — it's because it was the **standard** pattern that data was expected to follow. It became the benchmark against which other data was measured.

<p align="center">
  <img src="images/part4-gaussian-distribution/Statistics4-2.png" width="400"/>
</p>

### Definition

A normal distribution or Gaussian distribution is a type of **continuous probability distribution** for a real-valued random variable defined completely by two parameters: **mean (μ) and variance (σ²)**.

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

**The Two Parameters:**
- **μ (Mean):** Controls the location. Shifting mean moves the entire curve left or right
- **σ (Std Dev):** Controls the scale. A small sigma gives a tall, thin peak; a large sigma gives a flat, wide curve
- **The Shape:** It is asymptotic — the tails get closer to the x-axis but never touch it

<p align="center">
  <img src="images/part4-gaussian-distribution/Statistics4-5.png" width="400"/>
</p>

### The Probability Density Function (PDF)

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

- **The "Decay" (e^(-x²)):** The exponential part creates the slope. As x moves away from the mean, the exponent becomes a large negative number, pushing the curve down to zero (creating the tails)
- **The Square (...)²:** Because of the square, it doesn't matter if x is above or below the mean — the result is the same. This creates the **symmetry**
- **The Scaling Factor (1/σ√2π):** This constant ensures the total area under the curve equals exactly 1 (or 100% probability)

<p align="center">
  <img src="images/part4-gaussian-distribution/Statistics4-6.png" width="400"/>
</p>

### The 68-95-99.7 Rule (Empirical Rule)

The "Golden Rule" of the Normal Distribution. If your data is normal, you can predict exactly where it will land:

| Range | Coverage | Interpretation |
|-------|----------|----------------|
| μ ± 1σ | **68%** | Most people are average |
| μ ± 2σ | **95%** | Uncommon, but normal |
| μ ± 3σ | **99.7%** | Almost everything fits here |

<p align="center">
  <img src="images/part4-gaussian-distribution/Statistics4-8.png" width="400"/>
</p>

### Outliers

Anything outside **±3σ** (the remaining 0.3%) — in data science, we often flag these points as **Outliers or Anomalies**.

<p align="center">
  <img src="images/part4-gaussian-distribution/Statistics4-9.png" width="400"/>
</p>

### The Standard Normal Distribution (Z-Score)

What if we want to compare two different datasets? We convert them to the **Standard Normal Distribution**. This helps gradient-based algorithms converge faster and lets us compare "apples to oranges" by putting everything on the same scale.

**Z Score** is a special case of the Gaussian distribution where:
- **Mean = 0**
- **Std Dev = 1**

> "Standardization is the universal translator of statistics. By converting raw data into Z-scores, we center everything at 0 with a spread of 1. This is why `StandardScaler()` is one of the most used imports in Python!"

<p align="center">
  <img src="images/part4-gaussian-distribution/Statistics4-10.png" width="400"/>
</p>

---

## Progress

- [x] Part I: Foundations
- [x] Part II: IQR, Covariance & Correlation
- [x] Part III: Permutations, Combinations & Probability
- [x] Part IV: Gaussian/Normal Distribution
- [ ] Part V: Coming soon...
