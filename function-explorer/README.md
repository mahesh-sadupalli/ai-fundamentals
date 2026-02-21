# Function Explorer

An interactive, browser-based visualization tool for exploring mathematical functions and their properties in real-time.

**[Launch Function Explorer →](https://mahesh-sadupalli.github.io/function-explorer/)**

---

## Table of Contents

1. [Linear Function](#1-linear-function)
2. [Exponential Function](#2-exponential-function)
3. [Logarithmic Function](#3-logarithmic-function)

---

## 1. Linear Function

A linear function describes a straight-line relationship between an input and an output. A constant change in input always produces a constant change in output.

```
f(x) = mx + b
```

- **m (Slope)** — the rate of change; how much y changes per unit change in x
- **b (Y-Intercept)** — the value of f(x) when x = 0, i.e. where the line crosses the y-axis

### Properties

| Property | Description | Formula |
|----------|-------------|---------|
| Slope | Rate of change | m = (y₂ - y₁) / (x₂ - x₁) |
| Y-Intercept | Starting value | b = y - mx |
| Point-Slope Form | Line from a known point | y - y₁ = m(x - x₁) |
| Derivative | Constant gradient | df(x)/dx = m |

> The y-intercept is the value of the function at x = 0, i.e. b = f(0). The expression b = y - mx is used when computing the intercept from a known point (x, y) on the line.

### Properties and ML Applications

**1. Constant Slope (Rate of Change)**

The slope remains the same everywhere on the line — the ratio Δy/Δx is constant.

- **Linear Regression**: The slope coefficient tells us how much the expected output changes per unit increase in input
- **Gradient Descent**: Constant derivatives make optimization predictable and efficient

**2. No Local Minima or Maxima (when m ≠ 0)**

A non-constant linear function has no turning points — it is strictly monotonic.

- **Optimization Landscapes**: Unlike sigmoid or tanh, there are no local optima to get stuck in
- **Gradient Flow**: Gradients never vanish since the derivative is constant
- **Why Activation Functions Matter**: Neural networks need non-linear activations to create complex decision boundaries — linear functions alone cannot model them

**3. Additivity: f(x₁ + x₂) = f(x₁) + f(x₂)**

The output of a sum equals the sum of the individual outputs (when b = 0).

- **Weighted Sum in Neural Networks**: Inputs are combined linearly as z = Σ wᵢxᵢ + b
- **Feature Engineering**: The total effect equals the sum of individual feature effects

**4. Homogeneity (Scaling): f(αx) = αf(x) (when b = 0)**

Scaling the input by a factor α scales the output by the same factor.

- **Feature Normalization**: Scaling inputs preserves linear relationships
- **Regularization (L1/L2)**: Penalty scales proportionally with weights

**5. Continuous and Differentiable Everywhere**

The function has no gaps, jumps, or sharp corners. The derivative f'(x) = m is constant.

- **Optimization Algorithms**: Smooth gradients enable gradient descent convergence
- **Output Layer (Regression)**: Used to predict unbounded continuous values

**6. Unbounded Range: (-∞, +∞) when m ≠ 0**

The output can take any real number value. If m = 0, the range is just {b}.

- **Regression Output**: The model can predict any real number
- **Linear Regression Target**: No artificial limits on predictions

---

## 2. Exponential Function

An exponential function models growth or decay. In the natural exponential case, the rate of change is proportional to the current value.

```
General Exponential:   f(x) = aˣ
Natural Exponential:   f(x) = eˣ
```

- **a (The Base)** — must be a positive number (a > 0) and not equal to 1. When a > 1, the function represents growth. When 0 < a < 1, it represents decay
- **x (The Exponent)** — the independent variable, often representing time or steps

### Conversion

Any exponential can be rewritten in terms of the natural exponential:

```
aˣ = e^(x · ln(a))
```

> The natural exponential uses the mathematical constant e ≈ 2.71828 (Euler's number).

### Properties

| Property | Description | Formula |
|----------|-------------|---------|
| Domain | All real numbers | x ∈ ℝ |
| Range | Positive numbers only | y > 0 |
| Growth Rate | Proportional to value | f'(x) = f(x) for eˣ |
| Base Cases | Powers of base | a⁰ = 1, a¹ = a |

### Properties and ML Applications

**1. Self-Derivative (Natural Exponential)**

The natural exponential eˣ is unique in that its derivative equals itself: d/dx(eˣ) = eˣ.

- **Gradient-Based Optimization**: Derivatives remain exponential, simplifying backpropagation
- **Activation Functions**: Appears naturally in softmax and sigmoid derivatives
- **Stability in Training**: Predictable gradient behavior

**2. No Local Minima or Maxima (Strictly Monotonic)**

The exponential function is always increasing (for a > 1) or always decreasing (for 0 < a < 1). It never changes direction.

- **Monotonic Activations**: Functions like exponential in attention mechanisms preserve ordering
- **Optimization Landscapes**: Exponential terms often contribute to convex behavior, reducing the risk of local minima

**3. Always Positive Range**

The output of eˣ is always greater than zero for any input x.

- **Probability Modeling**: Ensures non-negative outputs before normalization
- **Softmax Function**: Converts logits to positive values
- **Log-Likelihood**: Exponentiation ensures valid positive likelihoods before normalization

**4. Exponential Growth/Decay**

When a > 1 the function grows rapidly; when 0 < a < 1 it decays toward zero.

- **Learning Rate Decay**: Reduces learning rate over epochs
- **Gradient Explosion/Vanishing**: Understanding exponential behavior helps explain gradient explosion and vanishing in deep networks

**5. Unbounded Range and Inverse with Logarithm**

As x → ∞, eˣ → ∞. The range is (0, ∞). The logarithm is the inverse operation.

- **Cross-Entropy Loss**: -ln(ŷ) assigns exponentially increasing penalty to confident wrong predictions
- **Overflow Prevention**: Why we use log-softmax instead of raw softmax

### Exponential Laws (Natural Exponential)

```
1.  e^(x+y) = eˣ · eʸ
2.  ln(eˣ) = x
3.  e^(ln(x)) = x    for x > 0
4.  e^(x-y) = eˣ / eʸ
5.  (eˣ)ʸ = e^(xy)
```

---

## 3. Logarithmic Function

A logarithmic function is the inverse of an exponential function. It answers the question: "To what power must we raise the base to get x?"

```
y = log_b(x)    ⟺    b^y = x
```

- **b (The Base)** — the base of the exponential
- **y (The Logarithm)** — the exponent, i.e. the "answer"
- **x (The Argument)** — the value you want to reach

### Types of Logarithmic Functions

| Type | Notation | Base | Usage |
|------|----------|------|-------|
| Common Log | log(x) | 10 | Earthquake magnitude scales, pH levels |
| Natural Log | ln(x) | e ≈ 2.718 | Physics, biology, finance, continuous growth |

### Domain and Range

- **Domain**: x > 0 (strictly positive real numbers)
- **Range**: All real numbers (-∞, +∞)

> Why domain matters in ML: preventing log(0) errors. Many loss functions like cross-entropy use log internally, and passing zero or negative values causes numerical errors.

### Properties and ML Applications

**1. Monotonicity**

For base b > 1, the logarithm is a strictly increasing function: if x₁ < x₂, then log_b(x₁) < log_b(x₂).

- Preserves ordering in log-transformed features
- Used in ranking algorithms and gradient descent

**2. No Local Min/Max**

The logarithm is continuously increasing with no turning points. As x → 0⁺, log(x) → -∞. As x → ∞, log(x) → ∞.

- **Log Loss**: Unbounded penalty for confident wrong predictions
- Ensures strong gradients for misclassified samples

**3. Differentiability**

The logarithm is differentiable everywhere in its domain.

```
General base:    d/dx [log_b(x)] = 1 / (x · ln(b))
Natural log:     d/dx [ln(x)] = 1/x
```

- Smooth gradients in backpropagation
- Used in log-likelihood optimization
- Numerical stability in gradient computation

**4. Concavity**

The logarithm is a concave function — it curves downward. The second derivative is always negative:

```
d²/dx² [ln(x)] = -1/x²
```

- **Jensen's Inequality**: Used in probabilistic bounds
- **Diminishing Returns**: As x increases, the rate of gain keeps getting smaller — log transforms compress large values while spreading out small values

### Key Formulas

```
Power Rule:      log_b(xⁿ) = n · log_b(x)
Product Rule:    log_b(xy) = log_b(x) + log_b(y)
Quotient Rule:   log_b(x/y) = log_b(x) - log_b(y)
Change of Base:  log_b(x) = log_a(x) / log_a(b)
```

---

## License

[MIT](LICENSE)
