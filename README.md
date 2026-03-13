# AI Fundamentals

Machine learning is inherently visual — loss surfaces have curvature, distributions have shape, optimizers trace paths through high-dimensional space. Yet most learning resources reduce these rich geometric ideas to static equations on a page. AI Fundamentals takes a different approach: every concept is an interactive visualization you can manipulate, rotate, and experiment with in real time.

Understanding a gradient descent update rule on paper is one thing. Watching four different optimizers race across a 3D loss surface — adjusting learning rates and momentum on the fly — builds the kind of intuition that equations alone can't provide. Interactive visualizations turn abstract math into something tangible, letting you see *why* Adam converges faster, *how* a noise schedule destroys structure, and *what* a Beta distribution actually looks like when you drag its parameters.

This project is a growing collection of these interactive explorations, each built as a standalone page with zero dependencies — just HTML, CSS, and JavaScript.

**[Explore the course](https://mahesh-sadupalli.github.io/ai-fundamentals/)**

---

## Lessons

### [Function Explorer — Mathematical Foundations](https://mahesh-sadupalli.github.io/ai-fundamentals/function-explorer/)

Interactive tool for exploring mathematical functions commonly used in machine learning. Covers linear, quadratic, exponential, logarithmic, and trigonometric functions alongside ML activation functions like Sigmoid, Tanh, ReLU, Leaky ReLU, and Softmax. Each function includes real-time parameter controls, domain/range info, derivative visualization, and practical ML use cases.

### [Probability Distributions](https://mahesh-sadupalli.github.io/ai-fundamentals/distributions/)

Visual, interactive guide to the probability distributions that underpin statistics and machine learning. Includes Normal, Binomial, Poisson, Uniform, Exponential, Beta, Gamma, Log-Normal, Chi-Squared, and Student's t distributions. Each distribution features adjustable parameters, annotated PDF/PMF curves, real-world examples, and key statistical properties rendered with LaTeX.

### [Gradient Descent & Optimization](https://mahesh-sadupalli.github.io/ai-fundamentals/gradient-descent/)

Step-by-step visualization of how neural networks learn. Compare optimization algorithms — Vanilla SGD, Momentum, RMSProp, and Adam — side by side on interactive 3D loss surface contour plots. Adjust learning rate, momentum, and other hyperparameters in real time to see how each optimizer navigates toward the minimum.

### [Diffusion Models](https://mahesh-sadupalli.github.io/ai-fundamentals/diffusion-models/)

Interactive walkthrough of how diffusion-based generative AI works. Step through the forward noising process and reverse denoising process to see how models like Stable Diffusion and DALL-E transform random noise into structured data. Includes visual explanations of the noise schedule, the U-Net architecture, and the training objective.

---

## License

[MIT](LICENSE)
