# Bayesian Optimization — Project

This project demonstrates the implementation of **Bayesian Optimization**, a sample-efficient technique for optimizing expensive or unknown black-box functions.

Bayesian Optimization uses:
- A **surrogate model** (commonly a Gaussian Process) to approximate the objective function.
- An **acquisition function** to decide where to sample next (e.g., Expected Improvement, UCB).

---

## Key Features
- Gaussian Process regression for function approximation.
- Multiple acquisition functions to guide exploration vs. exploitation.
- Example scripts demonstrating optimization of non-trivial functions.
- Visualizations for understanding the optimization process.

---

## Example Use Case
```python
# Example: Optimize a noisy 1D function
from bayes_opt import BayesianOptimization

def black_box_function(x):
    return -x**2 + 4

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={'x': (-3, 3)},
    random_state=42
)

optimizer.maximize(
    init_points=2,
    n_iter=10
)

print(optimizer.max)
````

---

## **Requirements**

- Python 3.x
    
- numpy
    
- scipy
    
- scikit-learn
    
- matplotlib (for visualization)
    

  

Install dependencies:

```
pip install -r requirements.txt
```

---

## **Learning Outcomes**

- Understand how Gaussian Processes model unknown functions.
    
- Learn how acquisition functions balance exploration vs. exploitation.
    
- Gain experience applying Bayesian Optimization to real problems.
    

---

## **Attribution**

This repository is a fork of the original [bayesian-optimization](https://github.com/Badecar/bayesian-optimization) project.

Forked to preserve, study, and showcase the implementation as part of my portfolio.
