# Neural Network Design — A Sequential Pocket Reference

**How to use this:** Read top to bottom the first time — it mirrors the actual order of decisions you make
when designing a network from scratch. After that, dip into any single section to revise just that concept.
The Appendix at the end gives you the same information as pure lookup tables, no reading required.

**Running example (used throughout):** predict whether a student **passes or fails** using three features —
IQ ($x_1$), study hours ($x_2$), play hours ($x_3$). Output: $y = 1$ (pass) or $y = 0$ (fail).

---

## 1. Define the Problem

Every downstream decision — architecture, activation, loss — depends on answering this first.

| Problem Type | Example | Output Neurons | Output Activation | Loss Function |
|---|---|---|---|---|
| Regression | Predict a house price | 1 | Linear (none) | MSE / MAE / Huber / RMSE |
| Binary Classification | Pass / Fail | 1 | Sigmoid | Binary Cross-Entropy |
| Multi-class Classification | Cat / Dog / Horse | $C$ (number of classes) | Softmax | Categorical / Sparse Categorical Cross-Entropy |

**Dependency map** — this one decision cascades forward:

```
Problem Type
   ├──▶ Output layer size        (Section 2)
   ├──▶ Output activation        (Section 4)
   └──▶ Loss function            (Section 6)
```

Our running example is **binary classification** → output layer = 1 neuron → output activation = Sigmoid →
loss = Binary Cross-Entropy. Keep this thread in mind; it reappears in every section below.

**Code Structure** — every section from here adds one piece to these two class shells:

```python
import torch
import torch.nn as nn

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # architecture defined in Section 2 →

class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        # architecture defined in Section 2 →
```

---

## 2. Design the Architecture

- **Input layer size** = number of features. Here: 3 (IQ, study hours, play hours).
- **Hidden layer(s)**: your choice — any number of layers, any number of neurons per layer. More
  capacity to learn complex patterns, but more compute and more overfitting risk.
- **Output layer size**: from the table in Section 1.

**The core neuron computation**, repeated at every neuron in every layer:

$$z = \sum_{i=1}^{n} w_i x_i + b$$

**Matrix-shape logic**: if a layer has 3 inputs feeding into 2 hidden neurons, every input connects to
every neuron — so the weight matrix for that layer is $3 \times 2$ (one column of weights per hidden
neuron). This is why layers are called **fully connected / dense**.

**Code Structure** — input/hidden/output layers added:

```python
class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(in_features=3, out_features=8)  # 3 = IQ, study hrs, play hrs
        self.output = nn.Linear(in_features=8, out_features=1)  # regression → 1 output neuron

class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(in_features=3, out_features=8)
        self.output = nn.Linear(in_features=8, out_features=1)  # binary → 1 neuron (use C for multi-class)
```

---

## 3. Initialize the Weights

**Why it matters before anything else runs:**
- **Never initialize all weights to 0 (or the same value).** Every neuron would compute the identical
  output and identical gradient — they'd all learn the same thing forever (the *symmetry problem*).
- **Never initialize weights to large random values either.** Large weights push $z$ into the saturation
  region of Sigmoid/Tanh from the very first pass, and during backpropagation the repeated multiplication
  through the chain rule can make gradients **explode** (weights initialized in the 500–1000 range can
  cause $w_{new} \gg w_{old}$ — see Section 7's callback).

**Uniform Distribution:**

$$W_{ij} \sim \text{Uniform}\left(-\frac{1}{\sqrt{fan_{in}}},\ \frac{1}{\sqrt{fan_{in}}}\right)$$

**Xavier / Glorot Initialization** (pairs with **Sigmoid / Tanh**):

$$\text{Xavier Normal:}\quad W_{ij} \sim \mathcal{N}(0, \sigma^2), \qquad \sigma = \sqrt{\frac{2}{fan_{in}+fan_{out}}}$$

$$\text{Xavier Uniform:}\quad W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{6}{fan_{in}+fan_{out}}},\ \sqrt{\frac{6}{fan_{in}+fan_{out}}}\right)$$

**He / Kaiming Initialization** (pairs with **ReLU and its variants**):

$$\text{He Normal:}\quad W_{ij} \sim \mathcal{N}(0, \sigma^2), \qquad \sigma = \sqrt{\frac{2}{fan_{in}}}$$

$$\text{He Uniform:}\quad W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{6}{fan_{in}}},\ \sqrt{\frac{6}{fan_{in}}}\right)$$

**Quick pairing rule:**

| Activation in that layer | Use this init |
|---|---|
| Sigmoid / Tanh | Xavier / Glorot |
| ReLU / Leaky ReLU / ELU | He / Kaiming |
| Unsure / experimenting | Uniform |

**Code Structure** — init calls added, right after the layers are declared:

```python
class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 8)
        self.output = nn.Linear(8, 1)

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')  # He — hidden feeds ReLU
        nn.init.xavier_uniform_(self.output.weight)                      # output stays linear

class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 8)
        self.output = nn.Linear(8, 1)

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.output.weight)   # feeds into Sigmoid/Softmax next
```

---

## 4. Choose Activation Functions

**Why they exist:** without a non-linear activation, stacking any number of linear layers collapses
mathematically into a single linear function — the network could never learn a curved decision boundary.

| Function | Formula | Range | Advantage | Disadvantage |
|---|---|---|---|---|
| **Sigmoid** | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ | $(0, 1)$ | Clean probability output; ideal for binary output layer | Vanishing gradient (derivative maxes at 0.25); not zero-centered; costly (exponential) |
| **Tanh** | $\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$ | $(-1, 1)$ | Zero-centered → more efficient weight updates than Sigmoid | Still vanishing gradient in deep nets; costly (exponential) |
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | Solves vanishing gradient (derivative is 0 or 1); very fast to compute | "Dying ReLU": if $z<0$, derivative $=0$ → that weight stops updating; not zero-centered |
| **Leaky ReLU** | $\max(\alpha z,\ z),\ \alpha \approx 0.01$ | $(-\infty, \infty)$ | Fixes dying ReLU with a small non-zero slope for $z<0$ | $\alpha$ is a fixed hyperparameter you must choose |
| **Parametric ReLU (PReLU)** | same as Leaky ReLU, but $\alpha$ is **learned** | $(-\infty, \infty)$ | Network tunes $\alpha$ itself | Extra parameter to train |
| **ELU** | $z$ if $z>0$, else $\alpha(e^{z}-1)$ | $(-\alpha, \infty)$ | Zero-centered, no dead-neuron issue | Costlier to compute (exponential) |
| **Softmax** | $\dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | $(0,1)$, sums to 1 | Converts a whole output layer into a clean probability distribution | Only makes sense in the output layer, for multi-class problems |

**Worked Softmax example** — logits $z = [-1,\ 0,\ 3,\ 5]$ for classes $[\text{Cat, Dog, Monkey, Horse}]$:

$$e^{-1}\approx 0.368 \quad e^{0}=1 \quad e^{3}\approx 20.086 \quad e^{5}\approx 148.413$$

$$\text{Sum} \approx 169.867$$

$$P(\text{Cat})=\frac{0.368}{169.867}\approx 0.0022 \quad P(\text{Dog})=\frac{1}{169.867}\approx 0.0059$$

$$P(\text{Monkey})=\frac{20.086}{169.867}\approx 0.1183 \quad P(\text{Horse})=\frac{148.413}{169.867}\approx 0.8737$$

The highest logit wins, and all four probabilities sum to 1.

**Rule of thumb:**

| Layer | Recommended activation |
|---|---|
| Hidden layers | ReLU or a variant (Leaky ReLU, PReLU, ELU) — Sigmoid/Tanh in deep hidden stacks risks vanishing gradient |
| Output — regression | Linear (none) |
| Output — binary classification | Sigmoid |
| Output — multi-class classification | Softmax |

**Code Structure** — activation modules added; this is where the two classes first diverge:

```python
class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()                 # hidden-layer activation

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.output.weight)
        # no output activation → output stays linear, per Section 1's regression row

class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 8)
        self.output = nn.Linear(8, 1)              # → C neurons instead of 1, for multi-class
        self.relu = nn.ReLU()                       # hidden-layer activation
        self.out_activation = nn.Sigmoid()          # binary → Sigmoid; multi-class → nn.Softmax(dim=1)

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.output.weight)
```

---

## 5. Forward Propagation

Two steps, repeated layer by layer:

$$\text{Step 1 (linear):}\quad z = \sum_i w_i x_i + b$$

$$\text{Step 2 (activation):}\quad \text{output} = \text{activation}(z)$$

The final layer's output is the network's prediction, $\hat y$.

**Worked numeric example** (continuing the pass/fail dataset — IQ=95, study=4, play=4, actual $y=1$):

Hidden Layer 1 — weights $w_1=0.01,\ w_2=0.02,\ w_3=0.03$, bias $b_1=0.01$:

$$z = 95(0.01) + 4(0.02) + 4(0.03) + 1(0.01) = 1.151$$

$$O_1 = \sigma(1.151) = \frac{1}{1+e^{-1.151}} \approx 0.759$$

Hidden Layer 2 → Output — weight $w_4=0.02$, bias $b_2=0.03$:

$$z = O_1 \times w_4 + b_2 = 0.759 \times 0.02 + 0.03 = 0.04518$$

$$\hat y = \sigma(0.04518) \approx 0.51129$$

The network predicted **0.511** (barely leaning "pass") when the truth was **1 (pass)** — close, but with
real error. That error is exactly what Section 6 measures.

**Code Structure** — the `forward()` method, added to both classes as built through Section 4:

```python
class RegressionNet(nn.Module):
    # ... __init__ as built through Section 4 ...

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)          # y_hat — linear, no activation
        return x

class ClassificationNet(nn.Module):
    # ... __init__ as built through Section 4 ...

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.out_activation(self.output(x))   # y_hat — a probability
        return x
```

---

## 6. Choose the Loss Function

**Loss vs. Cost** — the one distinction that trips people up:
- **Loss** = error for **one** data point.
- **Cost** = average error across **all** data points (weights get updated once per batch/epoch using this).

$$\text{Loss (single point)}: \quad (y-\hat y)^2$$

$$\text{Cost (all } n \text{ points)}: \quad \frac{1}{n}\sum_{i=1}^{n} (y_i-\hat y_i)^2$$

From the worked example above: $\text{error} = y-\hat y = 1 - 0.511 \approx 0.489$ — this is the raw signal
that backpropagation (Section 7) will use to correct every weight.

### Regression losses

| Loss | Formula | When to prefer it |
|---|---|---|
| **MSE** | $\dfrac{1}{n}\sum (y-\hat y)^2$ | Default choice; differentiable, single global minimum, converges faster |
| **MAE** | $\dfrac{1}{n}\sum \lvert y-\hat y\rvert$ | Robust to outliers; but not smooth at 0 (needs sub-gradient), slower convergence |
| **Huber Loss** | $\begin{cases} \frac12(y-\hat y)^2 & \lvert y-\hat y\rvert \le \delta \\ \delta\lvert y-\hat y\rvert - \frac12\delta^2 & \text{otherwise}\end{cases}$ | Best of both — behaves like MSE normally, like MAE on outliers. $\delta$ is a tunable hyperparameter |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same unit as the target variable — easier to interpret |

### Classification losses (the "Cross-Entropy" family)

| Loss | Formula | Pairs with | Notes |
|---|---|---|---|
| **Binary Cross-Entropy** | $-y\log(\hat y) - (1-y)\log(1-\hat y)$ | Sigmoid output | Same log-loss formula as logistic regression |
| **Categorical Cross-Entropy** | $-\sum_{j=1}^{C} y_{ij}\ln(\hat y_{ij})$ | Softmax output | Labels must be **one-hot encoded**; gives full probability info for every class |
| **Sparse Categorical Cross-Entropy** | same idea, integer labels | Softmax output | Labels are plain integers (no one-hot needed); **downside**: loses probability info for non-predicted classes |

### The "Right Combination" table — the single most useful lookup here

| Hidden Layer Activation | Output Activation | Problem | Loss Function |
|---|---|---|---|
| ReLU / variants | Sigmoid | Binary Classification | Binary Cross-Entropy |
| ReLU / variants | Softmax | Multi-class Classification | Categorical or Sparse Categorical Cross-Entropy |
| ReLU / variants | Linear | Regression | MSE / MAE / Huber / RMSE |

**Code Structure** — the loss is declared *outside* the class, not inside it:

```python
# Regression
criterion = nn.MSELoss()
# criterion = nn.L1Loss()      # MAE
# criterion = nn.HuberLoss()   # Huber

# Classification
criterion = nn.BCELoss()               # binary — pairs with the Sigmoid output above
# criterion = nn.CrossEntropyLoss()    # multi-class — expects raw logits;
                                        # drop nn.Softmax from forward() if you use this
```

---

## 7. Backpropagation & Chain Rule

**Goal**: find $\dfrac{\partial \text{Loss}}{\partial w}$ for every single weight in the network, then correct it.

**Weight update rule** (the one formula everything else in this section serves):

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w_{old}} \qquad (\eta = \text{learning rate})$$

**Chain rule**, expanded — say a weight $w$ feeds a neuron whose output flows through several more
neurons before reaching the loss:

$$\frac{\partial L}{\partial w_{old}} = \frac{\partial L}{\partial o_3}\cdot\frac{\partial o_3}{\partial o_2}\cdot\frac{\partial o_2}{\partial o_1}\cdot\frac{\partial o_1}{\partial w_{old}}$$

Every link in that chain is itself $\dfrac{\partial(\text{activation})}{\partial z}\cdot\dfrac{\partial z}{\partial(\text{previous output})}$
— this is why the choice of activation function directly controls how well gradients survive the trip
backward.

**Callback to Section 3 — this is where init/activation choices bite:**
- If every activation in the chain is **Sigmoid**, each link is bounded to $[0, 0.25]$. Multiply several
  small numbers together ($0.25 \times 0.25 \times 0.25 \times \dots$) and the product shrinks toward
  **0** — so $w_{new} \approx w_{old}$. The weight barely moves. This is the **vanishing gradient
  problem**, and it's exactly why deep hidden stacks avoid Sigmoid/Tanh.
- If weights were initialized far too large instead (Section 3's warning), the same chain multiplication
  can blow up instead of vanish — $w_{new} \gg w_{old}$, the **exploding gradient problem**.

Both failure modes are the *same formula*, just pushed toward opposite extremes — which is exactly why
Sections 3 (init) and 4 (activation) exist before you ever get here.

**Code Structure** — same shape for both networks, one training step:

```python
y_hat = model(x_batch)
loss = criterion(y_hat, y_batch)
loss.backward()     # gradients now populated in every param.grad
```

---

## 8. Choose an Optimizer

Read this section as a chain of inventions — each optimizer exists to fix the previous one's specific flaw.

**1. Batch Gradient Descent** — updates weights using the **entire dataset** per step.

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w_{old}}$$

✅ Converges reliably. ❌ Huge RAM/GPU requirement on large datasets → **too resource-intensive.**

**2. Stochastic Gradient Descent (SGD)** — updates using **one data point** at a time.
✅ Fixes the resource problem. ❌ Very noisy convergence path, and more total time complexity → **too noisy.**

**3. Mini-batch SGD** — updates using a small batch (e.g. batch_size = 1000) per step.
✅ Balances resource use and noise, faster convergence than pure SGD. ❌ Noise still exists, just reduced
→ **needs smoothing.**

**4. SGD with Momentum** — smooths the path using an Exponentially Weighted Moving Average (EWMA) of past gradients:

$$V_t = \beta V_{t-1} + (1-\beta)\frac{\partial L}{\partial w_{t-1}}$$

$$w_t = w_{t-1} - \eta V_t$$

✅ Faster, smoother convergence. ❌ Still uses **one fixed learning rate** for the whole network →
**needs an adaptive rate.**

**5. Adagrad** (Adaptive Gradient) — makes the learning rate **dynamic**, shrinking it as training progresses:

$$\eta' = \frac{\eta}{\sqrt{\alpha_t+\epsilon}} \qquad \alpha_t = \sum_{i=1}^{t}\left(\frac{\partial L}{\partial w_i}\right)^2$$

✅ Naturally high learning rate early, low near convergence. ❌ $\alpha_t$ only ever grows, so $\eta'$ can
shrink toward **zero too aggressively** in deep networks, stalling training entirely → **decays too hard.**

**6. RMSprop / Adadelta** — replaces the ever-growing sum with an EWMA of squared gradients instead:

$$S_{dw,t} = \beta S_{dw,t-1} + (1-\beta)\left(\frac{\partial L}{\partial w_{t-1}}\right)^2$$

$$\eta' = \frac{\eta}{\sqrt{S_{dw,t}+\epsilon}}$$

✅ Dynamic learning rate without Adagrad's runaway shrinkage.

**7. Adam** (Adaptive Moment Estimation) — Momentum's numerator **+** RMSprop's denominator, combined:

$$V_{dw,t} = \beta_1 V_{dw,t-1} + (1-\beta_1)\frac{\partial L}{\partial w_{t-1}} \qquad \text{(momentum term)}$$

$$S_{dw,t} = \beta_2 S_{dw,t-1} + (1-\beta_2)\left(\frac{\partial L}{\partial w_{t-1}}\right)^2 \qquad \text{(adaptive-rate term)}$$

$$w_t = w_{t-1} - \frac{\eta}{\sqrt{S_{dw,t}+\epsilon}}\, V_{dw,t}$$

The most widely used default today — smooth, momentum-driven convergence **and** a per-parameter
adaptive learning rate, in one optimizer.

**The chain, at a glance:**

```
Batch GD → (too heavy) → SGD → (too noisy) → Mini-batch SGD → (still noisy)
   → Momentum → (fixed LR) → Adagrad → (LR decays too hard) → RMSprop/Adadelta
   → combine with Momentum → Adam
```

**Code Structure** — same declaration for both networks:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 9. Regularize with Dropout

**Overfitting, defined**: training accuracy is high (e.g. 90%) but test accuracy is much lower (e.g. 60%) —
the model memorized the training set instead of learning a generalizable pattern.

**Dropout mechanism**:
- On every training pass, randomly deactivate a fraction $p$ of neurons (a hyperparameter you choose).
- A *different* random set of neurons is dropped each epoch/iteration — no neuron can become "load-bearing."
- **At test/inference time**, dropout is switched off — every neuron is active and contributes.

```
model.train()   →  dropout active,   p% of neurons randomly off
model.eval()    →  dropout disabled, 100% of neurons active
```

**Code Structure** — `nn.Dropout` added into `__init__` and `forward()` of both classes:

```python
class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 8)
        self.drop = nn.Dropout(p=0.2)             # NEW
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.drop(x)                           # NEW
        x = self.output(x)
        return x

class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 8)
        self.drop = nn.Dropout(p=0.2)              # NEW
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.out_activation = nn.Sigmoid()

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.drop(x)                           # NEW
        x = self.out_activation(self.output(x))
        return x
```

---

## 10. Train the Network

Three terms that are easy to blur together:

- **Epoch** = one full pass through the *entire* training dataset.
- **Batch size** = how many data points are used per weight-update step.
- **Iteration** = one weight-update step.

$$\text{iterations per epoch} = \frac{\text{dataset size}}{\text{batch size}}$$

**Worked example**: dataset = 1,000,000 points, batch size = 1,000:

$$\text{iterations per epoch} = \frac{1{,}000{,}000}{1{,}000} = 1{,}000 \text{ iterations}$$

So *1 epoch* here means the optimizer takes 1,000 update steps before it has seen every data point once.

**Code Structure** — the full loop, calling every piece above in order:

```python
model = RegressionNet()                                     # or ClassificationNet()
criterion = nn.MSELoss()                                    # or nn.BCELoss() / nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):                              # Section 10 — epoch
    model.train()                                            # dropout active (Section 9)
    for x_batch, y_batch in train_loader:                     # Section 10 — iteration
        optimizer.zero_grad()
        y_hat = model(x_batch)                                # forward (Section 5)
        loss = criterion(y_hat, y_batch)                      # loss (Section 6)
        loss.backward()                                       # backprop (Section 7)
        optimizer.step()                                      # optimizer (Section 8)
```

---

## Assembled Code Skeletons

Everything from Sections 1–10, combined — nothing new appears here, it's just the two classes standing
complete.

```python
import torch
import torch.nn as nn

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(in_features=3, out_features=8)
        self.drop = nn.Dropout(p=0.2)
        self.output = nn.Linear(in_features=8, out_features=1)   # 1 neuron, linear — regression
        self.relu = nn.ReLU()

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.drop(x)
        x = self.output(x)             # y_hat — no output activation
        return x


class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(in_features=3, out_features=8)
        self.drop = nn.Dropout(p=0.2)
        self.output = nn.Linear(in_features=8, out_features=1)   # 1 neuron (binary); use C for multi-class
        self.relu = nn.ReLU()
        self.out_activation = nn.Sigmoid()                        # multi-class → nn.Softmax(dim=1)

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.drop(x)
        x = self.out_activation(self.output(x))   # y_hat — a probability
        return x


# --- Regression setup ---
model = RegressionNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Classification setup ---
model = ClassificationNet()
criterion = nn.BCELoss()                          # nn.CrossEntropyLoss() for multi-class
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Shared training loop ---
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)
        loss.backward()
        optimizer.step()
```

---

## Appendix — Quick-Lookup Tables

**Which activation?**

| Situation | Use |
|---|---|
| Hidden layers, general default | ReLU |
| Hidden layers, worried about dying ReLU | Leaky ReLU / PReLU / ELU |
| Output — binary classification | Sigmoid |
| Output — multi-class classification | Softmax |
| Output — regression | Linear (none) |

**Which loss?**

| Problem | Loss |
|---|---|
| Regression, no major outliers | MSE or RMSE |
| Regression, with outliers | MAE or Huber |
| Binary classification | Binary Cross-Entropy |
| Multi-class, need all class probabilities | Categorical Cross-Entropy |
| Multi-class, only need the predicted class | Sparse Categorical Cross-Entropy |

**Which optimizer?**

| Situation | Use |
|---|---|
| Small dataset, simplicity fine | Batch Gradient Descent |
| Large dataset, need speed + stability | Mini-batch SGD |
| Want smoother convergence | SGD with Momentum |
| Want adaptive learning rate, simple case | RMSprop |
| Default choice for most modern networks | **Adam** |

**Which weight init?**

| Activation used in that layer | Init |
|---|---|
| Sigmoid / Tanh | Xavier / Glorot |
| ReLU / Leaky ReLU / ELU | He / Kaiming |

---

## PyTorch Imports — In Step Order

Every step below either introduces a genuinely **new** import, or reuses one already imported in an
earlier step. Both are marked explicitly, so nothing is left implicit.

| Step | New Import Needed? | Import Statement | Function/Class Used at This Step | Where It Comes From |
|---|---|---|---|---|
| 1. Define the Problem | ✅ Yes | `import torch`<br>`import torch.nn as nn` | `nn.Module` — base class both `RegressionNet` and `ClassificationNet` inherit from | Introduced here |
| 2. Design the Architecture | ❌ No new import | *(reuses Step 1's `nn`)* | `nn.Linear` | `torch.nn` |
| 3. Initialize the Weights | ❌ No new import | *(reuses Step 1's `nn`)* | `nn.init.xavier_uniform_`, `nn.init.kaiming_normal_`, `nn.init.uniform_` | `torch.nn.init` — a submodule already available the moment `torch.nn` is imported, no separate import line needed |
| 4. Choose Activation Functions | ❌ No new import | *(reuses Step 1's `nn`)* | `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.LeakyReLU`, `nn.PReLU`, `nn.ELU`, `nn.Softmax` | `torch.nn` |
| 5. Forward Propagation | ❌ No new import | *(reuses Step 1's `torch`/`nn`)* | no new class — just the `forward()` method definition on the `nn.Module` subclass | `torch.nn.Module` (already imported) |
| 6. Choose the Loss Function | ❌ No new import | *(reuses Step 1's `nn`)* | `nn.MSELoss`, `nn.L1Loss`, `nn.HuberLoss`, `nn.BCELoss`, `nn.CrossEntropyLoss` | `torch.nn` |
| 7. Backpropagation | ❌ No new import | *(reuses Step 1's `torch`)* | `.backward()` — a method available on any loss tensor via autograd | `torch.autograd`, active automatically once `torch` is imported |
| 8. Choose an Optimizer | ✅ Yes | `import torch.optim as optim` | `optim.SGD`, `optim.Adam` | Introduced here |
| 9. Regularize with Dropout | ❌ No new import | *(reuses Step 1's `nn`)* | `nn.Dropout` | `torch.nn` |
| 10. Train the Network | ✅ Yes | `from torch.utils.data import Dataset, DataLoader` | `DataLoader` — batches data for the epoch/iteration loop | Introduced here |

**Cumulative import block** — by Step 10, this is everything you need at the top of the file, in the order
it became necessary:

```python
import torch
import torch.nn as nn            # Steps 1–7, 9 — Module, Linear, init, activations, losses, Dropout
import torch.optim as optim      # Step 8 — optimizers
from torch.utils.data import Dataset, DataLoader   # Step 10 — batching/training loop
```

Only **three** import lines are ever genuinely new across all 10 steps — everything else is `nn.<something>`,
reused from the single `torch.nn` import in Step 1.

---

## Interview Questions — Senior Data Scientist Bar

These are calibrated to what actually gets asked at the senior level: trade-off reasoning, failure
diagnosis, and "why this over that" judgment — not definitions. Organized by the same topics as the
sections above, plus a final set of cross-cutting scenario questions.

### Weight Initialization

**Q1. Why would you choose He initialization over Xavier for a ReLU-based hidden layer, and what
specifically breaks if you use Xavier instead?**
Xavier assumes the activation is roughly linear/symmetric around zero (true for Sigmoid/Tanh), so its
variance term $2/(fan_{in}+fan_{out})$ is derived to keep signal variance stable through such activations.
ReLU zeroes out roughly half its inputs, effectively halving the variance passing forward. He init
compensates by doubling the effective variance ($2/fan_{in}$) specifically to counter that. Using Xavier
with ReLU under-estimates the needed variance, so activations shrink layer over layer — a mild
vanishing-signal issue introduced at initialization itself, causing slower early convergence, especially
in deeper ReLU networks.

**Q2. You initialize a deep network with every weight set to the same large constant. Training loss
doesn't move at all after several epochs. Walk through why.**
This isn't the classic vanishing-gradient case — it's the symmetry problem. Every neuron in a layer
receives identical weights, so they produce identical outputs and identical gradients, and every neuron
in that layer updates identically forever — the layer effectively learns only one feature no matter how
large the weight was. To confirm this diagnosis specifically (not vanishing gradients), check whether
neurons within a layer have near-identical weight values and gradients throughout training — in vanishing
gradients, weights would still move, just imperceptibly; here they move identically.

**Q3. If you initialize weights using He initialization but apply a Sigmoid activation instead of ReLU,
what happens and why?**
He init assumes ReLU zeroes about half the inputs, so it compensates with a larger variance than Xavier's.
Applying it to Sigmoid means pre-activation values land in Sigmoid's saturating tails more often than
Sigmoid's own derivation assumes, so gradients there are near zero — symptoms resemble vanishing gradient,
but the root cause is a mismatched init/activation pairing, not depth. Fix: use Xavier, which matches
Sigmoid's assumptions.

**Q4. For transfer learning where you freeze most layers and fine-tune only the last few, does weight
init still matter, and if so where?**
Initialization is irrelevant for frozen layers — their weights come from the pretrained model and are
never reinitialized. It matters entirely for the newly added, unfrozen layers (usually a fresh
classification head): use He if that head uses ReLU, Xavier if it ends in a Sigmoid/Softmax you're
training directly.

### Activation Functions

**Q5. In a deep binary classifier with Sigmoid in every hidden layer and the output, early layers barely
update while later layers train fine. Diagnose this.**
Classic vanishing gradient. The chain rule multiplies Sigmoid's derivative (bounded to $[0, 0.25]$) at
every layer during backprop. In an $N$-layer all-Sigmoid network, the gradient reaching layer 1 is a
product of $N$ terms each $\le 0.25$, shrinking exponentially with depth. Later layers are closer to the
loss (fewer multiplications) so they still get usable signal; earlier layers get almost none. Fix: switch
hidden layers to ReLU/variants (derivative is 0 or 1, no shrinking multiplier), keep Sigmoid only at the
output.

**Q6. When would you actually prefer Leaky ReLU or PReLU over plain ReLU, and what's the real cost?**
When you observe a large fraction of dead neurons — always outputting 0, gradient stuck at 0 — commonly
from too-high a learning rate or unlucky initialization pushing many pre-activations negative. Leaky ReLU
keeps a small non-zero gradient ($\alpha \approx 0.01$) for negative $z$ so those neurons can recover.
Cost: an extra hyperparameter to tune ($\alpha$) for Leaky ReLU, or extra trainable parameters for PReLU
(more compute, slightly higher overfitting risk on small datasets) — and empirically the gain over plain
ReLU is often marginal outside dying-ReLU-heavy setups.

**Q7. You build a 100-class classifier using Sigmoid at the output with independent BCE losses per class
instead of Softmax. What's mathematically wrong (or right) about this?**
It's valid for **multi-label** classification (classes independent — an image can have several true tags)
but wrong for **multi-class** (mutually exclusive) classification. Sigmoid outputs aren't constrained to
sum to 1, so they don't model the "exactly one class true" competition the way Softmax's normalization
does — the model can produce multiple confident classes at once, or none, hurting calibration and
accuracy. Diagnostic: check the label encoding — a single one-hot true class means it should be Softmax +
Categorical Cross-Entropy, not per-class Sigmoid + BCE.

**Q8. ELU has a hyperparameter $\alpha$ controlling its negative saturation value. How does increasing
$\alpha$ affect training dynamics?**
For $z<0$, ELU's output approaches $-\alpha$ as $z \to -\infty$. A larger $\alpha$ lowers that saturation
floor, pushing the resting value for very negative inputs further from zero, which works against ELU's
own zero-centering benefit if pushed too far. It also moves the function further from a near-linear region
for moderately negative $z$, which can slow learning if the loss landscape needed that near-linearity.
$\alpha=1$ is the standard default balancing this; pushing it further usually gives diminishing returns.

### Loss Functions

**Q9. Your regression targets have a small number of extreme but valid outliers you can't remove. MSE,
MAE, or Huber — and how would you set $\delta$?**
Huber is the natural fit: quadratic (like MSE) for small errors, giving stable gradients near the optimum,
and linear (like MAE) beyond $\delta$, capping any single outlier's influence on the gradient. Set
$\delta$ near the boundary between "typical noise" and "outlier" in your residual distribution — e.g., if
normal residuals sit within $\pm 1$ but rare outliers spike to $\pm 20$, start around $\delta \approx 1$–$1.5$
and tune against validation performance. Pure MAE is the fallback if outliers are common enough that even
Huber's dampening isn't sufficient, at the cost of MSE's faster, smoother convergence near the minimum.

**Q10. You trained with Sparse Categorical Cross-Entropy and get good accuracy, but a teammate wants label
smoothing and per-class probability calibration analysis. What do you tell them?**
Sparse CCE only tracks the log-probability of the true class index during loss computation — it discards
information about how probability mass is spread across the incorrect classes, even though the model's
Softmax output still technically produces a full distribution at inference. Label smoothing and full
calibration analysis both need the complete predicted distribution against a soft/one-hot target, which is
what standard Categorical Cross-Entropy is built for. Fix: switch the loss to Categorical Cross-Entropy
with one-hot (or smoothed) labels — the architecture and Softmax output don't need to change.

**Q11. Binary Cross-Entropy includes $\log(\hat y)$ and $\log(1-\hat y)$. What happens numerically if
$\hat y$ is exactly 0 or 1, and how do frameworks handle it?**
$\log(0) = -\infty$, so if the true label doesn't match a saturated $\hat y$, the loss explodes to
infinity — a numerically catastrophic, non-differentiable point. Frameworks typically clamp probabilities
away from exact 0/1 internally, or better, use a combined logits-based loss (e.g., PyTorch's
`BCEWithLogitsLoss`) that folds Sigmoid and BCE into one numerically stable computation via the
log-sum-exp trick, avoiding an explicit $\log$ on a saturated probability entirely. This is why
logits-based combined losses are generally preferred over manually chaining Sigmoid then BCE.

**Q12. Why does Categorical Cross-Entropy pair with Softmax specifically, mathematically — not just by
convention?**
The gradient of CCE with respect to the pre-activation logits, when the output uses Softmax, simplifies
to $\hat y - y$ — predicted probability minus the true one-hot label, per class. That clean, well-scaled
gradient is a direct consequence of Softmax's normalization interacting with the cross-entropy log term;
it doesn't hold for other output activations. This is why frameworks bundle them — PyTorch's
`CrossEntropyLoss` applies Softmax internally rather than requiring you to add it separately.

### Backpropagation & Chain Rule

**Q13. `loss.backward()` runs fine, but one specific layer's `.grad` is consistently `None`. What are the
likely causes, and how do you debug it?**
Most likely: that layer's parameters aren't part of the computation graph — the tensor was `.detach()`ed,
created inside a `torch.no_grad()` block, or has `requires_grad=False` (intentionally for frozen layers,
or by accident). Second cause: the layer's output was never actually routed into the loss — autograd only
tracks gradients along paths that contribute to it. Debug by printing `param.requires_grad`, and tracing
the forward pass to confirm that layer's output genuinely reaches the loss computation.

**Q14. Explain precisely why exploding and vanishing gradients are "the same formula, different
direction" — what's the shared mechanism?**
Both come from the chain rule's repeated multiplication:
$\partial L/\partial w_{early}$ = product of many local derivative terms chained across layers. If each
term has magnitude $<1$ (e.g., Sigmoid's bounded $[0,0.25]$ derivative), the product shrinks exponentially
with depth — vanishing gradient. If each term has magnitude $>1$ (large weight init, or activation
derivatives $>1$ in some regions), the product grows exponentially — exploding gradient. It's the identical
multiplicative structure; only the direction of the per-layer terms differs.

**Q15. Would gradient clipping alone fix a vanishing-gradient problem? Why or why not?**
No. Clipping caps gradients that are too large by rescaling them past a norm threshold — it addresses
exploding gradients (common in RNNs). It does nothing for gradients that are already near-zero; there's
no equivalent "floor." Vanishing gradients need a different fix: better activation choice (ReLU/variants
over Sigmoid/Tanh in deep stacks), matched weight init, architectural changes (residual/skip connections),
or normalization techniques that keep activations in a well-conditioned range.

### Optimizers

**Q16. Fine-tuning a large pretrained model with a small learning rate — SGD with Momentum or Adam, and
why might the answer differ from training from scratch?**
For fine-tuning, Adam (or AdamW) is often preferred: its adaptive per-parameter rate handles the uneven
gradient scales across a pretrained model's layers without manual per-layer LR scheduling. Training from
scratch on abundant data, well-tuned SGD with Momentum can sometimes generalize better (seen empirically
in vision, e.g. ResNets), because Adam's adaptive rates can converge to sharper minima that generalize
slightly worse. In practice the answer is often architecture/domain-specific convention (NLP transformers
almost universally use Adam/AdamW) rather than a universal rule.

**Q17. Adam's loss curve is smooth and converging, but validation performance is worse than an SGD +
Momentum baseline on the same setup. What would you investigate?**
A known pattern: naive weight decay under Adam gets divided by the same adaptive $S_{dw}$ denominator as
the gradient, unintentionally weakening regularization for parameters with large historical gradients. I'd
check whether **AdamW** (decoupled weight decay) was used instead of Adam with an L2 penalty folded into
the loss — switching often closes the gap — plus whether LR schedule/warmup differed between runs, and
whether Adam is simply overfitting faster (compare train/val gap over epochs).

**Q18. In Adagrad, a weight's gradient is unusually large in the first few iterations, then near-zero
afterward. What happens to its effective learning rate for the rest of training — and is that desirable?**
Adagrad's $\alpha_t$ accumulates the sum of squared gradients for that weight over *all* past
iterations — it only ever grows. A few early large gradients permanently inflate $\alpha_t$, permanently
shrinking that weight's effective rate $\eta'$ for the rest of training, even once it would benefit from
faster updates again. This is Adagrad's core flaw, and exactly why RMSprop/Adadelta replace the
ever-growing sum with an exponentially-decaying moving average — recent gradient history should matter
more than a permanent, unbounded accumulation from early training.

**Q19. Adam maintains $V_{dw}$ (first moment) and $S_{dw}$ (second moment). What happens if you set
$\beta_2$ unusually low, like 0.5, instead of the typical 0.999?**
A low $\beta_2$ makes $S_{dw}$ forget past squared-gradient history quickly, weighting only the last
couple of steps. The adaptive-rate term $\eta/\sqrt{S_{dw}+\epsilon}$ becomes much more reactive and noisy,
swinging based on very recent, possibly noisy gradients rather than a stable estimate of that parameter's
typical scale — reintroducing much of the instability Adam was designed to prevent. This is why
$\beta_2=0.999$ (long memory) is standard, while $\beta_1$ (momentum, 0.9) is intentionally shorter-memory
since it benefits from being responsive to recent gradient direction.

### Regularization / Dropout

**Q20. You add `Dropout(p=0.5)` to every layer of a shallow, 2-hidden-layer network and validation
performance gets *worse*. Why, and what would you change?**
Dropout's strength should match the model's overfitting risk — a shallow, low-capacity network doesn't
have much redundancy to spare, and $p=0.5$ can cripple its ability to learn at all (underfitting), rather
than just preventing memorization. I'd reduce $p$ substantially (0.1–0.2), apply it more selectively (only
near the output, or only where a train/val gap is actually observed), or check whether regularization was
even needed — if train and val performance were already close before adding Dropout, overfitting wasn't
the problem to begin with.

**Q21. A colleague forgets `model.eval()` before running inference on a Dropout model in production. What
symptom would you expect, and why is it dangerous?**
Dropout stays active, so each forward pass randomly drops a different subset of neurons — the same input
fed twice can produce **different outputs** each time. This is dangerous because it silently degrades
predictions without throwing any error: the model keeps running, just non-reproducibly and with reduced
effective capacity, typically showing up as a persistent, unexplained quality regression that's easy to
miss unless someone specifically checks output determinism.

**Q22. Why is Dropout sometimes described as "approximating an ensemble of exponentially many
sub-networks"? What's the intuition?**
Every training forward pass samples one random "thinned" sub-network (whichever neurons survive that
pass's mask). Across training, the shared weights are jointly shaped by an enormous number of these
distinct sub-networks ($2^n$ possible masks for $n$ droppable neurons). At test time, with Dropout
disabled and all neurons active, the resulting single dense network approximates the averaged prediction
behavior of that implicit ensemble — a similar generalization benefit to training and averaging many
separate networks, at a fraction of the cost, since all of them share the same underlying weights.

### Architecture & Problem-Framing (Cross-Cutting Scenarios)

**Q23. You're predicting customer churn (yes/no), and the business also wants a calibrated probability
score to prioritize outreach, not just a label. How does this change your output layer, loss, and
evaluation?**
The output layer stays Sigmoid and Binary Cross-Entropy remains the right loss — BCE is already a proper
scoring rule that optimizes for calibrated probabilities, not just correct labels, so this alone doesn't
force an architecture change. What changes is evaluation: report calibration directly (reliability
diagrams, Brier score) rather than only accuracy/F1 at a 0.5 threshold, and consider post-hoc calibration
(Platt scaling, isotonic regression) if raw outputs are miscalibrated — common under class imbalance,
since BCE optimizes average log-loss across the dataset, not per-bucket calibration.

**Q24. Training accuracy climbs to 97%, validation plateaus at 65% and starts degrading after epoch 10.
Walk through your full diagnostic process.**
This is a classic overfitting signature. Order of checks: (1) rule out data leakage between train/val
before anything else; (2) check model capacity versus dataset size — reduce width/depth if oversized;
(3) add or increase Dropout, consider weight decay; (4) verify weight init/activation choices didn't cause
the model to memorize noise unusually fast; (5) consider whether more data or augmentation is feasible
instead of just constraining capacity; (6) add early stopping tied to the validation metric — epoch 10's
degradation point is itself informative as roughly the ideal stopping point regardless of other fixes.
Present this as a prioritized checklist, since more than one cause is often true simultaneously.

**Q25. Choosing between a wide single hidden layer and a deep stack of narrower layers for ~50 features
and 100k rows of tabular data — what are the real trade-offs?**
A single wide layer can, per the universal approximation theorem, approximate any continuous function in
principle, but may need an impractically large neuron count and doesn't build hierarchical features
naturally. A deeper, narrower stack builds progressively abstract representations and is often more
parameter-efficient, but is more prone to vanishing/exploding gradients and needs more careful init and
data volume to justify. For a moderate tabular dataset like this, a moderate depth (2–3 hidden layers)
is the sensible default — deep architectures are usually reserved for large, hierarchical data like images
or text, and it's worth flagging that gradient-boosted trees often remain competitive or superior on this
exact data profile, which is a legitimate alternative to raise, not just a hyperparameter question.

**Q26. Justify, end-to-end, why ReLU + He init + Adam + Binary Cross-Entropy + Sigmoid output cohere as a
*system* for a binary classification MLP — not just what each piece does individually.**
These choices are mutually reinforcing, not independent. He init is derived assuming ReLU's
variance-halving behavior, so pairing them keeps activation variance stable across depth from the first
forward pass. ReLU's non-saturating derivative then prevents vanishing gradients during backprop — which
matters because Adam's adaptive rate normalizes gradient *magnitude* per parameter but does nothing for a
gradient that's structurally near-zero from saturation; they solve different problems, and Adam can't
substitute for the wrong activation/init choice. On the output side, Sigmoid is required because the
problem is binary, and BCE is the loss whose gradient interacts cleanly with Sigmoid's output. The system
coheres because each piece's assumptions match the one before it — swap one (say, Tanh for ReLU) without
reconsidering the others (init should shift to Xavier) and the coherence breaks, reintroducing exactly the
failure modes the stack was built to avoid.

**Q27. Offline evaluation looks great, but a stakeholder asks: "how do you know it'll still work in 6
months as customer behavior shifts?" How does this connect to overfitting/regularization, and what would
you monitor?**
This is generalization extended over time — concept drift — a superset of the train/val overfitting
problem. A model can generalize well to a same-distribution validation set yet still fail as the true
distribution shifts post-deployment, effectively "overfitting" to a moment in time rather than to a
training set. I'd monitor live prediction distributions against training-time feature distributions, track
a delayed-ground-truth performance proxy where available, and set up periodic retraining triggers rather
than assuming a static deployment. This connects back to regularization directly: an appropriately
regularized model (right amount of Dropout/weight decay, not over-memorized to training-period noise)
tends to degrade more gracefully under drift than one that overfit to training-period idiosyncrasies.
