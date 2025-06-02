# LaTeX Markdown Cheat Sheet


--------------------------------------------------------------------
## 1. LaTeX Math 101 

| Idea                    | Write                                    | Renders as |
|-------------------------|------------------------------------------|------------|
| Superscript / Subscript | `$x^{2}$`, `$a_{ij}$`                    | \(x^{2}\), \(a_{ij}\) |
| Fractions / Binomials   | `$\frac{a}{b}$`, `$\binom{n}{k}$`        | \(\frac{a}{b}\), \(\binom{n}{k}\) |
| Greek letters           | `$\alpha,\beta,\Gamma$`                  | \(\alpha,\beta,\Gamma\) |
| Roots                   | `$\sqrt{2}$`, `$\sqrt[n]{x}$`            | \(\sqrt{2},\sqrt[n]{x}\) |
| Functions (upright)     | `$\sin x,\; \ln y$`                      | \(\sin x, \ln y\) |
| Spacing                 | `\, \; \quad`                            | small → big |
| Text inside math        | `$\text{if } x>0$`                       | “if \(x>0\)” |
| Sets / logic            | `$\forall x\in\mathbb R$`                | \(\forall x\in\mathbb R\) |
| Vectors / matrices      | `$\mathbf v,\; \begin{bmatrix}1&0\\0&1\end{bmatrix}$` | |

--------------------------------------------------------------------
## 2. Single-variable calculus

| Concept                           | Write |
|-----------------------------------|-------|
| Derivative (prime)                | `$f'(x) = \frac{d}{dx}f(x)$` |
| Derivative (Leibniz)              | `$\displaystyle \frac{dy}{dx} = \lim_{\Delta x\to0}\frac{\Delta y}{\Delta x}$` |
| n-th derivative                   | `$\frac{d^{n}y}{dx^{n}}$` |
| Indefinite integral               | `$\int f(x)\,dx$` |
| Definite integral                 | `$\int_a^b f(x)\,dx$` |
| Fundamental Thm. of Calculus      | `$\displaystyle \frac{d}{dx}\int_{a}^{x}f(t)\,dt = f(x)$` |
| Taylor series (about 0)           | `$f(x)=\sum_{k=0}^{\infty}\frac{f^{(k)}(0)}{k!}x^{k}$` |
| Limit                             | `$\lim_{x\to c} f(x)$` |

--------------------------------------------------------------------
## 3. Multivariable calculus

| Concept                 | Write |
|-------------------------|-------|
| Gradient                | `$\nabla f = \left\langle\frac{\partial f}{\partial x},\frac{\partial f}{\partial y},\frac{\partial f}{\partial z}\right\rangle$` |
| Divergence              | `$\nabla\!\cdot\!\mathbf F$` |
| Curl                    | `$\nabla\times\mathbf F$` |
| Laplacian               | `$\nabla^{2}f = \Delta f$` |
| Multiple integral       | `$\iiint_\Omega g(x,y,z)\,dV$` |
| Line integral           | `$\displaystyle \int_{\mathcal C}\mathbf F\!\cdot d\mathbf r$` |
| Surface integral        | `$\displaystyle \iint_{S} \mathbf F\!\cdot d\mathbf S$` |
| Jacobian matrix         | `$\mathbf J_{ij} = \frac{\partial f_i}{\partial x_j}$` |
| Change-of-vars (2-D)    | `$\displaystyle \iint_{R} f(x,y)\,dx\,dy = \iint_{R'} f\bigl(x(u,v),y(u,v)\bigr)\left|\frac{\partial(x,y)}{\partial(u,v)}\right|du\,dv$` |

--------------------------------------------------------------------
## 4. Linear algebra

| Concept                         | Write |
|---------------------------------|-------|
| Column vector                   | `$\mathbf v=\begin{bmatrix}v_1\\v_2\\v_3\end{bmatrix}$` |
| Matrix                          | `$\mathbf A=\begin{bmatrix}a&b\\c&d\end{bmatrix}$` |
| Transpose                       | `$\mathbf A^{\mathsf T}$` |
| Inverse                         | `$\mathbf A^{-1}$` |
| Determinant                     | `$\det(\mathbf A)$` |
| Dot product                     | `$\mathbf u\cdot\mathbf v = \sum_{i}u_i v_i$` |
| Cross product                   | `$\mathbf u\times\mathbf v$` |
| Eigenproblem                    | `$\mathbf A\mathbf v = \lambda\mathbf v$` |
| Singular Value Decomposition    | `$\mathbf A = \mathbf U\boldsymbol\Sigma\mathbf V^{\mathsf T}$` |
| Projection onto $\mathbf u$     | `$\operatorname{proj}_{\mathbf u}\mathbf v = \frac{\mathbf v\cdot\mathbf u}{\mathbf u\cdot\mathbf u}\mathbf u$` |

--------------------------------------------------------------------
## 5. Statistics & Probability

| Concept            | Write |
|--------------------|-------|
| Expectation        | `$\mathbb E[X] = \sum_x x\,p(x)$` |
| Variance           | `$\operatorname{Var}(X)=\mathbb E[(X-\mu)^2]$` |
| Covariance         | `$\operatorname{Cov}(X,Y)=\mathbb E[(X-\mu_X)(Y-\mu_Y)]$` |
| Gaussian pdf       | `$\displaystyle f(x)=\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$` |
| Bernoulli pmf      | `$p(x)=p^{\,x}(1-p)^{1-x},\; x\in\{0,1\}$` |
| Poisson pmf        | `$P(X=k)=\dfrac{\lambda^{k}e^{-\lambda}}{k!}$` |
| CDF                | `$\displaystyle F_X(x)=P(X\le x)=\int_{-\infty}^{x}f(t)\,dt$` |
| Probability        | `$P(A)$` |
| Conditional prob.  | `$P(A\mid B)=\dfrac{P(A\cap B)}{P(B)}$` |
| Bayes’ rule        | `$P(A\mid B)=\dfrac{P(B\mid A)P(A)}{P(B)}$` |
| Law of Large Numbers| `$\bar X_n\xrightarrow{a.s.}\mu$` |

--------------------------------------------------------------------
## 6. Machine Learning essentials

| Concept                    | Write |
|----------------------------|-------|
| Linear regression model    | `$\hat y = \mathbf w^{\mathsf T}\mathbf x + b$` |
| MSE loss                   | `$\mathcal L = \frac{1}{n}\sum_{i=1}^{n}(\hat y_i - y_i)^{2}$` |
| Cross-entropy loss         | `$\mathcal L = -\sum_{i=1}^{n} y_i \log \hat y_i$` |
| Softmax                    | `$\sigma(\mathbf z)_j=\frac{e^{z_j}}{\sum_k e^{z_k}}$` |
| Logistic sigmoid           | `$\sigma(z)=\frac{1}{1+e^{-z}}$` |
| Gradient descent update    | `$\theta \leftarrow \theta - \eta\,\nabla_{\!\theta}\mathcal L$` |
| Back-prop (chain rule)     | `$\dfrac{\partial\mathcal L}{\partial\mathbf W} = \dfrac{\partial\mathcal L}{\partial\mathbf Z}\dfrac{\partial\mathbf Z}{\partial\mathbf W}$` |
| Batch-norm transform       | `$\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^{2}+\varepsilon}}$` |
| Attention weights          | `$\operatorname{softmax}\!\bigl(\frac{\mathbf Q\mathbf K^{\mathsf T}}{\sqrt{d_k}}\bigr)$` |
| KL divergence              | `$\operatorname{KL}(P\;\|\;Q) =\sum_x P(x)\log\dfrac{P(x)}{Q(x)}$` |

--------------------------------------------------------------------
## 7. Quick syntax reminders

| Tip | Example |
|-----|---------|
| Escape underscores | `$\text{my\_var}$` |
| Force display-style | `$\displaystyle \frac{a}{b}$` |
| Bold math symbols | `\mathbf{A}`, `\boldsymbol{\theta}` |
| Spacing commands | `\,`, `\;`, `\quad` |
| Backtick-dollar in code | `` `$E=mc^2$` `` |

--------------------------------------------------------------------
