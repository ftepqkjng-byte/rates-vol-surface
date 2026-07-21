1. Why P1?

   (1M, 3M, 1Y, ... from same factor)
   Heston: all vol comes from instant var $v_t$
   Bergomi: $v_t \to \xi_t^u = \mathbb{E}_t[v_u]$ fwd var curve
            a curve (piecewise) as state i.o. 1 state $v_t$
   P1 (2 factor Bergomi): the curve can steepen / flatten w/ $2^{nd}$ factor
      1-factor will have $d\xi_t^u = f(u) dW_t$ then $\text{Corr}(\xi_t^{u_1}, \xi_t^{u_2}) \equiv 1$

   For Auto Call, what matters is $\mathbb{P}(S_{t_i} > B)$ } i.e. $(S_i)$ joint distri-
   For Cliquet, Payoff related to every $\frac{S_{t_i}}{S_{t_{i-1}}} - 1$ }   bution

   Need a reasonable fwd var term structure
   And long + short factor to separate short-term influences
   e.g. FOMC

---

2. Model:
   $$\frac{dS_t}{S_t} = r_t dt + \sqrt{\xi_t^t} \sigma_{LV}(t, S_t) dW_t^S$$
   $$\frac{d\xi_t^u}{\xi_t^u} = 2 \nu \alpha_\theta \left( (1-\theta) e^{-k_1(u-t)} dW_t^X + \theta e^{-k_2(u-t)} dW_t^Y \right)$$
   $$\qquad \hookrightarrow = \left((1-\theta)^2 + \theta^2 + 2\rho_{XY}\theta(1-\theta)\right)^{-1/2}$$

   Params: $\nu, k_1, k_2, \rho_{SX}, \rho_{SY}, \rho_{XY}, \theta$

   Change:
   $$\omega_1 = 2\nu\alpha_\theta(1-\theta), \quad \omega_2 = 2\nu\alpha_\theta\theta, \quad \lambda_1 = \rho_{SX}\omega_1, \quad \lambda_2 = \rho_{SY}\omega_2$$
   $$\chi = (\rho_{XY} - \rho_{SX}\rho_{SY}) / \sqrt{1 - \rho_{SX}^2}\sqrt{1 - \rho_{SY}^2}$$

   $\Rightarrow$ Params $\chi, k_1, k_2, \omega_1, \omega_2, \lambda_1, \lambda_2$

   $\lambda_1, \lambda_2$ can control (1) ATM skew (Vanilla)
   $\omega_1, \omega_2$ can control VolVar
   (while (2) spot-vol cov invariant)

   // Skew can be computed using $\rho_{SX}, \rho_{SY}$

   Calib skew & ATMF vol at the same time:
   (1) VarSwap $\to \xi_0^t$  (2) solve $\rho_{SX}, \rho_{SY}$  (3) Re-calib $\xi_0^t$ to fit vanilla

---

3. $\frac{1}{k_1}, \frac{1}{k_2}$ : for a shock (short or long) how long it will last

   Omegas: 
   (1) $\omega_i$ control volvar movement
   (2) $\omega_1^2$ vs $\omega_2^2$ whether var comes from short or long

   Lambdas:
   (1) Larger $\lambda \Rightarrow$ Larger skew
   (2) Suppose mkt $\downarrow\downarrow$ how much will short / long var $\uparrow$

   $\chi$: Curve twisting. Large $\chi \Rightarrow$ fwd var curve more like parallel shift

   Essentially, regime-switch?