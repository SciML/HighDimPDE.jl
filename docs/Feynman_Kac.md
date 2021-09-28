## Feynman Kac

Feynman Kac formula is generally stated for Terminal Value problem, where
$$
\begin{equation}
\partial_t u(t,x) + \mu(x) \nabla_x u(t,x) + \frac{1}{2} \sigma^2(x) \Delta_x u(t,x) + f(x, u(t,x))  = 0
\end{equation}
$$
with terminal condition $u(T, x) = g(x)$, where $u \colon \R^d \to \R$ is the solution we wish to approximate. 

In this case the FK formula states that for all $t \in (0,T)$ it holds that

$$
\begin{equation}
u(t, x) = \int_t^T \E \left[ f(X^x_{s-t}, u(s, X^x_{s-t}))ds \right] + \E \left[ u(0, X^x_{T-t}) \right]
\end{equation}
$$
where 
$$
\begin{equation}
X_t^x = \int_0^t \mu(X_s^x)ds + \int_0^t\sigma(X_s^x)dB_s + x,
\end{equation}
$$



How to transform previous equation to an initial value problem?

Define $v(\tau, x) = u(T-\tau, x)$. Observe that $v(0,x) = u(T,x)$. Further observe that by the chain rule
$$
\begin{equation}
\begin{aligned}
\partial_\tau v(\tau, x) &= \partial_\tau u(T-\tau,x)\\
                        &= (\partial_\tau (T-\tau)) \partial_t u(T-\tau,x)\\
                        &= -\partial_t u(T-\tau, x)
\end{aligned}
\end{equation}
$$

From Eq. (1) we get that 
$$
\begin{equation}
- \partial_t u(T - \tau,x) = \mu(x) \nabla_x u(T - \tau,x) + \frac{1}{2} \sigma^2(x) \Delta_x u(T - \tau,x) + f(x, u(T - \tau,x)) 
\end{equation}
$$
Replacing we get that $v$ satisfies
$$
\begin{equation}
\partial_\tau v(\tau, x) = \mu(x) \nabla_x v(\tau,x) + \frac{1}{2} \sigma^2(x) \Delta_x v(\tau,x) + f(x, v(\tau,x)) 
\end{equation}
$$
and from Eq. (2) we get that

$$
\begin{equation}
v(\tau, x) = \int_{T-\tau}^T \E \left[ f(X^x_{s- T + \tau}, v(s, X^x_{s-T + \tau}))ds \right] + \E \left[ v(0, X^x_{\tau}) \right]
\end{equation}
$$
By using the substitution rule $\cdot \to \cdot -T$ (shifting by T), then $\cdot \to - \cdot $ (inversing) and finally inversing the integral bound we get that 
$$
\begin{equation}
\begin{aligned}
v(\tau, x) &= \int_{-\tau}^0 \E \left[ f(X^x_{s + \tau}, v(s + T, X^x_{s + \tau}))ds \right] + \E \left[ v(0, X^x_{\tau}) \right]\\
            &= - \int_{\tau}^0 \E \left[ f(X^x_{\tau - s}, v(T-s, X^x_{\tau - s}))ds \right] + \E \left[ v(0, X^x_{\tau}) \right]\\
            &= \int_{0}^\tau \E \left[ f(X^x_{\tau - s}, v(T-s, X^x_{\tau - s}))ds \right] + \E \left[ v(0, X^x_{\tau}) \right]
\end{aligned}
\end{equation}
$$
