# O7 serial synthetic twin inversion

O7 is a same-model software-validation twin experiment, not a real-data
identifiability study. It uses the fixed-mesh one-step
`adjoint_time_dependent_temperature` fixture and the O6 reduced increment
convention `T = T_fixed + P m`.

The deterministic interior truth is defined in reduced-control coefficient
space by `m_true(i) = 3e-3 * (0.7 + 0.3 sin(2 pi (i+1)/N))`; the initial guess
is `m_init = 0`. Bounds are `[-1e-2, 1e-2]`. A truth forward evaluation copies
its terminal state into the existing terminal-temperature objective as the
in-memory observation. No optimizer trajectory, gradient, or cache is reused.

The serial run obtained `J_true=0`, `||m_true||_L2=0.00205378`,
`J_init=2.109e-06`, and `E_m(init)=0.00205378`. Explicit Type-B Lin-More with
L-BFGS history five converged in two iterations to `J=2.02864e-27`,
`||g||=6.36965e-14`, and `E_m=6.37161e-14`, with 9 forward and 5 reverse
evaluations. Initial and final reduced-L2 Taylor remainders were second order;
the fresh final gradient matched exactly. RSS remained below 150 MiB and no
Hessian callback was used. The deterministic repeated run agrees.

The discrete observation extension is an optional mode of the existing
temperature objective; its parsed analytic target behavior remains unchanged.
