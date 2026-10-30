# O8: qualified two-rank MPI optimization

O8 qualifies the existing O6 production ROL Type-B Lin--More optimizer and
the O7 synthetic terminal-temperature twin on two MPI ranks. No production
source change was required during this milestone.

## Distributed layout and collective audit

The reduced initial-temperature control has global dimension 63. The two ranks
own 36 and 27 coefficients, respectively; the reduced mass matrix is 63 by
63. Controls, bounds, and ROL vectors are owned-only distributed vectors, and
the terminal observation is a copied distributed ASPECT `BlockVector`.

The deterministic truth and Taylor direction are both functions of the global
control index. The O5 cache equality check reduces the local mismatch flag over
the vector communicator before making a PDE-execution decision, ensuring that
all ranks take the same cache branch.

## Numerical qualification

The 2-rank truth self-check gave `J(m_true) = 0`. Initial quantities were
`J = 2.1090010703053379e-06`, reduced-L2 gradient norm
`0.0020537769640221067`, and control error `0.0020537780949402007`.
The initial distributed Taylor order was `2.0000000000416982`.

With the standard O7 bounds `[-1e-2, 1e-2]`, the one-iteration result was
`J = 5.5522383095709878e-07`, gradient norm
`0.0010537775146811992`, and control error `0.0010537780949417515`.
Both ranks reported the same callbacks, cache counts, and L-BFGS history.

The full twin converged in two iterations on both ranks:

```
J                         = 2.2523077848639189e-27
reduced-L2 control error  = 6.7126609016141431e-14
gradient norm             = 6.7116097391364445e-14
forward / reverse solves  = 9 / 5
L-BFGS pairs              = 2
updates Initial/Trial/Revert/Accept = 3 / 2 / 0 / 2
fresh value difference    = 0
fresh gradient difference = 0
final Taylor order        = 2.0000000000004974
```

The deterministic second run was exactly identical in these quantities. Its
control difference from the serial O7 solution is bounded by the sum of the
two reported truth errors, below `1.31e-13` in the reduced-L2 norm.

With history cap two, both ranks retained exactly two L-BFGS pairs. With tight
increment bounds `[-1e-3, 1e-3]`, the two ranks returned the same feasible
active-bound solution in one iteration, with projected gradient norm zero.

## Memory and kernel regression

All MPI runs used one process group at a time and direct worker-PID RSS
monitoring with a 1 GiB stop threshold. No worker exceeded the expected scale.
The full twin peaks were 143,664 and 135,968 KiB; the deterministic repeat was
143,600 and 136,048 KiB. There is no per-iteration memory growth over the two
accepted updates of this fixture.

Post-O8 kernel mode remains unchanged:

```
serial kernel J = 0.004837537772946448
2-rank kernel J = 0.004837537772950052
```

The difference is roundoff-level (`3.6e-15`).

## 中文说明

O8 已完成双 rank MPI 优化资格验证。约化控制的全局维数为 63，两个
rank 分别拥有 36 和 27 个系数；控制、边界、质量矩阵和终端观测都保持
分布式布局，没有引入全局控制向量复制。缓存相等性会进行全通信器归约，
因此所有 rank 必定在同一轮中作出相同的缓存命中或未命中决定。

完整 synthetic twin 在两次迭代后收敛，最终目标函数为
`2.2523077848639189e-27`，控制误差为 `6.7126609016141431e-14`；两次
MPI 完整运行的结果完全一致。每个工作进程的 RSS 均处在约 136--144 MiB
的正常范围内。串行和双 rank kernel 回归均保持 O7 参考值，仅有浮点舍入
级别差异。
