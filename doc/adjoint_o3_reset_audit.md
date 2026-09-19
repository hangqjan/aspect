# O3 reusable initial-temperature evaluation: reset audit

This audit describes the current one-shot simulator lifecycle before adding an
in-memory, repeated initial-temperature-control evaluation. It is deliberately
an implementation inventory, not a claim that O3 is complete.

本审计描述了在添加内存内、可重复初始温度控制评估之前，当前一次性模拟器的生命周期。
它有意作为实现清单，而不宣称 O3 已完成。

## Current execution boundary

`Simulator<dim>::run()` owns initial mesh setup, initial-condition creation,
the accepted forward time loop, and the terminal reverse call. The initial
condition managers are reset after the first accepted iteration in
`source/simulator/core.cc` (currently lines 2323--2333), so they cannot be
used as a trial-control mechanism on a second evaluation.

`Simulator<dim>::run()` 负责初始网格设置、初始条件创建、被接受的正向时间循环和终端
反向调用。初始条件管理器在 `source/simulator/core.cc`（当前第 2323--2333 行）中首次接受
迭代后被重置，因此不能在第二次评估中被用作试验控制机制。

`InitialTemperatureControl` is currently production-unwired: repository
references are its declaration, implementation, and focused O2 tests only.
In particular, no simulator or adjoint-manager path constructs it, installs a
control, applies `P^T`, or applies the L2 Riesz map.

`InitialTemperatureControl` 当前尚未接入生产路径：仓库中的引用仅包含其声明、实现和聚焦
O2 测试。尤其是，没有模拟器或伴随管理器路径构造它、安装控制、应用 `P^T` 或应用 L2
Riesz 映射。

## Baseline state that must be restored

The canonical pre-forward snapshot must retain owned-only copies of
`solution`, `old_solution`, `old_old_solution`, `current_linearization_point`,
and `system_rhs`. It must also retain `time`, `time_step`, `old_time_step`,
`timestep_number`, and `last_pressure_normalization_adjustment`. These are the
same state components that the current reverse helpers explicitly save and
restore in `source/adjoint/core/manager.cc` (for example the snapshot-replay code
around lines 1636--1655 and the reverse entry around lines 3269--3279).

规范的正向前快照必须保留 `solution`、`old_solution`、`old_old_solution`、
`current_linearization_point` 和 `system_rhs` 的仅所有权副本。它还必须保留 `time`、
`time_step`、`old_time_step`、`timestep_number` 和
`last_pressure_normalization_adjustment`。这与现有反向辅助代码显式保存和恢复的状态一致，
例如 `source/adjoint/core/manager.cc` 中约第 1636--1655 行的快照重放代码，以及约第
3269--3279 行的反向入口。

**RESTORE** means assigning the canonical scalar/vector state before every
trial. **重置（RESTORE）**指在每次试验前重新赋值规范的标量和向量状态。

## State that must be rebuilt

After restoring the baseline and injecting the owned temperature block,
`current_constraints` must be recomputed, initial pressure must be recomputed,
and state-dependent linear solver data must be invalidated. The existing
cold-restart path demonstrates the minimum reconstruction sequence:
`initialize_current_linearization_point()`, `compute_current_constraints()`,
system matrix/preconditioner setup when required, matrix-free `setup_dofs()`,
and setting both Stokes rebuild flags (manager lines 3214--3245).

在恢复基线并注入仅所有权温度块后，必须重新计算 `current_constraints` 和初始压力，并使依赖
状态的线性求解器数据失效。现有冷重启路径展示了最小重建顺序：
`initialize_current_linearization_point()`、`compute_current_constraints()`、必要时的系统矩阵/
预条件器设置、矩阵自由 `setup_dofs()`，以及设置两个 Stokes 重建标志（管理器第
3214--3245 行）。

**REBUILD** means discarding runtime solver objects and reconstructing them
from restored primal state; matrices and preconditioners are not baseline data.
**重建（REBUILD）**指丢弃运行时求解器对象并从恢复后的原始状态重建；矩阵和预条件器不是基线数据。

## State that must be cleared per evaluation

`Manager` owns `objective_results`, `adjoint_states`, `trajectory_store`,
`trajectory_reference_store`, `reverse_state_cache`, `forward_replay_windows`,
`terminal_anchor_history`, sparse-anchor metadata, and the cached initial
gradients (manager header lines 405--451). The current reverse entry clears
only its replay caches (lines 3257--3267), and later recreates terminal
objective results (lines 3557--3575). A reusable evaluation must clear or
recreate all evaluation-owned data before recording a new forward trajectory.

`Manager` 拥有 `objective_results`、`adjoint_states`、`trajectory_store`、
`trajectory_reference_store`、`reverse_state_cache`、`forward_replay_windows`、
`terminal_anchor_history`、稀疏锚点元数据和缓存的初始梯度（管理器头文件第 405--451 行）。
当前反向入口仅清除其重放缓存（第 3257--3267 行），之后才重建终端目标结果（第
3557--3575 行）。可复用评估必须在记录新的正向轨迹前清除或重建所有评估拥有的数据。

**CLEAR** means deleting evaluation-local trajectory, adjoint, objective, and
gradient caches without altering the canonical baseline.
**清理（CLEAR）**指删除评估局部的轨迹、伴随、目标和梯度缓存，而不修改规范基线。

## Explicit O3-v1 exclusions

The existing validation path already restricts replay to a fixed mesh and
rejects unsupported mutable subsystems. O3-v1 must likewise reject AMR,
particles, mesh deformation/free surface, repartitioning, and plugins without
an explicit reset contract. Different-rank restart remains outside O3.

现有验证路径已将重放限制为固定网格，并拒绝不受支持的可变子系统。O3-v1 同样必须拒绝 AMR、
粒子、网格变形/自由表面、重新分区以及没有明确重置契约的插件。不同 rank 的重启仍不属于 O3。

## Required refactoring boundary

O3 needs a new internal forward-evaluation boundary extracted from the
one-shot body of `Simulator::run()`, not a copy of that loop. The boundary
must start from an already prepared canonical state, suppress trial output and
ordinary checkpoint/postprocess side effects, run the accepted forward loop,
and hand the terminal trajectory to a separately callable reverse phase.
Only after that extraction can `Manager` expose an internal
`evaluate_initial_temperature_control(m, need_gradient)`-style operation.

O3 需要从 `Simulator::run()` 的一次性主体中提取新的内部正向评估边界，而不是复制该循环。
该边界必须从已准备好的规范状态开始，抑制试验输出以及常规检查点/后处理副作用，运行被接受的
正向循环，并把终端轨迹交给可单独调用的反向阶段。只有在完成该提取后，`Manager` 才能公开类似
`evaluate_initial_temperature_control(m, need_gradient)` 的内部操作。

## Implemented O3 foundation (not yet a complete evaluation API)

The current implementation captures the canonical vectors, constraints,
time metadata, nonlinear-failure counters, and owned fixed-temperature block
at the post-initialization/pre-forward boundary. Restore clears manager-owned
trajectory/adjoint/objective/kernel caches, restores the canonical fields, and
forces normal matrix and Stokes-preconditioner reconstruction. Control
installation is lazy: it constructs `InitialTemperatureControl` only after a
restore, injects `T_fixed + Pm`, synchronizes the initial history temperature
blocks, and recomputes initial pressure.

当前实现会在初始化后、正向推进前的边界捕获规范向量、约束、时间元数据、非线性失败计数器和
仅所有权固定温度块。恢复会清除管理器拥有的轨迹/伴随/目标/核缓存，恢复规范字段，并强制按正常
路径重建矩阵和 Stokes 预条件器。控制安装采用惰性方式：仅在恢复后构造
`InitialTemperatureControl`，注入 `T_fixed + Pm`，同步初始历史温度块，并重新计算初始压力。

This foundation passed the existing serial one-step temperature-adjoint
benchmark, including its complete forward/reverse sweep and process teardown,
with a 136 MiB peak RSS. It does not yet provide the extracted reusable
forward-loop boundary, a repeat-evaluation driver, or O3 reproducibility/Taylor
tests.

该基础已通过现有串行单步温度伴随基准，包括完整的正向/反向扫描和进程清理，峰值 RSS 为
136 MiB。它尚未提供提取出的可复用正向循环边界、重复评估驱动，或 O3 可重复性/Taylor 测试。

## O3 block-vector interface audit and D1--D5 localization

ASPECT first applies `DoFRenumbering::component_wise()` in
`Simulator::setup_dofs()` and then calls `IndexSet::split_by_block()` in
`Simulator::setup_introspection()` using `system_dofs_per_block`. Therefore
`AffineConstraints` and FE assembly use mixed global DoF indices, while
`solution.block(temperature_block)` uses the corresponding temperature-block
index space. The explicit conversion is the cumulative size of preceding
blocks; it is an ASPECT numbering contract established by those two source
functions, not a guessed offset.

ASPECT 首先在 `Simulator::setup_dofs()` 中应用
`DoFRenumbering::component_wise()`，随后在 `Simulator::setup_introspection()` 中使用
`system_dofs_per_block` 调用 `IndexSet::split_by_block()`。因此，`AffineConstraints` 和 FE
装配使用混合全局 DoF 索引，而 `solution.block(temperature_block)` 使用相应的温度块索引空间。
显式转换是此前各块大小的累积和；它是由这两个源函数确立的 ASPECT 编号契约，而不是猜测出的偏移量。

`InitialTemperatureControl` now retains mixed-global indices for constraint
relations, sparse reduced numbering, FE mass assembly, and P/P^T. It also
caches the temperature-block prefix and block owned/relevant `IndexSet`s. Field
injection and raw-covector restriction translate only at this boundary. Mapped
physical and homogeneous affine constraints are applied in temperature-block
coordinates, so no mixed-system index is used to access an 81-entry temperature
block.

`InitialTemperatureControl` 现在为约束关系、稀疏降阶编号、FE 质量矩阵装配和 P/P^T 保留混合全局
索引。它也缓存温度块前缀以及该块的 owned/relevant `IndexSet`。场注入和原始协变量限制只在这个
边界进行转换。映射后的物理与齐次仿射约束在温度块坐标中应用，因此不会再使用混合系统索引访问
81 项温度块。

The post-run stepwise diagnostic found two independent D5 prerequisites. First,
the original reset rejected ASPECT's default stateless convection time-step
plugin and Debug exception formatting grew past the memory guard. The reset now
accepts only the stateless built-in convection/conduction estimators and still
rejects mutable time-step plugins. Second, the mixed/block vector mismatch was
corrected as described above. Serial D1--D5 now pass with peaks of roughly
137--145 MiB; D5 completed physical installation, initial-pressure refresh,
and linearization refresh. The ordinary forward/reverse regression also passed
afterwards at roughly 140 MiB with terminal objective
`0.004837537772946448`.

运行后逐步诊断发现 D5 有两个相互独立的前置条件。首先，原始重置拒绝了 ASPECT 默认的无状态对流
时间步插件，而 Debug 异常格式化使内存增长超过保护阈值。重置现在仅接受无状态的内建对流/传导
估算器，并继续拒绝可变时间步插件。其次，混合/块向量不匹配已按上述方式修正。串行 D1--D5 现在
均通过，峰值约为 137--145 MiB；D5 完成了物理安装、初始压力刷新和线性化刷新。其后普通正向/
反向回归也以约 140 MiB 通过，终端目标值为 `0.004837537772946448`。

## O3 completion record

O3 is complete. The reusable `m0 -> m1 -> m0 -> m1 -> m0` sequence passed
both serially and on two ranks with exact repeated objectives, raw reduced
covectors, and L2 Riesz gradients. The two-rank reference objectives were
`0.004837537772950052` for `m0` and `0.004924298646014869` for `m1`; the
maximum observed worker RSS was about 143 MiB. The serial and MPI Taylor tests
showed second-order remainders, and fixed-K sparse replay passed with
state/pressure errors around `4.5e-14` and recovered-velocity error around
`6.3e-9`, within the established `1e-8` Stokes-recovery tolerance.

O3 已完成。可复用的 `m0 -> m1 -> m0 -> m1 -> m0` 序列在串行和双进程下均通过，重复目标值、
原始降阶协变量和 L2 Riesz 梯度完全一致。双进程的参考目标值为：`m0` 的
`0.004837537772950052` 和 `m1` 的 `0.004924298646014869`；观察到的单工作进程峰值 RSS 约为
143 MiB。串行和 MPI Taylor 测试均显示二阶余项；固定 K 稀疏回放也通过，其中状态/压力误差约为
`4.5e-14`，恢复速度误差约为 `6.3e-9`，满足既有 `1e-8` Stokes 恢复容差。

The final O2 two-rank lifetime investigation used an explicit inner scope
around the exact production control, mass, Riesz, and ROL-vector objects.
That non-Catch harness passed before MPI finalization. The earlier allocator
failure was then eliminated by an ABI-clean rebuild of every focused object
that included modified control-space headers; the two-rank Catch O2 test,
the O1 distributed-vector test, and the serial O0/O1/O2 stack now all pass.
Physical injection also explicitly reapplies inhomogeneous affine rows using
owned-only point-to-point imports when a remote master is required.

最终的 O2 双进程生命周期调查为生产控制、质量矩阵、Riesz 和 ROL 向量对象设置了显式内部作用域；
该非 Catch 程序在 MPI 结束前正常通过。此前的分配器失败随后通过对所有包含已修改控制空间头文件的
聚焦对象进行 ABI 干净重建而消除；双进程 Catch O2 测试、O1 分布式向量测试以及串行 O0/O1/O2
测试栈现已全部通过。物理注入还会在需要远端主自由度时，使用仅所有权的点对点导入显式重施加非齐次
仿射行。
