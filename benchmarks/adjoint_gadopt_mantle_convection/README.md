# TD-GADOPT: scaled G-ADOPT mantle-convection adjoint benchmark

This directory contains the durable configuration for the benchmark inserted
between O8 and O9. O9 sparse replay remains paused: the reference uses dense
trajectory storage (`Trajectory anchor spacing = 1`, cleanup policy `keep`).

`reference_forward.prm` is a serial 16 by 16 ASPECT smoke case with Q2/Q1
velocity-pressure, continuous Q1 temperature, 80 fixed ASPECT time steps of
`4`, and the G-ADOPT analytic initial-temperature field. Its physical
parameters are `rho = cp = H = alpha = g = eta = 1` and `k = 1e-6`, so
`kappa = k/(rho cp) = 1e-6` and `Ra = rho alpha g H^3/(eta kappa) = 1e6`.

The time and velocity conversions are

```
tau = H^2/kappa = 1e6,  U = kappa/H = 1e-6,
t* = t/tau = 1e-6 t,  u* = u/U = 1e6 u.
```

Thus ASPECT `dt = 4` represents G-ADOPT `dt* = 4e-6`; 80 steps end at
`t = 320`, or `t* = 3.2e-4`.

The target G-ADOPT tutorial uses a 150 by 150 extruded quadrilateral mesh,
CG2/CG1 velocity-pressure, DQ2 temperature, CG1 control, and implicit-midpoint
time integration. This ASPECT benchmark uses a 16 by 16 box mesh, Q2/Q1
velocity-pressure, continuous Q1 temperature, the current reduced Q1-like
initial-temperature control, and ASPECT's validated BDF-family transient path.
The temperature-space difference is intentional and is unrelated to the
velocity-pressure LBB (inf-sup) condition.

The temperature boundaries are strongly fixed to one at the bottom and zero at
the top. Tangential velocity boundary indicators on all four walls implement
the free-slip/impermeable condition. Side temperature boundaries are natural,
as in the reference specification.

`reference_forward_32.prm` is a refinement-5 overlay for the stability-qualified
32 by 32 truth trajectory. It leaves every physical and temporal setting from
the authoritative 16 by 16 input intact. The 16 by 16 Q1 run reached
`max(T80)=1.00647`; the 32 by 32 run retained `0 <= T <= 1` at every archived
step without artificial or entropy viscosity. Future inverse-observation work
uses the 32 by 32 trajectory unless a deliberate coarse-grid study is intended.

中文说明：本目录保存位于 O8 和 O9 之间的 TD-GADOPT 基准配置。O9 的稀疏
重放仍暂停；当前参考算例保留每一个时间步的稠密轨迹。`reference_forward.prm`
使用 16×16 串行网格、Q2/Q1 速度-压力、连续 Q1 温度，并以 ASPECT 的固定
时间步 `4` 运行 80 步。由于 `kappa=1e-6`，该时间步对应 G-ADOPT 的无量纲
`dt*=4e-6`，最终 `t*=3.2e-4`。温度空间的差异是明确记录的设计选择，与
速度-压力的 LBB／inf-sup 稳定性无关。`reference_forward_32.prm` 仅将细化
级别改为 5（32×32）；它消除了 16×16 在第 80 步的温度上冲，且未启用人工黏性。

## How to regenerate the retained reference and inversion evidence

The current source tree no longer contains the temporary G-ADOPT observation
import and accepted-iterate diagnostics. A retained local executable at
`/private/tmp/aspect-td-gadopt-diagnostics` still contains those hooks. The two
`*_legacy_diagnostics.prm` inputs preserve its pre-refactor parameter layout
and use its exact `g-adopt reference` initial-temperature plugin. They are
therefore the reproducibility path for the archived 32 by 32 experiment on
this workstation; they are not inputs for a newly built ASPECT executable.

Regenerate the 81 reference states, extract T70/T80 and u70...u79, run the
12-iteration inversion, and rebuild all enabled figures as follows:

```bash
cd benchmarks/adjoint_gadopt_mantle_convection

/private/tmp/aspect-td-gadopt-diagnostics \
  reference_forward_32_legacy_diagnostics.prm

python3 analysis/extract_reference_observations.py \
  --reference-dir output-adjoint-gadopt-reference-32 \
  --output-dir output-adjoint-gadopt-reference-32/diagnostics

/private/tmp/aspect-td-gadopt-diagnostics \
  inverse_terminal_temperature_10step_32_legacy_diagnostics.prm

conda run -n hang python analysis/make_all_plots.py \
  --output-dir output-adjoint-gadopt-inverse-terminal-10step-32
```

The canonical `inverse_terminal_temperature_10step_32.prm` remains a parameter
record rather than a standalone rerun recipe: the compatibility behavior is
enabled only by the explicitly named legacy input and retained executable.
The numerical record is
`output-adjoint-gadopt-inverse-terminal-10step-32/diagnostics/statistics_adjoint.csv`.
The accepted physical controls, terminal predictions, and figures are under
the neighboring `accepted_controls/`, `terminal_predictions/`, and `plots/`
directories. The CSV is authoritative; PNG figures are a presentation of
those data. Each terminal-only row reports `J_total = J_T`; velocity, damping,
and smoothing columns are intentionally absent until those objective terms
exist.

The authoritative 12-iteration terminal-only diagnostic run produced the
following selected accepted states; the full, one-row-per-accepted-iteration
record is `statistics_adjoint.csv`. `E_T_rel` is the FE-integrated
terminal-temperature error; `E_control_rel` is the physical initial-temperature
L2 error against T70, not the reduced-coordinate metric error.

| accepted iter | J/J0 | E_T_rel | E_u_rel | E_control_rel | \|\|g\|\|_L2 | active lower/upper |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0 | 1.000000 | 0.374900 | N/A | 0.273663 | 0.286105 | 0 / 0 |
| 3 | 0.877528 | 0.351193 | N/A | 0.257148 | 0.268549 | 0 / 0 |
| 6 | 0.010949 | 0.039228 | N/A | 0.037110 | 0.026209 | 0 / 0 |
| 9 | 0.0000218 | 0.001749 | N/A | 0.003230 | 0.000851 | 0 / 0 |
| 12 | 0.000000417 | 0.000242 | N/A | 0.002521 | 0.0000849 | 0 / 0 |

中文说明：G-ADOPT 专用运行期诊断已经从当前核心 Manager 与 ROL 适配器中
删除，但本机保留的 `/private/tmp/aspect-td-gadopt-diagnostics` 可执行文件仍包含
原始观测导入和 accepted-iterate 输出 hook。两个 `*_legacy_diagnostics.prm`
文件与该旧参数树和精确初温插件配套，可以重建 32×32 reference、T70/T80、
u70…u79、12 次反演、VTK 和图片。canonical 的
`inverse_terminal_temperature_10step_32.prm` 仍只是参数记录，不应被视为当前
ASPECT 的独立反演配方。数值证据以
`diagnostics/statistics_adjoint.csv` 为准；`accepted_controls/`、
`terminal_predictions/` 和 `plots/` 分别保存已接受控制、终端预测和图像。
当前仅有终端温度目标，所以 `J_total=J_T`，速度误差 `E_u_rel` 标为 N/A，
而不是伪造为零。当前权威的 12 次迭代结果在第 12 次达到
`J/J0=4.17e-7`、`E_T_rel=2.42e-4` 和 `E_control_rel=2.52e-3`。表中的
`E_T_rel` 与 `E_control_rel` 都是有限元积分 L2 误差；后者在物理温度场中
对比 T70，不应与约化控制度量混淆。
