# ASPECT 密度定义与横向黏度研究方案：源码调研

日期：2026-10-27。检查对象：`/Users/hangqian/projects/aspect` 当前工作副本，HEAD `6969d30bb`。工作副本已有未提交修改；本报告描述实际读到的文件，不将 HEAD 当成全部实现的唯一版本标识。

## 1. 范围与结论

逐项检查了 `source/material_model/` 中注册的 **28 个内置 material model**，并追踪共享 equation of state（EOS）、组分混合和相关材料梯度代码。另扫描了仓库其余目录的材料插件注册：发现 **95 处注册声明**，主要属于测试、benchmark 和 cookbook。这些并非默认内置模型；附录给出索引，不声称逐一验证了它们的全部物理行为。仓库外的用户动态库无法由本次扫描穷尽。

主要结论：

1. ASPECT 没有统一的 `density = f(temperature)` 规定。插件返回 `rho(T,p,C,x,...)`，或者查表/外部模型给出的结果；有些模型以熵场为热力学变量。
2. `out.densities` 通常是材料总密度；密度异常、参考态、固相密度、混合物密度和浮力系数不能互相替代。
3. 横向黏度无需靠优化温度实现。固定温度/层析结构，优化径向黏度、温度敏感度、板片/克拉通/弱带强度，就已超出径向模型的能力。
4. 优化温度同时改变密度和黏度。它研究的是热结构与流变共同作用，而非给定密度下的纯黏度识别。
5. 有材料输出核，不等于对所有 material model 的所有控制量已有可靠梯度。当前局部 VJP 的依赖范围、点间耦合和参数映射均有限制。

## 2. 材料接口与 EOS 分类

材料接口接受温度、压力、组分、坐标、应变率等输入，填充密度、黏度和热物性。接口不强制所有输入影响所有输出；`NonlinearDependence::ModelDependence` 声明各输出的依赖，用于程序判断。

源码：[材料接口](/Users/hangqian/projects/aspect/include/aspect/material_model/interface.h:369)。

记号：`z` 为深度；`p_s` 为地表压力；`T_ad(x),p_ad(x)` 为绝热参考态；`f_j` 为体积分数，`w_j` 为质量分数。下面各公式描述当前代码，不把不同模型的参考温度统一成同一个数。

### A. 共同线性 EOS

`LinearizedIncompressible` 使用

\[
\rho_j=\rho_0[1-\alpha(T-T_0)]+\Delta\rho_j,\qquad \Delta\rho_0=0.
\]

热膨胀部分对各组分相同，组分密度差是加法项，并不乘热膨胀因子。`is_compressible=false` 不等于密度不随温度或组分变化；它描述质量守恒近似中是否考虑可压缩性。

源码：[LinearizedIncompressible](/Users/hangqian/projects/aspect/source/material_model/equation_of_state/linearized_incompressible.cc:47)。

| # | 内置模型 | 密度规定与实际控制变量 | 源码 |
|---|---|---|---|
| 1 | `simpler` | 单组分线性 EOS；只依赖 T，不读取组分作为密度来源；黏度为常数 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/simpler.cc:39) |
| 2 | `simple` | 同一 EOS，最多第一组分影响密度；先将该组分下限截为 0，再以背景份额和该份额算术混合。通常区间内为 `rho0[1-alpha(T-T0)]+C1*delta_rho1`；不要假定代码会把 C1 上限截到 1 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/simple.cc:34) |
| 3 | `composition reaction` | 同一 EOS，最多前两个组分影响密度；背景与两个组分算术混合，反应另行演化这些场 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/composition_reaction.cc:35) |
| 4 | `drucker prager` | 密度来自单组分线性 EOS；塑性/压力/应变率依赖主要出现在黏度中，不能据模型名称推断密度也使用塑性公式 | [密度赋值](/Users/hangqian/projects/aspect/source/material_model/drucker_prager.cc:156) |

对固定参考态、未触及组分截断的单组分线性密度，`partial rho/partial T=-rho0*alpha`。这是导数最清楚的起点。

### B. 多组分不可压缩 EOS

`MulticomponentIncompressible` 为每个组分/相设置独立参考密度和热膨胀系数：

\[
\rho_j=\rho_{0j}[1-\alpha_j(T-T_{ref})].
\]

若开启绝热加热，`T_ref=T_ad(x)`；否则用参数 `Reference temperature`。它与 A 类的参考温度选择不同。

源码：[多组分 EOS](/Users/hangqian/projects/aspect/source/material_model/equation_of_state/multicomponent_incompressible.cc:34)。

| # | 内置模型 | 密度规定与混合规则 | 源码 |
|---|---|---|---|
| 5 | `multicomponent` | `rho=sum f_j*rho_j`；化学组分按体积分数解释；黏度独立作指定形式的混合 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/multicomponent.cc:35) |
| 6 | `visco plastic` | 同上；额外先以相函数混合各组分的各相密度，再按组分体积分数混合。相函数可依赖 T、p、深度和参考密度，所以即便单相 EOS 不显式依赖 p，总密度仍可能通过相变依赖 p | [相混合与密度](/Users/hangqian/projects/aspect/source/material_model/visco_plastic.cc:134) |
| 7 | `viscoelastic` | 同一多组分 EOS 和体积算术混合；弹性记忆/应力场影响本构应力，不能直接当作独立的密度来源 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/viscoelastic.cc:35) |
| 8 | `diffusion dislocation` | 自行实现 `rho=sum f_j*rho0j[1-alphaj(T-T0)]`；密度无直接压力项，黏度可依赖 T、p、组分及应变率 | [密度循环](/Users/hangqian/projects/aspect/source/material_model/diffusion_dislocation.cc:35) |

多组分的 `compute_only_composition_fractions` 会按化学组分字段构造背景份额并处理分数范围；它与 `simple` 手写的第一组分混合并非完全相同的程序路径。不同参考密度、不同 alpha 的体积分数混合是这些模型采用的近似，不应自行换成质量分数混合。

### C. 解析可压缩 EOS、深度参考剖面

| # | 内置模型 | 当前密度公式/算法 | 源码 |
|---|---|---|---|
| 9 | `simple compressible` | `rho=rho0*exp[kappa(p-p_s)]*[1-alpha(T-T_ad(x))]`；无组分密度项；可压缩性随 kappa 是否为 0 判定 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/simple_compressible.cc:34) |
| 10 | `multicomponent compressible` | 每相使用下述 Murnaghan 型 EOS；先相混合，再把化学质量分数转为体积分数，最终作体积算术密度混合 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/multicomponent_compressible.cc:36) |
| 11 | `modified tait` | `rho=rho0/x`，`x=1-a[1-(1+b(p-p0-Pth))^(-c)]`；热压力 Pth 来自 Einstein 热能；非线性 T、p 依赖，忽略组分 | [Tait 实现](/Users/hangqian/projects/aspect/source/material_model/modified_tait.cc:58) |
| 12 | `ascii reference profile` | 从深度表读 `rho_ref(z),alpha(z),kappa(z)`，计算 `rho=rho_ref(z)[1-alpha(z)(T-T_ad)][1+kappa(z)(p-p_ad)]`；TALA 分支去掉最后压力因子 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/ascii_reference_profile.cc:65) |
| 13 | `nondimensional` | `rho=rho_ref*exp(z*Di/gamma)*[1-alpha_nd*(T-T_ad)+I_nonTALA*Di*gamma*(p-p_ad)]`，`alpha_nd=Di`（可压缩）或 1（不可压缩）；用于无量纲方程检验，不能直接把输出解释为地球 SI 密度并代入 Geoid | [evaluate](/Users/hangqian/projects/aspect/source/material_model/nondimensional.cc:61) |

第 10 项每相的实际 EOS 为：

\[
F_j=1+K'_j\kappa_{0j}\left[p_*-
\frac{\alpha_{0j}}{\kappa_{0j}}(\max(T,1)-T_{0j})\right],
\quad \rho_j=\rho_{0j}F_j^{1/K'_j}.
\]

通常 `p_*=in.pressure`；在存在 density 类型字段且采用 projected density field 质量守恒时，改用绝热压力。相函数仍是额外的依赖链。

源码：[可压缩多组分 EOS](/Users/hangqian/projects/aspect/source/material_model/equation_of_state/multicomponent_compressible.cc:36)。

### D. 相变与平衡熔融的解析模型

| # | 内置模型 | 当前密度公式/算法 | 源码 |
|---|---|---|---|
| 14 | `latent heat` | 通常 `rho=[rho0+delta_rho_C+rho0*kappa(p-p_s)+sum delta_rho_k*Gamma_k]*[1-alpha(T-T_ad)]`。Boussinesq 分支改为热项、组分项、相变项线性相加，去掉压力项。相函数 Gamma 可依赖 T、p、深度和组分权重 | [密度组合](/Users/hangqian/projects/aspect/source/material_model/latent_heat.cc:101) |
| 15 | `latent heat melt` | `rho=[rho0+delta_rho_C+rho0*kappa(p-p_s)]*F_T*[1-(1-relative_melt_density)*F_melt(T,p,C,x)]`。开启绝热加热时 `F_T=1-alpha(T-T_ad)`，否则代码为 `1-alpha*T`，不是统一的 `T-T0` | [evaluate](/Users/hangqian/projects/aspect/source/material_model/latent_heat_melt.cc:85) |

`latent heat` 的相变密度体积分布已经在材料密度中体现。若另加同一相变界面的质量薄层，必须先定义明确的线性参考态，避免重复计算。

### E. 热力学查表、矿物物理与熵

| # | 内置模型 | 密度来源与重要实现细节 | 源码 |
|---|---|---|---|
| 16 | `Steinberger` | 热力学 T-p 表给出各材料密度；求 EOS 前主动把输入压力换成 `p_ad(x)`；化学质量分数转体积分数，再体积算术混合；表可以包含矿物相变 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/steinberger.cc:266) |
| 17 | `grain size` | 不查表时 `rho=rho0*exp[kappa(p-p_s)]*[1-alpha(T-T0)]`；查表时单表直接取值，多表按代码中的 compositional_fields[i] 线性加权，并非 Steinberger 的质量到体积转换。其 evaluate 在绝热条件初始化后传入绝热压力，初始化前回退输入压力。粒径主要通过流变影响黏度，不能据名称认为密度直接依赖粒径 | [density](/Users/hangqian/projects/aspect/source/material_model/grain_size.cc:313)、[evaluate 调用](/Users/hangqian/projects/aspect/source/material_model/grain_size.cc:526) |
| 18 | `perplex lookup` | 调用 Perple_X/meemum 的 `phaseq`，使用返回 `sysprop[9]` 作为密度。当前 evaluate 先平均这一批输入点的 T、p、C，只调用一次，再向所有点填同一密度；需要编译 Perple_X 支持 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/perplex_lookup.cc:57) |
| 19 | `entropy model` | 每组分由熵字段 S_j 与绝热压力查表 `rho_j(S_j,p_ad)`，压力转为 bar；化学质量分数转体积分数并混合。温度由熵表/热平衡构造；不能默认独立改变 `in.temperature` 就能控制密度 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/entropy_model.cc:229) |

共享 `ThermodynamicTableLookup` 只读取调用者提供的 T、p；具体是否换成绝热压力由调用者决定。表插值、范围处理和相变跃迁决定实际光滑性。

源码：[表 EOS](/Users/hangqian/projects/aspect/source/material_model/equation_of_state/thermodynamic_table_lookup.cc:319)、[查表入口](/Users/hangqian/projects/aspect/source/material_model/utilities.cc:55)。

### F. 显式两相/熔体模型

| # | 内置模型 | 密度来源与固相/液相区别 | 源码 |
|---|---|---|---|
| 20 | `melt simple` | `out.densities` 为固相：`[rho_s0+delta_rho_depletion*C_peridotite]*F_T*exp[kappa_s(p-p_s)]`；绝热加热分支使用 T-T_ad，否则 T-T0。液相密度另由 Katz2003 模块填到 MeltOutputs | [固相](/Users/hangqian/projects/aspect/source/material_model/melt_simple.cc:80)、[流体输出](/Users/hangqian/projects/aspect/source/material_model/reaction_model/katz2003_mantle_melting.cc:302) |
| 21 | `melt global` | 固相公式同类：`[rho_s0+delta_rho_depletion*C_peridotite]*F_T*exp[kappa_s(p-p_s)]`；流体密度单独用 `rho_f0*F_T*exp[kappa_f(p-p_s)]` | [固相](/Users/hangqian/projects/aspect/source/material_model/melt_global.cc:135)、[流体](/Users/hangqian/projects/aspect/source/material_model/melt_global.cc:281) |
| 22 | `melt boukare` | 从端元热力学状态、摩尔质量、摩尔体积和相平衡计算；固相 `rho_s=M_s/V_s`，液相 `rho_f=M_f/V_f`；存在固相时 out.densities 返回固相，完全无固相时回退液相，MeltOutputs 另给流体密度 | [相混合](/Users/hangqian/projects/aspect/source/material_model/melt_boukare.cc:663)、[密度输出](/Users/hangqian/projects/aspect/source/material_model/melt_boukare.cc:724) |

这些模型的 `out.densities` 不能普遍看成总质量分布。两相总体积密度应按定义使用 `rho_bulk=(1-phi)rho_s+phi*rho_f`。当前普通/自引力 Geoid 的体积分读取材料 `out.densities`，未见在该积分路径上自动组合 phi 与流体密度。因此把 Geoid 直接用于两相模型前需另作物理和实现审查。

### G. 委托/组合模型

| # | 内置模型 | 密度如何确定 | 源码 |
|---|---|---|---|
| 23 | `depth dependent` | 原样继承 base model 密度；仅给 base viscosity 乘深度因子；可叠加在温度依赖黏度上 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/depth_dependent.cc:111) |
| 24 | `prescribed viscosity` | 原样继承 base density；在 indicator>0.5 区域用空间/时间函数替换黏度；横向变化不要求温度变化 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/prescribed_viscosity.cc:54) |
| 25 | `replace lithosphere viscosity` | 原样继承 base density；仅按岩石圈底界深度替换黏度 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/replace_lithosphere_viscosity.cc:45) |
| 26 | `compositing` | density 由 `Density` 指定的子模型提供，viscosity 可取另一个子模型；其他热物性也可分别指定，不自动保证热力学一致性 | [属性复制](/Users/hangqian/projects/aspect/source/material_model/compositing.cc:60) |
| 27 | `averaging` | 先调用 base，然后对本批输入点返回的密度以及其他物性做平均/平滑；不是新 EOS，且存在点间耦合 | [evaluate](/Users/hangqian/projects/aspect/source/material_model/averaging.cc:328) |
| 28 | `reactive fluid transport` | 固相密度继承 base；非 Katz 分支另输出 `rho_f=rho_f0*exp[kappa_f(p-p_s)]` 并处理流体反应与孔隙弱化；Katz 分支委托相应反应模块；内部另有 bulk density 计算，不表示全局覆盖 out.densities | [evaluate](/Users/hangqian/projects/aspect/source/material_model/reactive_fluid_transport.cc:175)、[bulk helper](/Users/hangqian/projects/aspect/source/material_model/reactive_fluid_transport.cc:61) |

## 3. 密度混合、参考态和 Geoid 必须核对的区别

### 3.1 质量分数与体积分数

体积分数输入：`rho_mix=sum f_j*rho_j`。

质量分数输入：`f_j=(w_j/rho_j)/(sum_k w_k/rho_k)`，于是 `rho_mix=1/(sum_j w_j/rho_j)`。

后者的体积分数本身依赖 T、p，求导不能只对单个 rho_j 求导而冻结 f_j。相函数同理。

源码：[质量到体积](/Users/hangqian/projects/aspect/source/material_model/utilities.cc:955)。

### 3.2 压力不是统一的一种用法

- `simple`/`simpler` 的密度不依赖压力。
- `simple compressible`/`modified tait` 等直接用输入压力。
- `Steinberger`/`entropy model` 上述路径用绝热压力；`grain size` 初始化前回退输入压力。
- 某些模型的 projected density approximation 改变 EOS 使用的压力并把密度写入指定字段。
- 有相变时，即使单相密度 EOS 不可压缩，密度仍可能通过相函数依赖压力。

若材料密度包含动态压力依赖，压力参考值和压力规范会成为物理设置的一部分，不能套用不可压缩模型中“压力任意加常数”的直觉。

### 3.3 四种密度不要混为一谈

1. 材料 EOS 返回的 rho。
2. 绝热/参考剖面 rho_ref(r)。
3. Stokes 浮力项实际使用的密度或有效系数，取决于 formulation、melt、projected density、averaging 等路径。
4. Geoid 体积分实际读取的密度。

当前 Geoid 投影直接读取材料密度，未显式减 rho_ref。完整球壳上球对称背景在 l>=2 连续积分中为零，但离散网格存在数值泄漏。两相模型则还存在固相与混合物密度差别。

源码：[普通 Geoid 密度积分](/Users/hangqian/projects/aspect/source/postprocess/geoid_harmonics.cc:294)、[自引力密度矩](/Users/hangqian/projects/aspect/source/postprocess/self_gravitating_geoid_operator.cc:486)。

此外存在两种不同平均：material model `averaging` 在插件内部平均；顶层 `Material averaging` 在组装时还可再次平均，部分选项只平均黏度。Geoid 直接 evaluate 的数据与 Stokes 组装后的系数未必相同。必须明确所用路径及其导数。

## 4. 与本任务直接相关的额外 cookbook 插件

`tomography based plate motions` 已有与“给定层析密度、允许横向流变”非常接近的实现。

- 无查表分支的深部密度用 `rho=rho_ref[1+R_rhoVs(z)*delta_log_Vs]`，常数转换默认分支为 0.15，也可读取深度相关转换文件。
- 地壳/岩石圈使用专门密度规则，Slab2 开启后还会替换部分密度来源。
- 查表分支用其构造的 new_temperature 与压力求密度。
- 黏度包括温压、扩散/位错蠕变、粒径、断层弱带、克拉通等；还通过横向平均剖面缩放，使其匹配参考径向黏度。

它证明“密度由层析给定，黏度另有三维变化”在 ASPECT 中已有先例。但其非线性和全局平均更新远超当前材料局部 VJP 的简单适用范围，不能直接宣称已经支持该插件的可靠优化。

源码：[层析密度](/Users/hangqian/projects/aspect/cookbooks/tomography_based_plate_motions/plugins/tomography_based_plate_motions.cc:1177)、[横向平均与缩放](/Users/hangqian/projects/aspect/cookbooks/tomography_based_plate_motions/plugins/tomography_based_plate_motions.cc:199)。

## 5. 当前梯度能力的实际边界

### 5.1 物性核与控制变量梯度不同

将总目标变化写成连续形式：

\[
\delta J=\int_\Omega(K_\rho\delta\rho+K_\eta\delta\eta)\,dV
 +\text{其他物性/边界/状态项}.
\]

这里 K_rho、K_eta 指已包含所需直接与间接通路的总物性导数。对于局部、光滑且参考态固定的本构律，才有

\[
K_T=K_\rho\rho_T+K_\eta\eta_T+\cdots.
\]

如果采用 `q=ln(eta/eta_*)` 控制，则 K_q=eta*K_eta。径向层参数、空间基函数参数、流变系数还需要自己的链式映射；已有物性核并不自动意味着这些参数已经接入优化器。

对时间演化问题，以上是某一时刻的局部本构链；初始温度梯度还必须穿过温度/组分输运的反向时间传播。不能把瞬时温度梯度称为初始温度梯度。

### 5.2 已读到的材料 VJP

`MaterialModelAdjoint` 对 `eta,rho,cp,k` 输出种子构造标量，再对输入 T、p、C 作中心有限差分。步长为 `1e-7*max(1,abs(input))`。它依赖模型声明的 NonlinearDependence，并按评价点逐点计算。

已明确拒绝：非零黏度种子所需的应变率依赖、压力依赖。代码说明当前这一层审查范围为 eta(T,C)，并非任意 eta(T,p,C,strain_rate)。压力依赖密度的局部导数有计算路径，但上层是否支持压力反馈和 scatter 仍须逐工作流验证。

源码：[限制条件](/Users/hangqian/projects/aspect/source/adjoint/operators/material_model_adjoint.cc:175)、[点局部有限差分](/Users/hangqian/projects/aspect/source/adjoint/operators/material_model_adjoint.cc:238)。

因此：

- 线性密度 + 平滑 eta(T,C) 最适合作为当前首个验证模型。
- `visco plastic`、`diffusion dislocation` 等虽然可正演，不能直接推断其完整非线性优化已支持。
- `averaging` 与 `perplex lookup` 有跨评价点依赖。当前 VJP 只收缩同一点输出，不能完整覆盖 `sum_i bar_y_i*partial y_i/partial x_q` 的所有 i。通用完整梯度未由这段实现保证。
- 横向平均、update 缓存、参考剖面重建、材料参数文件的变化，不在只扰动 `in.T/p/C` 的局部 VJP 内自动完成。
- 分数截断、黏度上下限、分段指示函数、离散相变可能导致不可导或导数在不同步长下不稳定；查表模型也要检查实际插值光滑性。
- 对材料参数（例如 Thermal viscosity exponent、每层参考黏度）的导数，需单独实现参数到物性的映射。局部 VJP 本身没有扰动这些参数。

这不是说全部这些模型一定无法优化，而是说必须补齐对应导数和独立验证，不能因模型接口相同就视为已完成。

## 6. 如何发挥横向黏度能力

### 6.1 推荐首选：固定密度/热结构，优化低维三维流变

首先固定层析给定的 density；温度可以来自一个固定的地震学/热学参考结构，也可直接用固定层析场构造黏度模板。使用：

\[
\ln\frac{\eta(x)}{\eta_*}
=a(r)+\beta(r)Q_T(x)
+b_{slab}M_{slab}(x)+b_{craton}M_{craton}(x)
+b_{weak}M_{weak}(x).
\]

- a(r)：少数径向层或径向样条参数。
- Q_T：固定热结构形成的冷/热异常模板，如 `-(T_fixed-mean_T_fixed(r))/DeltaT_*`。
- beta(r)：温度相关横向黏度对比的幅度，可只用一两个深度区间。
- M：固定板片、克拉通、板块边界等空间掩膜；b 控制其强度。

这仍是固定 density 下的黏度问题，且能研究三维黏度如何改变流动和边界应力。不需要先开放三维温度的每个自由度。

每层模板可以去均值，让 a 定义层的平均 log 黏度，避免横向幅度改变时同时偷偷移动径向参考值。例如 `Q_T <- Q_T-<Q_T>_layer`；此时 eta 的几何平均由 a 控制。若做指数后归一化以固定算术平均，则必须对归一化本身求导。黏度裁剪也会破坏原有平均约束。

当前可复用 `depth dependent`、`prescribed viscosity`、`compositing` 的正演机制，但上式参数的优化接线尚需实现。建议采用连续基函数、对数正黏度参数，不用移动硬阈值界面作为首个控制。

### 6.2 随后才开放温度结构修正

如果科学问题是“怎样的热结构和流变共同解释观测”，采用

\[
T(x)=T_{prior}(x)+\sum_k t_k\Phi_k(x),
\quad \rho=\rho(T,p,C),\quad\eta=\eta(T,p,C;a,b).
\]

先让 Phi_k 为少量平滑、低阶/区域基函数；逐步增加自由度。优化温度时，密度直接 Geoid 路径和黏度应力路径必须都保留，不能冻结其中一项后仍称为完整温度梯度。

要让温度具有地球物理含义，应使用与层析相容的约束：例如同一矿物物理模型给出 Vs(T,p,C)，再与 Vs_obs 比较；同时对温度修正施加幅度与光滑约束。低 Vs 不必全是高温，成分、相变和其他效应不能默认不存在。初期可固定 C，把它的不确定性作为多套先验/情景；后期有独立数据时再联合优化少量成分参数。

不建议同时自由优化不受约束的三维 rho、eta、T 并又强加 rho(T)、eta(T) 而不说明变量关系；可选独立物性场加本构罚项，或由少数热化学变量生成物性，二者是不同问题。

### 6.3 为什么仅 Geoid 不足以支撑任意三维温度/黏度恢复

球谐 l=2..L 的实独立 Geoid 系数只有 `(L+1)^2-4` 个：L=3 为 12，L=7 为 60。百万个网格控制即使都有梯度，数据也不因此增加；平滑正则化能选解，但不能凭空提供分辨率。

全局 Geoid 对横向黏度的响应依赖密度模型和黏度模板，不能假定三维能力必定提升 Geoid 拟合或可辨识性。应逐步计算参数的加权预测变化/Jacobian，检查相似列、奇异值和对噪声的响应。

可用的互补数据包括动态地形、板块速度/表面流动、与热结构一致的地震学约束；它们的误差、参考系、边界条件和非地幔成因需按具体观测处理。若把板块速度当作规定边界条件，就不能又将同一规定值当作独立预测数据获得虚假的约束。

齐次 free-slip、固定密度、线性牛顿瞬时模型仍有整体黏度尺度退化：eta 全部乘 a，速度除 a，应力与 Geoid 可不变。需固定一个参考黏度，或通过合适的独立速度/时间信息约束尺度。

### 6.4 建议的可执行研究顺序

1. 先解决上一轮发现的普通/自引力地形符号约定、CMB 重力处理，完成径向单一球谐正演基准。
2. 固定 density，用径向几层 log 黏度建立可比基准，固定绝对尺度。
3. 固定三维温度/层析模板，只增加一个横向黏度强度 beta；测试其对 Geoid、地形和速度的预测变化及梯度。
4. 再增加一个有独立地质依据的区域强度（例如板片），在数据误差尺度上检查可辨识性；与径向参数联合重估，避免把径向误差归因于横向变化。
5. 有互补约束后，开放少量温度修正参数，同时计算 rho_T 与 eta_T 两条通路；用合成恢复与交叉验证检验新增自由度。
6. 对任一新材料路径做完整目标的方向导数/Taylor 检查，验证所用参数映射及平均/裁剪操作；再扩大控制维度。

## 7. 外部文献核对

- Rudolph et al. (2020), §2.1：HC、propagator matrix、两端 free-slip、径向黏度；全文讨论低阶 Geoid 约束与密度模型选择的影响。[论文](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020GC009335)
- Ghosh, Becker & Zhong (2010)：横向黏度的 Geoid 影响随密度模型改变；分别用温度依赖与构造区域指定黏度引入 LVV，并讨论板块运动联合约束。支持“先固定结构、优化横向强度”的研究设计，不意味着其结果可以不经检验直接移植到本代码。[论文](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2009GL040426)
- ASPECT 官方材料参数手册：用于交叉检查模型和参数名称；本报告中具体公式以当前工作副本源码为准。[手册](https://aspect-documentation.readthedocs.io/en/latest/parameters/Material_20model.html)

## 8. 验证范围

本次完成源码静态检查、注册项清单核对和文献核对；没有修改物理实现，没有运行新的 ASPECT 正演、优化或梯度测试。报告中的“已实现”指发现明确代码路径，不表示这些路径已经通过新的跨程序数值验证。研究方案与需要新增的参数/导数映射已与现有能力分开说明。

## 附录：其余目录的材料插件注册索引

以下仅为源码扫描发现的注册声明，不等同于当前进程已加载的插件，也不等同于独立物理模型数；存在同名声明和基类包装。

| 注册名称 | 文件 |
|---|---|
| `adjoint cellwise perturbation` | [benchmarks/adjoint_dynamic_topography_2d_spherical/cellwise_perturbation.cc](/Users/hangqian/projects/aspect/benchmarks/adjoint_dynamic_topography_2d_spherical/cellwise_perturbation.cc:84) |
| `adjoint cellwise perturbation` | [benchmarks/adjoint_dynamic_topography_3d_spherical/cellwise_perturbation.cc](/Users/hangqian/projects/aspect/benchmarks/adjoint_dynamic_topography_3d_spherical/cellwise_perturbation.cc:84) |
| `adjoint material model` | [benchmarks/adjoint_time_dependent_temperature/plugin/adjoint_material_model.cc](/Users/hangqian/projects/aspect/benchmarks/adjoint_time_dependent_temperature/plugin/adjoint_material_model.cc:316) |
| `adjoint pressure material model` | [benchmarks/adjoint_time_dependent_temperature/plugin/adjoint_material_model.cc](/Users/hangqian/projects/aspect/benchmarks/adjoint_time_dependent_temperature/plugin/adjoint_material_model.cc:325) |
| `AnnulusMaterial` | [benchmarks/annulus/plugin/annulus.cc](/Users/hangqian/projects/aspect/benchmarks/annulus/plugin/annulus.cc:802) |
| `BursteddeMaterial` | [benchmarks/burstedde/burstedde.cc](/Users/hangqian/projects/aspect/benchmarks/burstedde/burstedde.cc:494) |
| `compressibility formulations` | [benchmarks/compressibility_formulations/plugins/compressibility_formulations.cc](/Users/hangqian/projects/aspect/benchmarks/compressibility_formulations/plugins/compressibility_formulations.cc:258) |
| `VoT` | [benchmarks/davies_et_al/case-2.3-plugin/VoT.cc](/Users/hangqian/projects/aspect/benchmarks/davies_et_al/case-2.3-plugin/VoT.cc:187) |
| `DoneaHuertaMaterial` | [benchmarks/doneahuerta/doneahuerta.cc](/Users/hangqian/projects/aspect/benchmarks/doneahuerta/doneahuerta.cc:338) |
| `HollowSphereMaterial` | [benchmarks/hollow_sphere/hollow_sphere.cc](/Users/hangqian/projects/aspect/benchmarks/hollow_sphere/hollow_sphere.cc:663) |
| `InclusionCompositionalMaterial` | [benchmarks/inclusion/compositional_fields/inclusion_compositional_fields.cc](/Users/hangqian/projects/aspect/benchmarks/inclusion/compositional_fields/inclusion_compositional_fields.cc:29) |
| `InclusionMaterial` | [benchmarks/inclusion/inclusion.cc](/Users/hangqian/projects/aspect/benchmarks/inclusion/inclusion.cc:29) |
| `king material` | [benchmarks/king2dcompressible/plugin/code.cc](/Users/hangqian/projects/aspect/benchmarks/king2dcompressible/plugin/code.cc:240) |
| `LayeredFlowMaterial` | [benchmarks/layeredflow/layeredflow.cc](/Users/hangqian/projects/aspect/benchmarks/layeredflow/layeredflow.cc:385) |
| `simple nonlinear` | [benchmarks/newton_solver_benchmark_set/nonlinear_channel_flow/simple_nonlinear.cc](/Users/hangqian/projects/aspect/benchmarks/newton_solver_benchmark_set/nonlinear_channel_flow/simple_nonlinear.cc:388) |
| `drucker prager compositions` | [benchmarks/newton_solver_benchmark_set/spiegelman_et_al_2016/drucker_prager_compositions.cc](/Users/hangqian/projects/aspect/benchmarks/newton_solver_benchmark_set/spiegelman_et_al_2016/drucker_prager_compositions.cc:642) |
| `nsinker` | [benchmarks/nsinker/nsinker.cc](/Users/hangqian/projects/aspect/benchmarks/nsinker/nsinker.cc:341) |
| `nsinker` | [benchmarks/nsinker_spherical_shell/nsinker.cc](/Users/hangqian/projects/aspect/benchmarks/nsinker_spherical_shell/nsinker.cc:342) |
| `exponential decay` | [benchmarks/operator_splitting/advection_reaction/advection_reaction.cc](/Users/hangqian/projects/aspect/benchmarks/operator_splitting/advection_reaction/advection_reaction.cc:467) |
| `exponential decay` | [benchmarks/operator_splitting/exponential_decay/exponential_decay.cc](/Users/hangqian/projects/aspect/benchmarks/operator_splitting/exponential_decay/exponential_decay.cc:378) |
| `RTinstabilityMaterial` | [benchmarks/rayleigh_taylor_instability/rayleigh_taylor_instability.cc](/Users/hangqian/projects/aspect/benchmarks/rayleigh_taylor_instability/rayleigh_taylor_instability.cc:192) |
| `rigid shear` | [benchmarks/rigid_shear/plugin/rigid_shear.cc](/Users/hangqian/projects/aspect/benchmarks/rigid_shear/plugin/rigid_shear.cc:366) |
| `shear bands material` | [benchmarks/shear_bands/shear_bands.cc](/Users/hangqian/projects/aspect/benchmarks/shear_bands/shear_bands.cc:816) |
| `SinkingBlockMaterial` | [benchmarks/sinking_block/sinking_block.cc](/Users/hangqian/projects/aspect/benchmarks/sinking_block/sinking_block.cc:225) |
| `SolCxCompositionalMaterial` | [benchmarks/solcx/compositional_fields/solcx_compositional_fields.cc](/Users/hangqian/projects/aspect/benchmarks/solcx/compositional_fields/solcx_compositional_fields.cc:30) |
| `SolCxMaterial` | [benchmarks/solcx/solcx.cc](/Users/hangqian/projects/aspect/benchmarks/solcx/solcx.cc:29) |
| `Solitary Wave` | [benchmarks/solitary_wave/solitary_wave.cc](/Users/hangqian/projects/aspect/benchmarks/solitary_wave/solitary_wave.cc:1010) |
| `SolKzCompositionalMaterial` | [benchmarks/solkz/compositional_fields/solkz_compositional_fields.cc](/Users/hangqian/projects/aspect/benchmarks/solkz/compositional_fields/solkz_compositional_fields.cc:29) |
| `SolKzMaterial` | [benchmarks/solkz/solkz.cc](/Users/hangqian/projects/aspect/benchmarks/solkz/solkz.cc:29) |
| `volatiles` | [benchmarks/solubility/plugin/solubility.cc](/Users/hangqian/projects/aspect/benchmarks/solubility/plugin/solubility.cc:419) |
| `Tan Gurnis` | [benchmarks/tangurnis/code/tangurnis.cc](/Users/hangqian/projects/aspect/benchmarks/tangurnis/code/tangurnis.cc:447) |
| `time dependent annulus` | [benchmarks/time_dependent_annulus/plugin/time_dependent_annulus.cc](/Users/hangqian/projects/aspect/benchmarks/time_dependent_annulus/plugin/time_dependent_annulus.cc:28) |
| `TosiMaterial` | [benchmarks/tosi_et_al_2015_gcubed/tosi.cc](/Users/hangqian/projects/aspect/benchmarks/tosi_et_al_2015_gcubed/tosi.cc:678) |
| `ViscosityGroovesMaterial` | [benchmarks/viscosity_grooves/viscosity_grooves.cc](/Users/hangqian/projects/aspect/benchmarks/viscosity_grooves/viscosity_grooves.cc:449) |
| `CPO-induced anisotropic viscosity` | [cookbooks/CPO_induced_anisotropic_viscosity/plugin/cpo_induced_anisotropic_viscosity.cc](/Users/hangqian/projects/aspect/cookbooks/CPO_induced_anisotropic_viscosity/plugin/cpo_induced_anisotropic_viscosity.cc:531) |
| `AV material` | [cookbooks/anisotropic_viscosity/av_material.cc](/Users/hangqian/projects/aspect/cookbooks/anisotropic_viscosity/av_material.cc:387) |
| `finite strain` | [cookbooks/finite_strain/finite_strain.cc](/Users/hangqian/projects/aspect/cookbooks/finite_strain/finite_strain.cc:114) |
| `simpler with crust` | [cookbooks/free_surface_with_crust/plugin/simpler_with_crust.cc](/Users/hangqian/projects/aspect/cookbooks/free_surface_with_crust/plugin/simpler_with_crust.cc:213) |
| `inner core material` | [cookbooks/inner_core_convection/inner_core_convection.cc](/Users/hangqian/projects/aspect/cookbooks/inner_core_convection/inner_core_convection.cc:468) |
| `magnetic stripes` | [cookbooks/magnetic_stripes/magnetic_stripes.cc](/Users/hangqian/projects/aspect/cookbooks/magnetic_stripes/magnetic_stripes.cc:135) |
| `Morency and Doin` | [cookbooks/morency_doin_2004/morency_doin.cc](/Users/hangqian/projects/aspect/cookbooks/morency_doin_2004/morency_doin.cc:272) |
| `phase transition kinetics` | [cookbooks/phase_transition_kinetics/phase-transition-kinetics.cc](/Users/hangqian/projects/aspect/cookbooks/phase_transition_kinetics/phase-transition-kinetics.cc:502) |
| `tomography based plate motions` | [cookbooks/tomography_based_plate_motions/plugins/tomography_based_plate_motions.cc](/Users/hangqian/projects/aspect/cookbooks/tomography_based_plate_motions/plugins/tomography_based_plate_motions.cc:1937) |
| `Material1` | [tests/additional_outputs_02.cc](/Users/hangqian/projects/aspect/tests/additional_outputs_02.cc:165) |
| `Material1` | [tests/additional_outputs_03.cc](/Users/hangqian/projects/aspect/tests/additional_outputs_03.cc:179) |
| `anisotropic` | [tests/anisotropic_viscosity.cc](/Users/hangqian/projects/aspect/tests/anisotropic_viscosity.cc:224) |
| `BursteddeMaterial` | [tests/burstedde_stokes_rhs.cc](/Users/hangqian/projects/aspect/tests/burstedde_stokes_rhs.cc:452) |
| `cell` | [tests/cell_reference.cc](/Users/hangqian/projects/aspect/tests/cell_reference.cc:55) |
| `simple with melt` | [tests/composition_active_with_melt.cc](/Users/hangqian/projects/aspect/tests/composition_active_with_melt.cc:153) |
| `iterated reaction` | [tests/composition_reaction_iterated_IMPES.cc](/Users/hangqian/projects/aspect/tests/composition_reaction_iterated_IMPES.cc:82) |
| `compressibility` | [tests/compressibility.cc](/Users/hangqian/projects/aspect/tests/compressibility.cc:82) |
| `compressibility iterated stokes` | [tests/compressibility_iterated_stokes.cc](/Users/hangqian/projects/aspect/tests/compressibility_iterated_stokes.cc:82) |
| `melt global compression` | [tests/compression_heating.cc](/Users/hangqian/projects/aspect/tests/compression_heating.cc:62) |
| `simpler with crust` | [tests/cookbook_simpler_with_crust.cc](/Users/hangqian/projects/aspect/tests/cookbook_simpler_with_crust.cc:201) |
| `melt free surface` | [tests/free_surface_blob_melt.cc](/Users/hangqian/projects/aspect/tests/free_surface_blob_melt.cc:322) |
| `grain size latent heat` | [tests/grain_size_latent_heat.cc](/Users/hangqian/projects/aspect/tests/grain_size_latent_heat.cc:185) |
| `melt global advection` | [tests/heat_advection_by_melt.cc](/Users/hangqian/projects/aspect/tests/heat_advection_by_melt.cc:122) |
| `test material` | [tests/hydrostatic_compression.cc](/Users/hangqian/projects/aspect/tests/hydrostatic_compression.cc:306) |
| `latent heat enthalpy` | [tests/latent_heat_enthalpy.cc](/Users/hangqian/projects/aspect/tests/latent_heat_enthalpy.cc:176) |
| `compressible melt material` | [tests/melt_compressible_advection.cc](/Users/hangqian/projects/aspect/tests/melt_compressible_advection.cc:197) |
| `test melt material` | [tests/melt_force_vector.cc](/Users/hangqian/projects/aspect/tests/melt_force_vector.cc:325) |
| `MeltMaterial` | [tests/melt_introspection.cc](/Users/hangqian/projects/aspect/tests/melt_introspection.cc:134) |
| `MeltMaterial` | [tests/melt_material_1.cc](/Users/hangqian/projects/aspect/tests/melt_material_1.cc:100) |
| `MeltMaterial2` | [tests/melt_material_2.cc](/Users/hangqian/projects/aspect/tests/melt_material_2.cc:121) |
| `MeltMaterial3` | [tests/melt_material_3.cc](/Users/hangqian/projects/aspect/tests/melt_material_3.cc:121) |
| `MeltMaterial4` | [tests/melt_material_4.cc](/Users/hangqian/projects/aspect/tests/melt_material_4.cc:224) |
| `test melt material` | [tests/melt_transport_adaptive.cc](/Users/hangqian/projects/aspect/tests/melt_transport_adaptive.cc:498) |
| `compressible melt material` | [tests/melt_transport_compressible.cc](/Users/hangqian/projects/aspect/tests/melt_transport_compressible.cc:354) |
| `test melt material` | [tests/melt_transport_convergence_simple.cc](/Users/hangqian/projects/aspect/tests/melt_transport_convergence_simple.cc:310) |
| `test melt material` | [tests/melt_transport_convergence_test.cc](/Users/hangqian/projects/aspect/tests/melt_transport_convergence_test.cc:314) |
| `EXPERIMENTAL melt visco plastic` | [tests/melt_visco_plastic.cc](/Users/hangqian/projects/aspect/tests/melt_visco_plastic.cc:1028) |
| `melting rate` | [tests/melting_rate.cc](/Users/hangqian/projects/aspect/tests/melting_rate.cc:121) |
| `prescribed field material` | [tests/multiple_named_additional_outputs.cc](/Users/hangqian/projects/aspect/tests/multiple_named_additional_outputs.cc:107) |
| `prescribed fields material` | [tests/multiple_prescribed_fields.cc](/Users/hangqian/projects/aspect/tests/multiple_prescribed_fields.cc:91) |
| `prescribed fields material` | [tests/multiple_prescribed_fields_dc.cc](/Users/hangqian/projects/aspect/tests/multiple_prescribed_fields_dc.cc:91) |
| `no adiabatic heating` | [tests/no_adiabatic_heating.cc](/Users/hangqian/projects/aspect/tests/no_adiabatic_heating.cc:51) |
| `no adiabatic heating` | [tests/no_adiabatic_heating_02.cc](/Users/hangqian/projects/aspect/tests/no_adiabatic_heating_02.cc:59) |
| `no adiabatic heating` | [tests/no_adiabatic_heating_03.cc](/Users/hangqian/projects/aspect/tests/no_adiabatic_heating_03.cc:51) |
| `test material` | [tests/nonlinear_convergence_for_small_composition.cc](/Users/hangqian/projects/aspect/tests/nonlinear_convergence_for_small_composition.cc:67) |
| `MyMaterial` | [tests/prescribed_dilation.cc](/Users/hangqian/projects/aspect/tests/prescribed_dilation.cc:394) |
| `prescribed field material` | [tests/prescribed_field.cc](/Users/hangqian/projects/aspect/tests/prescribed_field.cc:86) |
| `prescribed field material` | [tests/prescribed_field_temperature.cc](/Users/hangqian/projects/aspect/tests/prescribed_field_temperature.cc:86) |
| `prescribed field material` | [tests/prescribed_field_with_diffusion.cc](/Users/hangqian/projects/aspect/tests/prescribed_field_with_diffusion.cc:89) |
| `prescribed temperature material` | [tests/prescribed_temperature.cc](/Users/hangqian/projects/aspect/tests/prescribed_temperature.cc:89) |
| `prescribed temperature material` | [tests/prescribed_temperature_with_diffusion.cc](/Users/hangqian/projects/aspect/tests/prescribed_temperature_with_diffusion.cc:89) |
| `InclusionMaterial` | [tests/prescribed_velocity_boundary.cc](/Users/hangqian/projects/aspect/tests/prescribed_velocity_boundary.cc:423) |
| `DoneaHuertaMaterial` | [tests/q1_q1.cc](/Users/hangqian/projects/aspect/tests/q1_q1.cc:450) |
| `scaled viscosity profile` | [tests/rheology_scaled_profile.cc](/Users/hangqian/projects/aspect/tests/rheology_scaled_profile.cc:283) |
| `melting rate` | [tests/rising_melt_blob.cc](/Users/hangqian/projects/aspect/tests/rising_melt_blob.cc:97) |
| `shear thinning` | [tests/shear_thinning.cc](/Users/hangqian/projects/aspect/tests/shear_thinning.cc:77) |
| `finite strain` | [tests/simple_shear.cc](/Users/hangqian/projects/aspect/tests/simple_shear.cc:114) |
| `finite strain` | [tests/simple_shear_output_the_mobility.cc](/Users/hangqian/projects/aspect/tests/simple_shear_output_the_mobility.cc:114) |
| `spiegelman 2016` | [tests/spiegelman_fail_test.cc](/Users/hangqian/projects/aspect/tests/spiegelman_fail_test.cc:667) |
| `test material` | [tests/velocity_divergence.cc](/Users/hangqian/projects/aspect/tests/velocity_divergence.cc:65) |
| `TestMaterial` | [tests/write_plugin_graph.cc](/Users/hangqian/projects/aspect/tests/write_plugin_graph.cc:95) |
