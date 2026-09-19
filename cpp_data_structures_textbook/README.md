# ASPECT C++ 数据结构案例教材

这套教材面向 C++ 入门学习者，但案例全部来自 ASPECT 仓库真实 `.cc` 文件。学习目标不是把 ASPECT 的 960 个 `.cc` 文件逐行翻译，而是把高频数据结构和典型完整函数讲透：它们如何构造、如何填充、如何遍历、如何传递、如何在并行有限元程序中表达物理意义。

## 推荐阅读顺序

1. 先读第 1-3 章，建立顺序容器、二维容器和固定长度容器的基础。
2. 再读第 4-7 章，理解 set、map、list、pair 如何表达接口和集合关系。
3. 再读第 8-9 章，学习智能指针和 ASPECT 插件对象生命周期。
4. 最后读第 10-12 章，进入 deal.II/ASPECT 专用类型：几何张量、BlockVector、IndexSet、ComponentMask、FullMatrix 和 Table。

## 章节索引

- `chapter_01_vector_statistics.md`：`std::vector` 与后处理统计。
- `chapter_02_nested_vector_material_io.md`：二维 vector 与材料模型输入输出。
- `chapter_03_array_fixed_dimension.md`：`std::array` 与固定维度坐标。
- `chapter_04_set_boundary_ids.md`：`std::set` 与边界选择。
- `chapter_05_map_boundary_statistics.md`：`std::map` 与命名映射、统计聚合。
- `chapter_06_list_plugin_outputs.md`：`std::list` 与插件依赖、输出流水线。
- `chapter_07_pair_return_values.md`：`std::pair` 与返回值设计。
- `chapter_08_unique_ptr_factory.md`：`std::unique_ptr` 与插件工厂。
- `chapter_09_shared_ptr_additional_outputs.md`：`std::shared_ptr` 与 AdditionalOutputs。
- `chapter_10_point_tensor_physics.md`：`Point`、`Tensor`、`SymmetricTensor`。
- `chapter_11_blockvector_stokes.md`：`LinearAlgebra::BlockVector` 与 Stokes 子系统。
- `chapter_12_indexset_matrix_table.md`：`IndexSet`、`ComponentMask`、`FullMatrix`、`Table`。

## 源码检索方法

在 ASPECT 根目录运行：

```bash
rg -n "std::vector|std::set|std::map|BlockVector|IndexSet|ComponentMask|FullMatrix|Tensor" --glob '*.cc' source cookbooks tests
```

阅读某一章时，建议打开章节里标注的源码路径，再用 `rg` 搜函数名。教材只截取关键逻辑，完整上下文仍以源码为准。


## 详细版说明

本教材已扩展为详细版：每章末尾都有“详细版十页扩展”，显式包含第 1 页到第 10 页的教学分节。每一页都围绕一个具体阅读动作展开，包括问题场景、源码职责、构造初始化、填充读取、遍历下标、状态变化、常用方法、替代设计、调试路线和课后练习。

第 6 章原计划中的 `tests/execute_selected_postprocessors.cc` 在当前工作区不可用，因此教材使用真实存在且更核心的 `source/postprocess/interface.cc` 作为后处理输出流水线案例，同时保留 `tests/plugin_dependency.cc` 作为插件依赖的最小示例。
