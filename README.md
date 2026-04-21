# CoordiWorld Engineering Repository

## 项目目标
本仓库用于工程化实现一个 **provenance-aware structured world model**，服务于自动驾驶中的 **fixed candidate-set trajectory evaluation**。  
核心目标来自两篇论文：
- `papers/IOTJ-V3.pdf`：CoordiWorld evaluator 框架
- `papers/manuscript-InfoCoordiBridge.pdf`：InfoCoordiBridge / ICA / SceneSummary 方法

> 重要：CoordiWorld 在本项目中被定义为 **trajectory evaluator**，不是 trajectory generator。

## 论文锚点（实现约束）
1. 输入：
   - SceneSummary history `S_{t-h:t}`
   - candidate ego trajectories `T_t = {tau^(m)}`
2. 机制：对每条候选轨迹执行 action-conditioned structured rollout。
3. 输出：
   - collision risk
   - map-grounded rule-violation risk
   - predictive uncertainty
   - calibrated evaluator score `J(tau)`
   - post-hoc entity-level attribution
4. SceneSummary 生成需遵循 ICA：
   - coordinate normalization
   - cross-source entity alignment
   - conflict-aware attribute fusion
   - provenance / source lineage / ambiguity flags

## 当前状态
- 当前阶段：**项目上下文初始化（spec/init-agents）**
- 已建立：项目约束文档、概要文档、开发日志、基础忽略规则
- 尚未开始：模型代码、训练代码、实验代码、真实数据接入

## 开发路线（Roadmap）
1. **Spec 阶段**
   - 明确 CoordiWorld evaluator 接口与模块边界
   - 明确 SceneSummary schema 与 provenance 字段
2. **Scaffold 阶段**
   - 建立 `src/` 包结构与配置系统
   - 建立最小可运行 mock pipeline（synthetic only）
3. **Core Implementation 阶段**
   - 实现 structured rollout evaluator
   - 实现 deterministic SceneSummary 生成器（非 LLM）
4. **Evaluation 阶段**
   - 固定候选集评估流程
   - 指标统计、校准分析、可解释性归因输出
5. **Data Integration 阶段（条件满足后）**
   - 接入真实数据协议（NAVSIM/OpenScene/NuScenes/Waymo）
   - 逐步替换 synthetic fixtures

## 实验与结果声明策略
- 在缺乏真实数据时，仅允许：
  - interface 定义
  - mock test
  - synthetic smoke test
  - clearly marked stub
- **禁止**伪造、臆造或硬编码论文实验结果。
- README 及后续文档中不得声称“已复现论文 SOTA 指标”，除非有可审计实验记录支持。

## 仓库使用建议
- 优先小步提交，保证每步可审阅。
- 每新增模块必须配套 `pytest`。
- SceneSummary 相关流程保持 deterministic 与可追溯。
