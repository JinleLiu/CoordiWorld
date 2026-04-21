# Project Brief: CoordiWorld + SceneSummary

## 1) CoordiWorld 要实现什么
本项目将 CoordiWorld 落地为一个 **fixed candidate-set trajectory evaluator**。其职责是：
- 接收场景历史 SceneSummary 与候选自车轨迹集合；
- 对每条候选轨迹做 action-conditioned structured rollout；
- 输出多维风险与可信度相关指标，并形成可校准分数 `J(tau)`；
- 支持事后实体级归因，满足可解释与可审计需求。

本项目明确不将 CoordiWorld 作为轨迹生成器。

## 2) SceneSummary 在本项目中的角色
SceneSummary 是 evaluator 的结构化输入载体，负责把多源感知/地图/上下文信息整理为可追溯表示。  
本项目中的 SceneSummary 模块遵循 InfoCoordiBridge 的 ICA 思路：
- coordinate normalization
- cross-source entity alignment
- conflict-aware attribute fusion
- provenance/source lineage/ambiguity flags

SceneSummary 质量将直接影响 rollout 一致性、风险评估稳定性与 attribution 可靠性。

## 3) 实验代码最终需覆盖的指标
在完整实验阶段，评估代码应至少覆盖：
- collision risk 指标
- map-grounded rule-violation risk 指标
- predictive uncertainty 指标
- calibrated evaluator score `J(tau)` 的校准表现
- post-hoc entity-level attribution 的可解释性输出

同时需要支持 fixed candidate-set protocol 下的批量评估与结果汇总。

## 4) 当前不能做的事情
在没有真实 NAVSIM/OpenScene/NuScenes/Waymo 数据时：
- 不能宣称真实 benchmark 结果；
- 不能伪造或硬编码论文表格中的实验数值；
- 不能输出未被真实实验支撑的“复现结论”。

当前允许的工程活动限定为：
- 接口设计
- mock 测试
- synthetic smoke test
- clearly marked stub
