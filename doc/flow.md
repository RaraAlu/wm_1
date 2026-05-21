# LeWM / JEPA 项目流程文档

## 项目概览

本项目实现了一个基于 **JEPA（Joint Embedding Predictive Architecture）** 的世界模型。

**核心思想**：不在像素空间预测未来，而是把图像压缩成低维向量（嵌入），在嵌入空间里预测未来状态。训练完成后，这个"想象能力"用于规划——在脑子里模拟不同动作的结果，选出最优动作序列。

**两个阶段**：
- **训练**：从专家轨迹数据中学习"给我看前几帧+动作，我能猜出下一帧的样子"
- **评估**：用学到的世界模型做 CEM 规划，让机器人在真实环境中完成目标

**两个库的分工**：
- `jepa.py`（本项目）：世界模型的"大脑"，负责**评分**——给定一个动作序列，预测结果有多好
- `stable_worldmodel`（外部库）：规划框架的"外壳"，负责**搜索**——用 CEM 算法找到最优动作序列

---

## 一、训练数据

**数据来源**：`dates/pusht_expert_train.h5`  
专家演示轨迹，每条轨迹记录机器人完成 PushT 任务的完整过程。

**一条训练样本长什么样**（`num_steps = history_size + num_preds = 4`，`frameskip = 5`）：

```
帧0: 图像A,  动作"向右"（实际是 5 帧的叠加动作）
帧1: 图像B,  动作"向上"
帧2: 图像C,  动作"向左"
帧3: 图像D,  动作"向右"
```

**预处理**：
- 图像：Resize → 224×224，ImageNet 均值/方差归一化
- 动作/本体感知：StandardScaler 归一化（均值 0，方差 1）

---

## 二、训练流程

**目标**：让 JEPA 学会"给我看前 3 帧 + 动作，我能猜出后续帧的嵌入"

### 第一步：把图像压缩成向量（encode）

```
图像A (3,224,224) → ViT-Tiny → CLS token → projector MLP → emb_A (192维)
图像B             →    ...                              → emb_B
图像C             →    ...                              → emb_C
图像D             →    ...                              → emb_D

动作序列 (T, A) → action_encoder MLP → act_emb (T, 192)
```

ViT 把整张图像压缩为一个 192 维的向量，CLS token 是 ViT 中汇聚全局信息的特殊标记。

### 第二步：切分"输入"和"正确答案"

```
history_size = 3，num_preds = 1

输入（ctx）：[emb_A, emb_B, emb_C] + 动作 [右, 上, 左]
正确答案：   [emb_B, emb_C, emb_D]   ← 每步对应"下一帧的嵌入"
```

### 第三步：predictor 做预测

```
ARPredictor（Causal Transformer，depth=6，因果注意力）：

看到 emb_A + 动作"右"       → 预测 pred_B
看到 emb_A, B + 动作"上"    → 预测 pred_C
看到 emb_A, B, C + 动作"左" → 预测 pred_D
```

"Causal"意味着预测第 t 步时只能看到第 0~t 步，不能看到未来。

### 第四步：计算损失，反向传播

```
pred_loss   = MSE(pred_B, emb_B) + MSE(pred_C, emb_C) + MSE(pred_D, emb_D)
            ← 预测要和真实嵌入接近

sigreg_loss = SIGReg(所有嵌入)
            ← 防止所有图像都被压缩成同一个向量（嵌入坍缩）
            ← 通过随机投影检查嵌入分布是否接近标准高斯分布

loss = pred_loss + 0.09 × sigreg_loss
```

优化器：AdamW（lr=5e-5）+ LinearWarmupCosineAnnealing，训练 100 个 epoch。

---

## 三、模型架构

```
JEPA
├── encoder        ViT-Tiny (patch=14, img=224×224)
│                  → 输出 CLS token，形状 (B*T, 192)
├── projector      MLP(192 → 2048 → 192, BatchNorm1d)
│                  → 对 encoder 输出做进一步变换
├── action_encoder Embedder MLP(A_dim → 192)
│                  → 将动作向量映射到与观测嵌入相同的维度
├── predictor      ARPredictor
│                  └── Transformer(depth=6, heads=16, dim_head=64, mlp_dim=2048)
│                      带 Causal Self-Attention，接收 emb + act_emb，预测下一步 emb
└── pred_proj      MLP(192 → 2048 → 192, BatchNorm1d)
                   → 对 predictor 输出做投影，对齐目标嵌入空间
```

---

## 四、评估阶段：JEPA + CEM 规划

### 两者的分工

```
stable_worldmodel（搜索）          jepa.py（评分）
─────────────────────────          ──────────────────────────
CEMSolver：
  采样 300 个动作序列    ──调用──▶  JEPA.get_cost()
                                     ├─ encode(目标图像) → goal_emb
                                     ├─ rollout(候选动作) → predicted_emb
                                     └─ criterion() → cost (1, 300)
  选 top-30，更新分布
  重复 30 轮
  输出最优动作序列

WorldModelPolicy：
  缓冲区空了就触发规划
  执行 receding_horizon 步
  warm_start：用上轮尾部初始化下轮
```

JEPA 只回答"某个动作序列有多好"，CEM 负责搜索好的序列。两者通过 `Costable` 协议解耦，可以随时换成其他世界模型。

### CEM 规划详细过程（参数：`n_steps=30, num_samples=300, topk=30, horizon=5`）

**场景**：机器人要把方块推到目标位置，当前能看到当前画面和目标画面。

**初始化**：
```
mean = 全零序列  形状 (1, 5步, action_dim)   ← 动作均值
var  = 全 1      形状 (1, 5步, action_dim)   ← 动作方差
（若 warm_start=True，mean 用上一轮规划的尾部初始化）
```

**优化循环，重复 30 轮**：

```
第 1 轮：
  ① 从 N(mean, var) 采样 300 个候选序列，每个序列 = 5 步动作
     candidates[0]     = mean（强制保留当前最优解）
     candidates[1..299] = 随机扰动

  ② 把 300 个序列送给 JEPA.get_cost()
     JEPA 在嵌入空间"想象"每个序列执行后的结果
     → cost (1, 300)，每个候选对应一个标量

  ③ 选出 cost 最小的 30 个（topk=30，最接近目标的）

  ④ 更新分布
     new_mean = 30 个精英序列的均值   ← 向好的区域收敛
     new_var  = 30 个精英序列的标准差 ← 搜索范围逐渐缩小

第 2 轮 → 第 30 轮：重复，分布越来越聚焦在最优解附近
```

**执行**（`receding_horizon=5, action_block=5`）：
```
输出最优序列：[a0, a1, a2, a3, a4]，共 5 步
每步重复 5 次（frameskip）→ 实际执行 25 环境帧
执行完后动作缓冲区清空 → 触发下一轮规划
```

### rollout 展开原理

给定初始 3 帧观测和候选动作序列，JEPA 自回归地想象未来：

```
初始编码：  [e0, e1, e2]          （来自真实观测）
步骤 t=0：  [e0, e1, e2] + a2 → 预测 e3
步骤 t=1：  [e1, e2, e3] + a3 → 预测 e4   （滑动窗口，只取最近 3 步）
步骤 t=2：  [e2, e3, e4] + a4 → 预测 e5
```

最终用预测末态 e5 和目标嵌入 goal_emb 计算 MSE，得到代价。

---

## 五、完整评估流程（`eval.py`）

```
① 加载数据集 (HDF5)，拟合归一化器（StandardScaler）

② 随机采样 50 个评估起始帧（seed=42 固定）
   每个起始帧对应一个"从这里开始，25 步后的画面"作为目标

③ 构建策略
   if policy == "random":  RandomPolicy（随机动作，baseline）
   else:                   加载预训练 JEPA 权重
                           构建 WorldModelPolicy(CEMSolver, PlanConfig)

④ world.evaluate() 主循环
   for each episode:
     重置环境到起始状态
     while 步数 < eval_budget (50步):
       if 动作缓冲区为空:         ← 每 5 步触发一次规划
         CEM 规划（30轮×300样本）→ 最优动作序列
       执行缓冲区中的下一步动作
       观察新画面，更新历史

⑤ 保存 metrics + 视频录像到 results_path
```

---

## 六、文件结构速查

```
wm_1/
├── jepa.py          JEPA 世界模型（encode / predict / rollout / criterion / get_cost）
├── module.py        网络组件（ViT Block、Transformer、ARPredictor、SIGReg、MLP、Embedder）
├── train.py         训练入口（lejepa_forward、Hydra 配置、Lightning Trainer）
├── eval.py          评估入口（CEM 规划、环境交互、指标记录）
├── utils.py         工具函数（归一化器、图像预处理、CheckpointCallback）
├── config/
│   ├── train/
│   │   ├── lewm.yaml          训练主配置（超参数）
│   │   ├── model/lewm.yaml    模型结构配置（网络层数、维度）
│   │   └── data/pusht.yaml    数据集配置（路径、frameskip）
│   └── eval/
│       ├── pusht.yaml         评估主配置（回合数、目标偏移）
│       └── solver/cem.yaml    CEM 求解器配置（采样数、迭代轮数）
└── .venv/Lib/site-packages/
    ├── stable_worldmodel/     规划框架（CEMSolver、WorldModelPolicy、World、HDF5Dataset）
    └── stable_pretraining/    预训练工具（ViT 骨干、数据变换、训练框架）
```

---

## 七、关键超参数汇总

| 类别 | 参数 | 值 | 说明 |
|---|---|---|---|
| 训练 | `img_size` | 224 | 输入图像分辨率 |
| 训练 | `wm.history_size` | 3 | 上下文窗口（输入帧数） |
| 训练 | `wm.num_preds` | 1 | 预测偏移（标签从第 1 步开始） |
| 训练 | `wm.embed_dim` | 192 | 嵌入维度 D |
| 训练 | `loss.sigreg.weight` | 0.09 | SIGReg 正则权重 λ |
| 训练 | `optimizer.lr` | 5e-5 | AdamW 学习率 |
| 训练 | `trainer.max_epochs` | 100 | 最大训练轮数 |
| 规划 | `num_samples` | 300 | CEM 每轮采样候选数 |
| 规划 | `n_steps` | 30 | CEM 优化迭代轮数 |
| 规划 | `topk` | 30 | CEM 精英保留数 |
| 规划 | `horizon` | 5 | 规划步数 |
| 规划 | `action_block` | 5 | 每步重复执行次数（frameskip） |
| 评估 | `eval_budget` | 50 | 每回合最大步数 |
| 评估 | `goal_offset_steps` | 25 | 目标帧相对起始帧的偏移 |
| 评估 | `num_eval` | 50 | 评估回合数 |

---

## 八、数据流维度汇总

| 阶段 | 张量 | 形状 | 说明 |
|---|---|---|---|
| 训练输入 | `pixels` | `(B, T, 3, 224, 224)` | 原始图像帧序列 |
| 训练输入 | `action` | `(B, T, A)` | 动作序列（归一化后） |
| 编码后 | `emb` | `(B, T, 192)` | 观测嵌入序列 |
| 编码后 | `act_emb` | `(B, T, 192)` | 动作嵌入序列 |
| 训练预测 | `pred_emb` | `(B, 3, 192)` | 预测的下一状态嵌入 |
| 规划输入 | `action_candidates` | `(B, S, T, A)` | 候选动作计划，S=300 |
| 规划中间 | `predicted_emb` | `(B, S, T+1, 192)` | 各候选计划的预测轨迹 |
| 规划输出 | `cost` | `(B, S)` | 各候选计划的规划代价 |
