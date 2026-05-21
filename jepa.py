"""
JEPA（Joint Embedding Predictive Architecture）实现

JEPA 是一种自监督世界模型框架，其核心思想是：
  - 在「嵌入空间」中进行预测，而非像素空间
  - 通过编码器将原始观测（图像帧）映射到低维嵌入向量
  - 通过预测器在嵌入空间中预测未来状态
  - 结合动作条件，实现基于模型的规划（Model-Based Planning）

典型用途：
  1. 训练阶段：最小化预测嵌入与真实嵌入之间的 MSE 损失
  2. 推理阶段：给定目标图像，通过 CEM/Adam 等求解器搜索最优动作序列
"""

import torch
import torch.nn.functional as F
from einops import rearrange  # 优雅的张量维度重排库，替代繁琐的 view/permute 组合
from torch import nn


def detach_clone(v):
    """
    对张量执行「阻断梯度 + 深拷贝」操作，对非张量直接原样返回。

    作用：
      - detach()：将张量从计算图中分离，使后续操作不会向该张量反向传播梯度。
                  常用于目标网络（target network）或停止梯度流动的场景。
      - clone()：创建一块新的内存，避免原地修改污染源张量数据。

    参数：
      v: 任意值。若为 torch.Tensor 则执行 detach+clone；否则直接返回原值。

    返回：
      分离后的张量副本，或原始非张量值。
    """
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    """
    JEPA 世界模型主类，继承自 nn.Module。

    架构由五个可插拔组件构成：

    ┌──────────────────────────────────────────────────────────────────┐
    │  观测图像 (pixels)                                               │
    │       │                                                          │
    │  ┌────▼────┐   ┌───────────┐                                    │
    │  │ encoder │──▶│ projector │──▶ 观测嵌入 emb  (B, T, D)         │
    │  └─────────┘   └───────────┘                                    │
    │                                                                  │
    │  动作 (action)                                                    │
    │       │                                                          │
    │  ┌────▼──────────┐                                               │
    │  │ action_encoder│──▶ 动作嵌入 act_emb  (B, T, A_emb)           │
    │  └───────────────┘                                               │
    │                                                                  │
    │  ┌─────────────────────────────────────────────────────┐        │
    │  │  predictor(emb, act_emb) → 预测嵌入  (B, T, D)     │        │
    │  └─────────────────────────┬───────────────────────────┘        │
    │                            │                                     │
    │                    ┌───────▼──────┐                              │
    │                    │  pred_proj   │──▶ 最终预测嵌入              │
    │                    └──────────────┘                              │
    └──────────────────────────────────────────────────────────────────┘

    属性：
      encoder        : 视觉骨干网络（通常为 ViT），将图像帧编码为 CLS token 嵌入
      predictor      : 时序预测器（通常为 Transformer），接收历史嵌入+动作嵌入，预测未来嵌入
      action_encoder : 动作编码器（MLP），将原始动作向量映射到嵌入空间
      projector      : 观测嵌入的投影头（可选，默认为恒等映射 Identity）
      pred_proj      : 预测嵌入的投影头（可选，默认为恒等映射 Identity）
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,   # 观测投影头，若不提供则使用恒等映射
        pred_proj=None,   # 预测投影头，若不提供则使用恒等映射
    ):
        super().__init__()

        # 视觉编码器：接受图像张量，输出包含 last_hidden_state 的对象（HuggingFace 风格）
        self.encoder = encoder
        # 时序预测器：在嵌入空间中建模状态转移动态
        self.predictor = predictor
        # 动作编码器：将离散/连续动作映射到与观测嵌入同维度的空间
        self.action_encoder = action_encoder
        # 观测投影头：对 encoder 输出的 CLS token 做进一步变换（如降维/归一化）
        self.projector = projector or nn.Identity()
        # 预测投影头：对 predictor 的输出做进一步变换，使其与目标嵌入空间对齐
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """
        将原始观测（像素图像）和动作编码为嵌入向量，结果写回 info 字典。

        处理流程：
          1. 取出像素张量并转为 float32（原始数据可能为 uint8）
          2. 将 (B, T, ...) 的批量时序帧「压平」为 (B*T, ...) 以便 encoder 并行处理
          3. 调用视觉 encoder，取 CLS token 作为每帧的全局表示
          4. 经过 projector 投影后，恢复为 (B, T, D) 的时序嵌入
          5. 若 info 中含有 "action" 键，则同步编码动作序列

        参数：
          info (dict): 输入信息字典，必须包含：
            - "pixels": 形如 (B, T, C, H, W) 的图像张量
                        B=批大小, T=时间步数, C=通道数, H=高, W=宽
            可选包含：
            - "action": 形如 (B, T, action_dim) 的动作张量

        返回：
          info (dict): 原地修改后的字典，新增：
            - "emb"    : 观测嵌入，形如 (B, T, D)，D 为嵌入维度
            - "act_emb": 动作嵌入，形如 (B, T, A_emb)（仅当 "action" 存在时）
        """

        # 将像素转为 float32，避免后续运算出现类型不匹配错误
        pixels = info['pixels'].float()
        # 记录批大小 B，后续需要用来还原形状
        b = pixels.size(0)
        # 将 (B, T, C, H, W) 压平为 (B*T, C, H, W)
        # "b t ... -> (b t) ..." 表示将前两维合并，其余维度保持不变
        # 这样 encoder 可以一次性处理所有帧，提升并行效率
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # 压平时序维度以并行编码
        # 调用视觉编码器（如 ViT），interpolate_pos_encoding=True 允许处理训练时未见过的分辨率
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        # 取 last_hidden_state 的第 0 个 token，即 CLS token
        # CLS token 是 ViT 中用于汇聚全局图像信息的特殊标记，形状为 (B*T, D)
        pixels_emb = output.last_hidden_state[:, 0]  # CLS token，全局图像表示
        # 经过投影头，进一步变换嵌入（如对齐预测空间的维度）
        emb = self.projector(pixels_emb)
        # 将 (B*T, D) 还原为 (B, T, D)，恢复批量-时序的二维结构
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            # 对动作序列 (B, T, action_dim) 编码为动作嵌入 (B, T, A_emb)
            # action_encoder 通常是一个共享权重的 MLP，逐时间步独立作用
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """
        给定历史观测嵌入序列和动作嵌入序列，预测未来状态嵌入序列。

        这是 JEPA 的核心预测步骤：predictor 接收过去 T 步的嵌入和动作，
        并对每个时间步输出对应的「下一步」预测嵌入。

        参数：
          emb     (torch.Tensor): 观测嵌入序列，形如 (B, T, D)
                                  B=批大小, T=历史时间步数, D=嵌入维度
          act_emb (torch.Tensor): 动作嵌入序列，形如 (B, T, A_emb)
                                  A_emb=动作嵌入维度

        返回：
          preds (torch.Tensor): 预测嵌入序列，形如 (B, T, D)
                                 preds[:, t, :] 表示在时刻 t 执行动作后的预测下一状态嵌入

        处理细节：
          - predictor 输出 (B, T, D) 后，将其展平为 (B*T, D)
          - 经 pred_proj 投影（允许逐 token 独立变换）
          - 再恢复为 (B, T, D)
        """
        # predictor 内部通常为 Causal Transformer，建模时序依赖关系
        preds = self.predictor(emb, act_emb)
        # 展平时序维度，使 pred_proj（如 MLP）可以逐 token 独立处理
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        # 还原为 (B, T, D) 的时序格式
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## 仅推理时使用   ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """
        给定初始观测和一批候选动作序列，自回归地展开预测轨迹（rollout）。

        此函数是基于模型规划（MPC）的核心：它在「想象中」执行动作序列，
        生成未来状态的嵌入轨迹，供后续代价函数评估。

        参数：
          info (dict): 信息字典，必须包含：
            - "pixels": 形如 (B, S, T_hist, C, H, W) 的初始观测
                        B=批大小，S=候选动作计划数量，T_hist=历史帧数
          action_sequence (torch.Tensor): 候选动作序列
            - 形如 (B, S, T, action_dim)
            - S=候选计划数（如 CEM 采样的 N 个样本），T=总时间步（含历史）
          history_size (int): 预测时使用的历史窗口长度，默认 3
            - 仅将最近 history_size 步的嵌入喂给 predictor，避免序列过长

        返回：
          info (dict): 新增键：
            - "predicted_emb": 完整预测轨迹嵌入，形如 (B, S, T+1, D)
                                第 0 维对应 batch，第 1 维对应不同候选计划

        展开流程示意（以 T_hist=2, n_steps=3 为例）：

          初始编码：  [e0, e1]        （来自真实观测）
          步骤 t=0：  [e0, e1] + a0 → pred e2
          步骤 t=1：  [e1, e2] + a1 → pred e3   （滑动窗口，只取最近 HS 步）
          步骤 t=2：  [e2, e3] + a2 → pred e4
          最终补全：  [e3, e4] + a3 → pred e5
        """

        # 确保输入字典中包含像素观测
        assert "pixels" in info, "pixels not in info_dict"
        # H = 历史帧数（info["pixels"] 的时间维度）
        # info["pixels"] 形状为 (B, S, H, C, H_img, W_img)，size(2) 取时间轴
        H = info["pixels"].size(2)
        # 解析动作序列的维度：B=批大小，S=候选计划数，T=总时间步
        B, S, T = action_sequence.shape[:3]
        # 将动作序列切分为两部分：
        #   act_0      : 与历史观测对齐的前 H 步动作，用于初始编码  (B, S, H, action_dim)
        #   act_future : 需要自回归展开的未来动作                   (B, S, T-H, action_dim)
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        # 将历史动作存入 info，供 encode() 使用
        info["action"] = act_0
        # 未来需要自回归展开的步数
        n_steps = T - H

        # ── 步骤 1：编码初始状态 ──────────────────────────────────────────
        # 取所有候选计划的第 0 个样本（[:, 0]）来编码初始观测
        # 注意：B 个 batch 的初始观测是相同的，所有 S 个计划共享同一初始状态
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        # 将初始嵌入沿 S 维度复制，扩展为 (B, S, H, D)
        # unsqueeze(1) 插入 S 维度后，expand 广播到 S 份（节省内存，不实际复制数据）
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        # 对初始编码结果做 detach+clone，避免后续 inplace 操作污染计算图
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # ── 步骤 2：将 (B, S, ...) 压平为 (B*S, ...) ──────────────────────
        # 将批次维和样本维合并，便于 predictor 并行处理所有候选计划
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()  # (B*S, H, D)
        # (B*S, H, action_dim)
        act = rearrange(act_0, "b s ... -> (b s) ...")
        # (B*S, T-H, action_dim)
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # ── 步骤 3：自回归展开 n_steps 步 ────────────────────────────────
        HS = history_size  # 预测时使用的滑动历史窗口大小
        for t in range(n_steps):
            # 编码当前所有时间步的动作（act 会随循环逐步延长）
            # (B*S, current_T, A_emb)
            act_emb = self.action_encoder(act)
            # 取最近 HS 步的嵌入作为 predictor 的输入（滑动窗口，控制序列长度）
            emb_trunc = emb[:, -HS:]                        # (B*S, HS, D)
            act_trunc = act_emb[:, -HS:]                    # (B*S, HS, A_emb)
            # 预测下一步嵌入，取最后一个输出 token（对应最新时刻的预测）
            pred_emb = self.predict(emb_trunc, act_trunc)[
                :, -1:]  # (B*S, 1, D)
            # 将预测嵌入拼接到历史嵌入序列末尾，形成延长的嵌入轨迹
            # (B*S, current_T+1, D)
            emb = torch.cat([emb, pred_emb], dim=1)

            # 取出下一步的未来动作，拼接到当前动作序列末尾
            # (B*S, 1, action_dim)
            next_act = act_future[:, t: t + 1, :]
            # (B*S, current_T+1, action_dim)
            act = torch.cat([act, next_act], dim=1)

        # ── 步骤 4：预测最终一步（循环外额外执行一次）────────────────────
        # 循环展开 n_steps 步后，还需要对「最后加入的动作」再做一次预测，
        # 使预测嵌入序列与动作序列等长
        act_emb = self.action_encoder(act)                   # (B*S, T, A_emb)
        emb_trunc = emb[:, -HS:]                             # (B*S, HS, D)
        act_trunc = act_emb[:, -HS:]                         # (B*S, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (B*S, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)              # (B*S, T+1, D)

        # ── 步骤 5：还原维度并存储结果 ───────────────────────────────────
        # 将 (B*S, T+1, D) 还原为 (B, S, T+1, D)
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        # 将完整预测轨迹写入 info 字典
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """
        计算预测嵌入与目标嵌入之间的代价（Cost）。

        在基于模型的规划中，代价函数衡量「沿预测轨迹能够多大程度接近目标状态」。
        本函数仅关注轨迹的最后一步，即最终预测状态与目标状态的距离。

        参数：
          info_dict (dict): 必须包含：
            - "predicted_emb": rollout 生成的预测嵌入，形如 (B, S, T_pred, D)
            - "goal_emb"     : 目标状态嵌入，形如 (B, S, T_goal, D)
                               通常 T_goal=1（单一目标帧）

        返回：
          cost (torch.Tensor): 每个候选动作计划的代价，形如 (B, S)
                               值越小表示该计划越接近目标状态

        代价计算方式：
          1. 将 goal_emb 广播（expand）到与 pred_emb 相同形状
          2. 仅取最后一个时间步 [..., -1:, :] 计算 MSE
          3. 对嵌入维度求和（sum over D），得到标量代价
          4. goal_emb 使用 .detach() 阻断梯度（目标嵌入为固定参考点）
        """
        # 取出预测嵌入，形状 (B, S, T_pred, D)
        pred_emb = info_dict["predicted_emb"]  # (B, S, T_pred, dim)
        # 取出目标嵌入，形状 (B, S, T_goal, D)（通常 T_goal=1）
        goal_emb = info_dict["goal_emb"]       # (B, S, T_goal, dim)

        # 将目标嵌入（通常只有 1 个时间步）广播扩展到与 pred_emb 相同的形状
        # [..., -1:, :] 取最后一个目标帧，expand_as 广播到 (B, S, T_pred, D)
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # 仅计算预测轨迹最后一步的 MSE 代价，代表规划终点与目标的距离
        # reduction="none" 保留每个元素的损失值，不进行均值聚合
        # .sum(dim=...) 对第 2 维及之后的所有维度求和（即对时间步和嵌入维度求和）
        # 结果形状：(B, S)，每个候选计划对应一个标量代价
        cost = F.mse_loss(
            pred_emb[..., -1:, :],                     # 预测轨迹末尾一步
            goal_emb[..., -1:, :].detach(),            # 目标嵌入（阻断梯度）
            reduction="none",
            # 对所有非 batch/sample 维度求和，得 (B, S)
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """
        完整的规划代价计算入口：编码目标→展开预测→计算代价。

        这是推理阶段（如 CEM 规划）的顶层接口，封装了从原始信息到代价值的完整流程：
          1. 将所有张量移至模型所在设备（GPU/CPU）
          2. 提取并编码目标状态（goal frame）
          3. 调用 rollout() 生成候选动作的预测轨迹
          4. 调用 criterion() 计算每个候选计划的代价

        参数：
          info_dict (dict): 信息字典，必须包含：
            - "goal"  : 目标图像张量，形如 (B, 1, T_hist, C, H, W)
                        （包含目标帧，[:, 0] 取第一个样本的目标）
            - "pixels": 初始观测图像，用于 rollout 编码初始状态
            可选包含以 "goal_" 为前缀的键（如 "goal_mask"），会被重命名后传给 goal 编码
          action_candidates (torch.Tensor): 候选动作序列
            - 形如 (B, S, T, action_dim)
            - S 为候选计划数量（如 CEM 每轮采样数）

        返回：
          cost (torch.Tensor): 每个候选动作计划的规划代价，形如 (B, S)
                               代价越低的计划越接近目标状态，应被优先选择

        注意：
          - 此函数会原地修改 info_dict（添加 "goal_emb" 和 "predicted_emb" 等键）
          - goal 编码前会弹出 "action" 键，因为目标帧不需要动作条件
        """

        # 确保 info_dict 中存在目标图像
        assert "goal" in info_dict, "goal not in info_dict"

        # ── 步骤 1：设备对齐 ─────────────────────────────────────────────
        # 获取模型参数所在的设备（可能是 CPU 或 CUDA:N）
        device = next(self.parameters()).device
        # 将 info_dict 中所有张量移至模型设备，避免设备不匹配错误
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # ── 步骤 2：构建目标状态字典 ─────────────────────────────────────
        # 取所有候选计划的第 0 个样本（[:, 0]），因为所有样本共享同一目标
        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        # 将 "goal"（目标图像）复制给 "pixels"，以便 encode() 将其当作观测处理
        goal["pixels"] = goal["goal"]

        # 将 "goal_xxx" 前缀的键重命名为 "xxx"（如 "goal_mask" → "mask"）
        # 这些辅助信息（如掩码、深度图等）可能在编码目标时需要用到
        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_"):]] = goal.pop(k)

        # 目标编码不需要动作条件，移除 "action" 键
        goal.pop("action")
        # 调用 encode() 将目标图像编码为嵌入向量，结果存于 goal["emb"]
        goal = self.encode(goal)

        # ── 步骤 3：执行 rollout 并计算代价 ─────────────────────────────
        # 将目标嵌入存入 info_dict，供 criterion() 使用
        # goal["emb"] 形状 (B, T_goal, D)，unsqueeze(1) 后为 (B, 1, T_goal, D)
        # 此处 S 维度会在 criterion 的 expand_as 中广播
        info_dict["goal_emb"] = goal["emb"]
        # 调用 rollout() 生成候选动作对应的预测嵌入轨迹
        info_dict = self.rollout(info_dict, action_candidates)

        # 调用 criterion() 计算每个候选计划与目标的 MSE 代价
        cost = self.criterion(info_dict)

        return cost
