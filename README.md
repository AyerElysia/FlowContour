# FlowContour：基于 MoonViT 与 CR-MoE 的分布化医学轮廓演化

FlowContour 将整条闭合轮廓的联合位移视为条件动作，以 Flow Matching 学习动作分布。CR-MoE（Contour-Routed Mixture-of-Experts）提供轮廓级条件化适配；后训练分为在线模仿微调与 GDPO 几何探索强化学习。

当前主线（`flowcontour.full2.v2`）为：

```text
离线 MoonViT layer-18 特征 + GT box / anatomical class
    ↓
混合状态预训练：初始、连续中间、GT 附近扰动精修
    ↓
后训练一：在模型自身产生的轮廓上进行在线模仿微调
    ↓
后训练二：GDPO 几何探索，最终奖励与阶段信用分配
    ↓
两步完整位移推理：每步重新取特征、4 次 AB2 求值，共 8 NFE
```

## 1. 版本与验证范围

**三段训练、两步完整位移、局部几何探索与多尺度终点信用已实现。功能冒烟不等于分割精度验证。** 当前默认仍为128点、MoonViT＋CR-MoE；不把旧检查点或历史BTCV分数重新解释为新版VerSe成绩。验证记录见[实现与冒烟报告](docs/full2_v2_verification_20261022.md)。

| 项目 | 历史 v1 | 当前 v2 |
| --- | --- | --- |
| 视觉与轮廓网络 | MoonViT 缓存 + CR-MoE，128 点 | 保留架构与数据接口 |
| 监督状态 | 线性插值进度的混合采样 | 50% 初始、20% 连续中间、30% GT 附近精修 |
| 在线模仿 | 无独立入口 | 基于自身生成状态的完整残差监督 |
| GDPO 轨迹 | 五个分数动作，full-extrap | 两个完整动作，与部署一致 |
| 策略更新信号 | 各阶段独立归一化优势 | 最终轨迹优势 × 阶段质量权重 |
| 奖励 | delta-NSD@2px，带毛刺惩罚 | NSD@1/2/4/8px 等权，无毛刺／自交惩罚 |
| 几何探索 | 低频傅里叶法向场 | 保留低频和局部高幅单峰／单谷，另设匹配对照 |
| 部署位移 | `[0.6667, 1.0]` | 两步均执行完整预测位移，不使用 frac |
| 阶段条件 `s` | 由残余位移比例累计 | 首次更新=0，精修更新=1，不表示GT完成比例 |

只保留一个训练入口 `tools/train.py`，由 `--stage pretrain|online|gdpo` 选择阶段。`configs/common.yaml` 集中维护网络和公共协议，三个阶段配置只声明差异；没有复制三套DiT或求解器。

旧五阶段、frac、full-extrap、Delta奖励、毛刺／自交惩罚、旧名兼容壳和重复训练器已从执行代码移除。历史命令见[v1归档](docs/mainline_legacy_v1.md)，复现须在独立checkout检出 `a90427a`；不要在当前版本执行已删除的旧入口。

## 2. 方法：条件动作分布与多路径

### 2.1 从单一位移到分布化动作

给定图像 `I` 和当前轮廓 `C_t`，确定性方法预测一个更新 `Δ_t = f(I, C_t)`。在固定权重和相同输入下，其几何更新相同。

FlowContour 建模条件动作分布 `p(Δ_t | I, C_t)`：通过少步 ODE 求解，将潜变量 `z_t` 映射为整条轮廓的联合位移。对于相同输入，改变潜变量可以得到不同候选，而不是先将所有可能修正压缩成一个位移。

```text
固定 I、C₀
  ├─ z₁ → Δ₁ → C₁ → 重新观测 → Δ₂ → C₂
  ├─ z₁′ → Δ₁′ → C₁′ → 重新观测 → Δ₂′ → C₂′
  └─ z₁″ → Δ₁″ → C₁″ → 重新观测 → Δ₂″ → C₂″
```

多路径来自连续的条件动作采样与视觉反馈，不是多个器官输出，也不是多个专家各自输出一条轮廓。CR-MoE 为同一次动作生成提供条件化适配，不能把专家数量当作路径数量。

模型不保证每个种子都产生明显不同的位移，也不保证每个候选都更好。困难边界的标注歧义是研究动机；本仓库尚未提供多标注者数据证据，不能把人工示意或跨样本差异当成已经证明的条件多模态。

### 2.2 模型多样性与训练探索必须分开

- **模型自身多样性**：固定同一图像、同一轮廓和阶段条件，只改变潜变量，比较输出位移。
- **GDPO 几何探索**：在生成动作上额外构造受控形变，用于寻找有价值的训练候选；不是部署时必开的噪声。
- **完整轨迹分散程度**：同时包含输入状态已经不同造成的差异，不能直接当作同一状态下的随机性。

阶段间比较多样性时，每个阶段选定或遍历明确的锚定状态，在每个固定状态下重复采样，再跨锚点统计。不能将来自不同输入的阶段二结果混在一起估计条件随机性；跨实例比较还应登记尺度归一化方法。

## 3. 保持不变的网络与数据边界

当前网络是逐张矢状位切片的纯 2D 模型。只读取中心切片及其 MoonViT `layer_18` 缓存，不使用 Memory、相邻切片特征、3D 网络、SAM 或内部检测器。GT box 与 GT anatomical class 属于 `oracle_prompt`，不能与自动提示结果混榜。

```text
中心切片 → 离线 MoonViT（448×448，patch 14，layer 18）
         → float16 token [1152, Htoken, Wtoken]
         → 特征替换器 → 128×128 轮廓特征网格

GT box / class → Route-B box-octagon → 128 点初始轮廓
              → CR-MoE Flow → 联合轮廓动作
```

CR-MoE 对当前轮廓位置的局部图像 token 做 mean/std 汇聚，一个共享 router 一次输出 `[N, 3, 4]` logits，分别用于 DiT 第 2/4/6 层。每层的 Top-2 权重无点维度，整条轮廓共享专家组合；共享 Dense FFN 始终执行，稀疏专家仅提供残差，速度输出头保持 Dense。

主干 global/local cross-attention 与时间／阶段 AdaLN 保留，但不作为 router 的额外输入。没有点级路由、独立逐层 router、全层 MoE 或输出头混合。

| 组成 | 参数量 | 混合状态预训练 | 在线模仿 | GDPO |
| --- | ---: | --- | --- | --- |
| 共享 Flow | 11,127,108 | 训练 | 训练 | 训练 |
| CR-MoE router 与残差专家 | 2,560,524 | 训练 | 训练 | 训练 |
| 完整 Flow | 13,687,632 | 训练 | 训练 | 训练 |
| MoonViT 特征替换器 | 3,246,336 | 训练 | 冻结 | 冻结 |
| MoonViT 编码器 | 不进入训练图 | 冻结缓存 | 冻结缓存 | 冻结缓存 |
| 总模型 | 16,933,968 | 16,933,968 可训练 | 13,687,632 可训练 | 13,687,632 可训练 |

128 点暂保留为主线规格。256 点是候选，不因追加训练改善就直接替换默认值；其点数配置、目标构造、形变宽度、检查点和推理接口仍需独立验收。

## 4. 混合状态预训练：完整残差与末端精修

迁移起点为近期实验中的以下配方，不宣称比例已经最优：

| 状态来源 | 概率 | 构造与目的 |
| --- | ---: | --- |
| 初始轮廓 | 50% | GT box 初始化，保留原框扰动，学习大范围定位修正 |
| 连续中间轮廓 | 20% | 在初始轮廓与对齐 GT 之间连续采样，覆盖中间状态 |
| GT 附近精修轮廓 | 30% | 在 GT 上施加傅里叶几何扰动，学习到位后的剩余修正 |

这里的 50/20/30 是状态类别概率，不是执行位移的比例。中间轮廓仍可被采样，不再特别偏重几个固定线性插值位置。原框扰动的概率、平移、缩放和边缘参数保留，并随配置签名。

无论来自哪类状态，监督目标均为 `Δ* = C_GT − C_current`，不乘外层 frac。GT 的点数、方向、循环起点与坐标尺度必须按同一规则对齐，再构造 Flow Matching 的噪声插值与速度监督。不能把大幅初始化任务改成只训练 GT 周围的小扰动。

该阶段训练完整Flow和特征替换器。默认AdamW、LR 1e-5、warmup 1000、batch48、60000次有效更新；以有效更新数为唯一停止上限，没有另一个epoch上限提前截断。每1000步保存，不自动删除历史权重。

精修使用8个正交傅里叶模式，系数标准差为3个网络输入像素，沿GT轮廓法向施加；128点上的单点期望RMS约0.75px，而不是每点固定3px。参数见公共配置，不宣称是最优困难度。

首次初始化可从签名Dense源step19000受控weights-only迁移：`a337ba1566fe423c10a82dc4c08f8d6936ce8fc49ff1d61c8f735435854a337f`。全部旧张量必须精确匹配，只允许新增router和第2/4/6层专家四组前缀，随后新建AdamW；没有partial copy或shape skip。旧检查点只作为初始化来源，不获得v2推理身份。

## 5. 后训练一：在线模仿微调

人工扰动能扩展状态覆盖，但不能完全替代模型自身的错误。在线模仿直接在当前策略的生成状态上建立监督：

1. 从预训练检查点出发，按两步完整位移协议生成轮廓。
2. 收集自身产生的中间状态；状态生成不反传跨轨迹梯度。
3. 用同一 GT 对齐规则构造该状态到 GT 的完整残差，执行监督 Flow Matching 更新。
4. 冻结视觉缓存和特征替换器，只更新完整 Flow，包括 CR-MoE router 与专家。

每个batch用当前策略重新生成 `C₀→C₁→C₂`，每个实例等概率选择首次更新输入C₀或精修输入C₁，阶段标签分别为0或1；不是从GT插值冒充在线状态。默认500次有效更新、batch1、LR 1e-5，无重新预热。对照应匹配额外监督更新数，区分“在线状态有用”与“仅多训了若干步”；500步是继承起点，不是已证明的VerSe最优预算。

## 6. 后训练二：GDPO 几何探索强化学习

GDPO 在自身生成的两步轨迹上，根据最终分割质量优化动作倾向。名称统一为 GDPO；组相对优势、PPO 式裁剪、参考策略约束及动作概率实现都必须按实际代码说明，不能仅靠改名宣称换了优化算法。

### 6.1 探索：整体低频与局部高幅峰谷

保持 ODE 生成过程不变，在生成的轮廓动作上沿法向施加受控形变：低频傅里叶场负责连续整体修正，局部单峰／单谷负责局部突起或凹陷候选。候选变差是允许的，目标是产生有价值的修正并改善训练后的策略，而不是让每条候选都获胜。

当前实现迁移以下BTCV候选数值，**功能已实现，VerSe上的额外收益未验证**：

| 项目 | BTCV 候选配方 |
| --- | --- |
| 每组轨迹数 | 8，每条两步完整动作 |
| 低频 | 8 个正交模式，两阶段系数尺度 `[1.2, 0.8]` 输入像素 |
| 局部触发 | 第一步不加；第二步 8 个候选中 1 个叠加局部项 |
| 峰／谷 | 正负等概率，位置不使用 GT 选优 |
| 幅度 | 中心 2/3/4px，概率 0.70/0.25/0.05，连续高度标准差 1px |
| 支撑宽度 | 轮廓周长的8/128或12/128，概率0.30/0.70；按实际弧长而非固定索引距离计算 |

幅度中心不是硬上限；傅里叶系数尺度也不是每个顶点固定移动的距离。换点数时应按归一化弧长重新定义局部宽度，不可机械照搬点数。

低频单独配置保留为同预算消融，不能悄悄删掉局部项后继续称作混合探索。局部项的分量条件、投影、概率与重放规则必须匹配；若使用投影高斯代理，不得称作完整 Flow 动作的精确似然。概率重放误差门与 PPO 的 approx-KL 门分别记录，不绕过检查或放宽阈值掩盖失败。

### 6.2 奖励：多尺度边界质量

新主线的候选质量分数为硬阈值多尺度 NSD：

```text
Q(C) = [NSD@1px(C, GT) + NSD@2px(C, GT)
      + NSD@4px(C, GT) + NSD@8px(C, GT)] / 4
R_rollout = Q(C_final)
```

不使用0.5px项，不加入毛刺或自交惩罚。这里明确指**网络输入图像的二维像素**：预测和GT从特征网格乘stride4转换，通常对应512×512输入；不是医学原图坐标，也不是VerSe三维毫米。若需医学原图容差，必须另做逆仿射协议，不混称同一分数。

栅格NSD沿用两个方向的边界覆盖率各占一半、OpenCV `DIST_L2/mask3`距离变换；任一边界为空时记0。四个硬阈值等权平均，不将软分数混入。

最终分数使用采样轨迹本身的质量，不使用旧版“采样外推端点减确定性外推端点”的奖励。组内比较由优势归一化完成。Dice、逐器官指标和物理空间指标仍需单独评估，不能拿更宽容的 8px 得分上涨替代质量提升。

新 1/2/4/8px 组合尚无稳定训练收益结论。验收包括候选排序、单项饱和率、组内分差及训练后固定指标；旧尺度对照保留。软边界分数如被用于消融，必须单独命名，不能冒充硬阈值 NSD。

### 6.3 信用：最终结果决定优势，阶段质量决定分配

在同一实例的 K 条轨迹内，用最终奖励构造组相对优势 `A_k`。设 `q[k,t] = Q(C[k,t+1])` 是第 t 个动作之后的非负质量，迁移现有实验的分配规则：

```text
A_k = clip((R_k − mean_k R) / max(std_k R, std_floor))
w[k,t] = T × q[k,t] / sum_t q[k,t]     # 分母为零时，各阶段权重取 1
A[k,t] = stop_gradient(w[k,t] × A_k)
T = 2
```

该归一化配合按阶段取平均的策略损失，保证 `mean_t w[k,t] = 1`、`mean_t A[k,t] = A_k`；不能把它误写成未归一化的“总和守恒”。阶段权重不能改变最终优势的正负号；最终优势为零时，不能独立产生阶段奖励更新。

它是阶段质量加权的启发式信用分配，不宣称是精确因果归因。不得改回各阶段独立归一化即时优势，也不能只计算或记录最终优势，却在 PPO 更新中使用另一个信号。

### 6.4 优化与有效更新

保留 `K=8`、PPO epochs2、clip0.05、approx-KL0.002、KL beta0.01、优势标准差下限0.1、优势裁剪2.0、梯度裁剪0.25、LR4e-8；默认3000次有效更新。参考策略固定为最初的在线模仿源，恢复训练不会把参考重置为恢复后的策略。

空 batch 不计有效更新。真实局部触发、冻结张量零变化、有限梯度、概率重放及信用检查均通过后才可启动正式版本；短预检权重不作为正式训练起点。

## 7. 目标推理协议：两步完整位移，8 NFE

```text
C₀ ── AB2×4 → 完整 Δ₁ ──→ C₁
C₁ ── 重新采样图像特征 ── AB2×4 → 完整 Δ₂ ──→ C₂
```

两个动作均执行全部预测位移，不使用 frac，也不构造 full-extrap 端点。4 次 AB2 速度场求值属于数值求解，不是 4 个 RL 动作；两步共有 8 NFE。训练、在线状态生成、验证与部署必须共用同一生产采样语义。

阶段条件固定 `s=0` 表示首次更新、`s=1` 表示精修更新，不再由frac累计，不表示GT完成比例。预训练初始状态用0、GT附近状态用1，连续中间状态同时覆盖两个标签；在线模仿、GDPO与生产推理均采用相同定义，第二步不假定已经到达GT。

固定评价关闭额外几何探索，登记 seed、每阶段初始 latent、尺度、采样数和求解器。展示多路径时则固定输入、改变明确登记的潜变量种子，不挑 GT 最优轨迹冒充单次预测。

特别注意：本仓库 v1/v2 使用真实 AB2 和 latent scale 1.0；近期 BTCV 评价使用原 V5 求值与零 latent。这两个协议不相同，不能改换 latent 或求解器后仍宣称同一对照。

## 8. 数据、缓存与评价标准

### 8.1 VerSe 主线固定边界

| 用途 | 划分 | 数量 | 约束 |
| --- | --- | ---: | --- |
| 训练，包括两段后训练 | Train72 | 72 cases / 13,261 rows | 可以更新参数 |
| 开发选择 | Val37 | 37 cases / 5,914 rows | 固定 80-row panel，不更新参数 |
| 最终开发评估 | Dev8 | 8 cases / 1,123 rows | 不参与调参 |
| 锁定病例 | sub-verse010/011/013 | 3 cases | 禁止读取 |

`slice_manifest.csv` 至少含 `split,case_id,slice_idx,image_path,mask_path`；路径绝对或相对数据根目录，case 内切片索引唯一连续。3D 评价还需要 case metadata、NIfTI 与签名 centroid JSON；不能拿 2D 切片指标替代完整病例指标。

MoonViT 缓存使用 `{training,validation,test}/<case_id>/x<slice_idx:04d>.npz`，固定 448 输入、patch 14、`layer_18`、1152 通道、float16、`center_only`。保留 `grid_hw/orig_hw/resized_hw/padded_hw/pad/scale/patch_size/input_size/normalization/checkpoint/case_id/slice_idx` 校验；不能缺失后降级到匹配子集。缓存来源标识为 `Eagle/Embodied/work_dirs/1232_final_locany_full_more10000/checkpoint-3000`，不能更换权重重建后仍称同一血缘。

VerSe 报告保留前景与实例指标、完整病例的 Dice/ID rate/maxHD/dmean，以及固定协议的实际轮廓图。像素尺度和物理尺度、逐切片与逐病例加权、oracle 与自动提示必须明确区分，不为了数值更高而切换口径。

### 8.2 BTCV 证据不直接转成 VerSe 成绩

近期 BTCV 对照固定：全部 847 切片等权的联合前景 Dice，保留失败；GT box/class、Memory off、seed20260731、两步完整位移、每步 4 次原 V5 求值、零 latent、推理关闭探索。逐器官 Dice 和多尺度 NSD 只是辅助，不能替换主指标。

该口径仅用于 BTCV 历史对照。不能将其分数、检查点、优化预算或求解实现直接移作 MoonViT＋CR-MoE VerSe 成绩。完整、带限制的结果见[近期证据快照](docs/mainline_evidence_20261022.md)。

## 9. 仓库入口与运行状态

执行代码只保留以下主线：

```text
configs/common.yaml                        单一公共网络、数据与训练协议
configs/{pretrain,online,gdpo}.yaml          三个阶段的差异配置
lib/networks/mainline.py                    MoonViT cache + Flow
lib/networks/diffusion/mainline_denoiser.py  6-layer / 256-dim 主干
lib/networks/diffusion/cr_moe.py             CR-MoE 主实现
lib/networks/diffusion/flow_matching_evolution.py
                                            唯一AB2与完整动作采样、FM监督
lib/protocol.py                            协议身份和阶段条件
lib/rl/geometry.py                         几何候选、条件投影概率与KL
lib/train/states.py                        混合状态与终点信用分配
lib/train/engine.py                        共用训练、验证、严格恢复与检查点
lib/train/wrapper.py                       保留net.*权重命名，不另建训练器
lib/train/rewards/region_reward.py          绝对多尺度NSD与报告指标
tools/verify_installation.py                CPU 数据、权重、数学核验
tools/train.py                             唯一三阶段预检／正式训练入口
tools/infer.py                             生产推理、完整评估和冒烟可视化
lib/visualizers/contours.py                 实际8候选及同位置局部放大
```

当前可复现环境为 Python 3.11、PyTorch 2.4.1 + CUDA 12.1，其他依赖见 `requirements.txt`。数据、权重、约 84GB MoonViT 缓存与训练输出均不进入 Git；正式命令显式传入绝对路径，不能把说明性默认路径当作真实权重。

正式训练与完整报告推理仍要求物理GPU连续两次、间隔15秒满足显存≤20MiB、利用率0、compute app为空。已有共用授权时，功能预检或非报告推理可显式加 `--allow-shared-gpu`：两次均须显存≤30000MiB、利用率≤50%，记录已有PID，不终止其他作业。**共用或CPU冒烟不能批准正式训练，也不能用预检权重初始化正式后训练或产生完整报告。**

### 9.1 初始化与CPU核验

以下命令从仓库根目录执行，路径按实际位置填写，输出目录每次必须新建：

```bash
PYTHON=/path/to/python
DATA_ROOT=/path/to/sagittal_2d_fixed
MANIFEST="$DATA_ROOT/manifests/slice_manifest.csv"
CACHE=/path/to/sagittal_moonvit_cache
SOURCE=/path/to/pure2d_moonvit_dense_step19000.pt
SOURCE_SHA=a337ba1566fe423c10a82dc4c08f8d6936ce8fc49ff1d61c8f735435854a337f

"$PYTHON" tools/verify_installation.py \
  --data-root "$DATA_ROOT" --slice-manifest "$MANIFEST" --moonvit-cache "$CACHE" \
  --checkpoint "$SOURCE" --checkpoint-sha256 "$SOURCE_SHA" \
  --expected-step 19000 --initialize-dense
```

### 9.2 预训练：两次真实更新预检，然后正式运行

```bash
"$PYTHON" tools/train.py --stage pretrain --mode preflight --gpu 0 \
  --source-checkpoint "$SOURCE" --source-sha256 "$SOURCE_SHA" --source-step 19000 \
  --initialize-dense --data-root "$DATA_ROOT" --slice-manifest "$MANIFEST" \
  --moonvit-cache "$CACHE" --output data/outputs/pretrain_preflight_YYYYMMDD

"$PYTHON" tools/train.py --stage pretrain --mode train --gpu 0 \
  --source-checkpoint "$SOURCE" --source-sha256 "$SOURCE_SHA" --source-step 19000 \
  --initialize-dense --data-root "$DATA_ROOT" --slice-manifest "$MANIFEST" \
  --moonvit-cache "$CACHE" --preflight-output data/outputs/pretrain_preflight_YYYYMMDD \
  --output data/outputs/pretrain_YYYYMMDD
```

正式预检使用配置中的真实batch（预训练为48）、0 worker、完整80行验证面板和两次有效更新。只有CPU／共用GPU的功能冒烟才缩成batch1与两行验证，并标记 `functional_smoke=true`，不能批准正式训练。运行中的实际batch和worker另外记录，不把配置值冒充实际值。

当前模型训练和推理固定使用Float32，不保留旧AMP分支；缓存文件仍为Float16，几何概率逆变换使用Float64。完整batch48的显存验收尚未执行，需在严格空闲卡上完成正式预检。身份包含代码、合并配置、源SHA、数据清单、缓存路径、设备与seed，不复用预检模型作为正式起点。

### 9.3 在线模仿与GDPO

选择正式预训练产生的检查点，登记绝对路径、SHA256和step，再执行在线模仿：

```bash
"$PYTHON" tools/train.py --stage online --mode preflight --gpu 0 \
  --source-checkpoint "$PRETRAIN_CHECKPOINT" --source-sha256 "$PRETRAIN_SHA" \
  --source-step "$PRETRAIN_STEP" --data-root "$DATA_ROOT" \
  --slice-manifest "$MANIFEST" --moonvit-cache "$CACHE" \
  --output data/outputs/online_preflight_YYYYMMDD
```

通过后，用同一源、`--mode train`、`--preflight-output`和另一个新输出目录启动正式在线模仿。GDPO同理，将阶段改为 `--stage gdpo`，源必须换成正式在线模仿检查点；源的协议、阶段、SHA和step全部严格校验。不能从旧v1模型或预检权重直接跳过前序阶段。

中断恢复继续传入同一个原始源，并增加 `--resume-checkpoint /path/to/step_N.pt`、使用新输出目录。严格恢复模型、AdamW、调度器与进程RNG，冻结张量相对原始源不变，GDPO参考策略仍来自原始源；不宣称多进程数据流逐位重放。不覆盖或删除旧权重。

### 9.4 生产推理与测试

```bash
"$PYTHON" tools/infer.py --gpu 0 --checkpoint "$GDPO_CHECKPOINT" \
  --result-dir data/outputs/inference_smoke_YYYYMMDD \
  --data-root "$DATA_ROOT" --slice-manifest "$MANIFEST" \
  --case-metadata "$DATA_ROOT/manifests/case_metadata.csv" \
  --locate-feat-cache-root "$CACHE" --batch-size 1 --num-workers 0 --smoke-slices 2

CUDA_VISIBLE_DEVICES='' FLOWCONTOUR_FULL_SMOKE=1 OMP_NUM_THREADS=4 \
  "$PYTHON" -m unittest discover -v
```

推理先输出 `SMOKE_PASS`、`reportable=false`。正式模型的完整Dev8评估将 `--smoke-slices` 改为0、另用新结果目录；必须处理全部1123张，输出2D、实例与完整3D指标及实际轮廓图。冒烟检查点被禁止生成可报告的完整评估。

## 10. 实现验收与待验证问题

- [x] 混合采样、原框扰动、在线状态与阶段标签已接入；单一DiT与单一AB2有测试。
- [x] 两步完整动作、各4次求值在实际模型与真实数据冒烟中核验。
- [x] 终点优势参与GDPO更新，阶段权重detach，符号及均值信用守恒。
- [x] 绝对1/2/4/8px奖励定义固定，无毛刺／自交项；同一批候选支持离线诊断。
- [x] 局部峰谷实际触发，概率重放保持严格小于0.002；保存实际8候选与局部放大。
- [x] 预训练更新Flow与替换器，后训练只更新Flow，冻结张量变化为0。
- [x] 有效更新循环、完整检查点、源身份及严格恢复均已实现并做短测。
- [ ] 在严格空闲GPU完成正式预检，随后运行有预算登记的正式三段训练。
- [ ] 完整预算下检验奖励排序、饱和及组内差异，并验证实际分割收益。
- [ ] 同预算低频对照、额外种子和固定端点评估完成，保留负结果，不把开发集最佳值当独立测试。
- [x] 真实Dev8非报告推理冒烟；完整固定划分评价与泛化结论尚未完成。

遇到 OOM、非有限值、概率重放失败、冻结张量变化、源身份变化或进程异常，保留日志、失败快照和检查点，先定位再开新版本预检；不覆盖旧模型或自动放宽门限。兼容的参数 shape 不等于兼容的新协议，也不能沿用旧预检批准新训练。

历史源码删除可由Git中的 `a90427a` 恢复；没有删除模型、数据或运行中实验。此次只进行有限功能冒烟，不自动启动长训；提交供审核，不推送。负结果与运行记录继续同步原实验总结。
