# FlowContour v2 实现与真实数据冒烟记录

北京时间2026-10-22。对应协议 `flowcontour.full2.v2`，从README提交 `a90427a26d58d5791fa202c5e4debb564f13075e` 开始修改。用户授权修改主线、清理旧代码、提交并做冒烟；本次没有推送，也没有启动正式长训。

## 1. 实际改动

- 保留 MoonViT 缓存、Route-B GT框初始化、128点、L6/F256、CR-MoE E4/Top2及Dense输出头，模型参数名和shape不变。模型共16,933,968参数，完整Flow 13,687,632，特征替换器3,246,336。
- 公共协议和配置集中维护；唯一训练入口为 `tools/train.py`，阶段为混合状态预训练、在线模仿、GDPO。
- 预训练按50%初始／20%连续中间／30%GT傅里叶扰动精修采样，保留原框扰动，学习完整残差。在线模仿每批重新生成自身状态，只训练Flow。
- 生产推理、在线状态生成和GDPO共用唯一AB2实现，两次完整位移、每次4 NFE，阶段条件0/1表示更新／精修，不使用frac。
- GDPO采用最终1/2/4/8输入像素硬NSD等权分数；终点优势乘以均值为1的阶段质量权重，权重detach，不改变终点优势符号。推理关闭额外几何探索。
- 保留低频法向形变及第二阶段1/8局部高幅峰谷候选；局部宽度按实际弧长计算。概率是固定分量条件下的投影高斯代理，不冒充完整Flow精确似然，重放误差门仍严格小于0.002。
- 严格恢复模型、AdamW、调度器与进程RNG；GDPO参考始终取最初在线模仿源，不取恢复后的策略。空批次不计有效更新，不宣称多进程数据流逐位重放。

删除旧 `train.py`、`tools/train_rl.py`、两套旧启动器、旧训练器／记录器／优化器、旧阶段配置、HA-SMoE与GRPO别名壳、重复傅里叶求解，以及frac、full-extrap、Delta奖励、毛刺／自交惩罚、旧进度高斯采样和AMP分支。现在只有一份 `MainlineDiTBlock` 和一份AB2积分实现。没有删除模型、缓存、数据或其他实验输出；旧源码可从Git提交 `a90427a` 在独立checkout恢复。

## 2. 固定测试身份

主线路径 `/home/medteam/Zhrch/FlowContour/2026.10.20/FlowContour` 是同目录 `FlowSnake` 的链接，两者不是两套实现。

| 项目 | 实际值 |
| --- | --- |
| Python | `/home/medteam/miniconda3/envs/sam1_lgz/bin/python` |
| 数据根 | `/home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed` |
| 数据清单SHA256 | `182f75e0fb2a13c51121eea8651c3b7e4c031f9f346269e10a03721813e09b9c` |
| MoonViT缓存 | `/home/medteam/Zhrch/DiffusionSnake-12-30/data/sagittal_moonvit_cache` |
| Dense源 | 原 `pure2d_mainline_l6_f256_routeb_v410_moonvit_cached_flowtune60k_from40000_v1/checkpoints/step_19000.pt` |
| Dense源SHA256 | `a337ba1566fe423c10a82dc4c08f8d6936ce8fc49ff1d61c8f735435854a337f` |
| 固定Val37清单SHA256 | `24a4f19651edb5d187029f0255e2b59f9dce40f320ee29c14b709e8e92e6e6ad` |
| 执行代码、配置与位移统计聚合SHA256 | `50af502dc39c455b48d35315e9690b90e458bdda715cea74b57265a4a3ad000e` |
| 最终STATUS.json SHA256 | `6014767c09c4c52c904c7376ccc0bb53eefff2f05307acd78e3662d045ce07ce` |
| seed | 20260731 |
| 计算精度 | 模型Float32；缓存Float16；几何逆变换Float64 |

Dense源全部156个张量逐项精确恢复，只允许新增44个router／专家张量；之后阶段严格加载全部200个张量，没有partial copy或shape skip。原源文件未改。

## 3. 最终全链路复测

最终输出根：

```text
/home/medteam/Zhrch/FlowContour/2026.10.20/FlowSnake/data/outputs/full2_v2_smoke_20261022_r4
```

流水线在北京时间18:59:59启动，19:03:43结束，`STATUS.json` 为 `COMPLETE/PASS`。每个阶段的完整命令、代码逐文件SHA和结果均保存在该STATUS及阶段RESULT中。

| 检查 | 实际执行 | 结果 |
| --- | --- | --- |
| CPU单元及完整模型测试 | 15项，含真实架构反传与部署前向 | 全部通过，未跳过完整模型测试 |
| 预训练 | 从签名Dense初始化，2次真实有效更新 | Flow 157个张量变化，替换器6个变化 |
| 预训练严格恢复 | 恢复step1，再执行step2 | 模型／优化器／调度器／RNG核验均通过 |
| 在线模仿 | 从冒烟预训练step2初始化，2次真实有效更新 | Flow 184个张量变化，替换器及冻结张量0变化 |
| GDPO | 从冒烟在线模仿step2初始化，2次有效组更新 | Flow 178个张量变化，冻结张量0变化 |
| GDPO严格恢复 | 原在线源不变，恢复GDPO step1，再执行step2 | 完整恢复检查通过，冻结张量0变化 |
| 每阶段末生产前向 | 真实批次、同一部署代码 | 恰好8次去噪器调用、坐标有限 |
| 独立推理入口 | Dev8清单顺序首两张前景切片 | `SMOKE_PASS`，严格加载200张量，8 NFE |

两次GDPO组更新分别使用8条完整两步轨迹。日志中的局部配额均为 `[0,1]`，PPO未变策略概率重放差为0，信用均值守恒误差最大 `1.49e-8`，信用符号违反数0。保存的最后一组候选采样逆变换最大误差为 `2.99e-7`；原0.002门限未放宽。

GDPO梯度日志记录裁剪前范数，实际仍按0.25裁剪。第二次组更新终点Q均值0.629434、实例内候选Q标准差均值0.008070。这只是4个训练实例的单组诊断，不是验证集质量，也不代表奖励已经最优；该组8px项有16/32候选实例达到1，仍须在完整实验中检查饱和。

最终GDPO冒烟step2 SHA256：`186efb630edfd8a62413b3ea876c19ed2bb1d070eaacb44c10d4f6ca152643a5`。

测试结果文件：

```text
STATUS.json                              完整命令及全链路状态
cpu_tests.log                            15项测试原始输出
{pretrain,strict_resume,online,gdpo,gdpo_resume}/RESULT.json
{pretrain,strict_resume,online,gdpo,gdpo_resume}/logs.jsonl
gdpo/actual_training_candidates.pt        真实8候选、状态、几何分量、GT与分数
gdpo/actual_training_candidates.png       原图和同位置局部放大
inference/SMOKE_RESULTS.json              非报告推理与2D诊断
```

实际候选图已经人工查看：绿色为GT、橙色为采样轮廓，局部差异可见，但轮廓仍有明显误差，不能用这张短训图声称方法有效。独立推理处理 `sub-verse004` 切片5和6，前景很小，一张未形成有效预测，失败计入后两张平均Dice为0.2。此低分不隐藏，也不作为正式VerSe模型指标；这条链只做了2步预训练、2步在线模仿和2步GDPO，并且新阶段条件尚未经过正式训练。

## 4. 资源、失败与边界

没有严格空闲卡。本次按既有“可共用相对空闲卡”授权，显式使用物理GPU2进行功能预检；两次资源检查间隔15秒，已有PID记录为1029549，未停止该任务。实际batch1、0 worker，仅检查固定Val37面板的两行。全部输出 `functional_smoke=true`、`reportable=false`。

正式空闲门仍为显存≤20MiB、利用率0、无compute app。共用／CPU功能冒烟不批准正式训练，冒烟权重不能初始化正式后训练，不能生成完整可报告评估。正式预训练预检必须在严格空闲卡上实际测试batch48及完整80行面板；本次尚未完成该项显存验收。

保留的排错过程：

1. 首次数据核验误指向VerSe20根目录，被固定病例清单拦截；改用正确签名数据，没有放松病例限制。
2. 单元测试参数被旧全局配置解析器读取；修为仅显式传入配置参数时解析。
3. 完整模型测试中“位移求和”与“终点减起点”的浮点表示差超过原 `1e-5` 比较阈值；统一为生产终点定义，没有放宽阈值。
4. `r1` 的GPU0利用率55/56%超过共用门，训练开始前拒绝；随后在GPU2重新预检，不停止其他作业、不放宽门限。
5. `r2/r3` 全链通过后继续清理与身份隔离；最终 `r4` 对最终执行代码重新完整复测。所有旧结果保留，不把先前版本的通过状态替代最终复测。

## 5. 复现与下一步

[主README](../README.md)给出统一CLI用法。最终测试的逐条真实命令在输出 `STATUS.json`；有限顺序测试包装位于 `/home/medteam/Zhrch/codex_tasks/flowcontour_v2_20261022/run_smoke.py`。它只生成新输出目录，不启动长训。

CPU全测：

```bash
CUDA_VISIBLE_DEVICES='' FLOWCONTOUR_FULL_SMOKE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/medteam/miniconda3/envs/sam1_lgz/bin/python -m unittest discover -v
```

后续先审核提交，再完成正式batch预检和有预算登记的三段训练、固定端点与同预算对照。当前只证明实现链路可执行，尚未证明新流程、高频探索或1/2/4/8奖励提高VerSe分割质量；BTCV已有实验与固定847切片评价标准不受本次主线实现改动影响。
