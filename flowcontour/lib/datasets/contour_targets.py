"""VerSe 切片共用的轮廓目标构造。

从旧 VOC loader 抽出、与数据集无关的部分：对齐点数、方向和循环起点，
再构造检测框、初始化多边形和演化 GT。不要再去 import 已退役的自然图像数据集。
"""

from __future__ import annotations

import numpy as np

from lib.utils.snake import snake_config, snake_voc_utils


class ContourTargetMixin:
    """为一条多边形构造检测、初始化和演化目标。"""

    def transform_original_data(
        self, instance_polys, flipped, width, trans_output, inp_out_hw
    ):
        """将原始图像坐标系下的实例多边形，经过数据增强变换后映射到模型输出网格。

        处理流程：
        1. 将每个实例的扁平化多边形坐标重塑为 (N, 2) 的点列格式；
        2. 若当前训练样本启用了水平翻转（flipped=True），先沿图像垂直中轴镜像所有点的 x 坐标；
        3. 调用仿射变换将多边形从原始图像空间映射到网络输出特征图空间（output_h × output_w）。

        Args:
            instance_polys: 原始多边形列表，每个元素是一个实例，实例内包含若干多边形对应的一维序列；
                            每个多边形对应一维序列 [x1, y1, x2, y2, ...]。
            flipped: 布尔标志，表示当前样本是否经过随机水平翻转；训练时由数据增强流水线注入。
            width: 原始输入图像的宽度（像素），用于计算翻转后的镜像坐标。
            trans_output: 2×3 仿射变换矩阵，将原始图像坐标映射到输出网格坐标；
                          由数据加载器根据随机裁剪/缩放/旋转等增强策略生成。
            inp_out_hw: 输入/输出尺寸元组，其中索引 [2:] 为输出特征图的高和宽 (output_h, output_w)。

        Returns:
            list: 变换后的实例列表，每个实例包含若干已映射到输出网格的多边形点列。
        """
        # 从 inp_out_hw 中提取输出特征图的高和宽
        # inp_out_hw 通常是一个包含 [input_h, input_w, output_h, output_w] 的元组
        # 这里取后两个元素，即模型输出特征图的尺寸
        output_h, output_w = inp_out_hw[2:]

        # 初始化空列表，用于存储所有变换后的实例
        # 每个实例对应图像中的一个目标对象（如一个椎体轮廓）
        transformed_instances = []

        # 逐个实例处理
        # instance_polys 是一个列表，每个元素代表一个实例的多边形集合
        for instance in instance_polys:
            # 将每个多边形的扁平坐标重塑为 (N, 2) 的点列
            # 原始格式是一维数组 [x1, y1, x2, y2, ...]，reshape(-1, 2) 将其变为 [[x1, y1], [x2, y2], ...]
            # -1 表示自动计算行数，2 表示每行 2 列（x 和 y 坐标）
            polygons = [polygon.reshape(-1, 2) for polygon in instance]

            # 若当前样本经过水平翻转，先镜像所有多边形的 x 坐标
            # flipped 由训练时的随机数据增强决定，用于增加数据多样性
            if flipped:
                # 初始化临时列表，存储翻转后的多边形
                flipped_polygons = []

                # 逐个多边形进行水平翻转
                for polygon in polygons:
                    # 沿图像垂直中轴镜像：新 x = width - 原 x - 1
                    # 减 1 是因为像素坐标从 0 开始，width 是总像素数
                    # 例如：图像宽 512 像素，坐标范围是 0~511
                    # 若某点 x=0（最左端），翻转后应在 x=511（最右端）：511 = 512 - 0 - 1
                    polygon[:, 0] = width - np.asarray(polygon[:, 0]) - 1

                    # 必须使用 .copy() 创建副本
                    # 原因：NumPy 数组是引用传递，如果不复制，后续 transform_polys 可能修改到原数组
                    # 导致数据污染，影响其他样本的处理
                    flipped_polygons.append(polygon.copy())

                # 用翻转后的多边形替换原始多边形，供后续仿射变换使用
                polygons = flipped_polygons

            # 将翻转（或未翻转）后的多边形通过仿射变换映射到输出网格空间
            # snake_voc_utils.transform_polys 执行以下操作：
            # 1. 使用 trans_output 矩阵对多边形坐标进行线性变换（缩放、旋转、平移）
            # 2. 将变换后的坐标裁剪到 [0, output_w-1] 和 [0, output_h-1] 范围内
            # 3. 返回映射到输出特征图坐标系的多边形点列
            transformed_instances.append(
                snake_voc_utils.transform_polys(
                    polygons, trans_output, output_h, output_w
                )
            )

        # 返回所有实例的变换结果
        # 返回值是一个列表，每个元素对应一个实例，包含其所有多边形的变换后坐标
        return transformed_instances

    def get_valid_polys(self, instance_polys, inp_out_hw):
        """去掉过短多边形，裁剪到输出范围，统一成顺时针且去重点。

        输入是 transform_original_data 的输出（已映射到 output_h × output_w 网格），
        但那里的结果是"能用"而不是"干净"：可能带越界浮点、面积过小的碎片、
        方向不一致、以及重复点。本函数负责把这些脏数据清理成后续
        get_extreme_points / prepare_evolution 可以直接消费的规范多边形。

        注意：所有过滤都是"丢弃"语义——不合格的多边形直接消失，
        因此返回结果中某个实例的多边形数量可能比输入更少（甚至为空列表）。
        """
        # 从 inp_out_hw 取出输出网格的高宽，作为裁剪的合法上界
        output_h, output_w = inp_out_hw[2:]

        # 逐实例收集清洗后的多边形；保持与输入实例一一对应的外层顺序
        valid_instances = []
        for instance in instance_polys:
            # 第一道过滤：点数太少的直接丢弃
            # 这里 polygon 已经是 (N, 2) 的点列，len() 就是点数而不是坐标个数；
            # 少于 4 个点既无法构成有意义的闭合轮廓（shapely 也无法算面积），
            # 更无法在后续被重采样成 128 点的监督目标
            polygons = [polygon for polygon in instance if len(polygon) >= 4]

            # 第二道处理：把坐标硬夹到 [0, W-1] × [0, H-1]
            # transform_polys 已经做过几何裁剪，这里再夹一次是防御性兜底：
            # 浮点误差、插值交点可能留下 -1e-6 或 W 这类边界外的值，
            # 而后面 prepare_box_target 会把中心四舍五入成整数并当作
            # 特征图索引（ct_ind = y * W + x）使用，越界索引会直接取错位置
            for polygon in polygons:
                # 原地修改：直接改 transform_polys 刚返回的新数组，不额外拷贝；
                # 但这意味着调用方拿到的 transformed_instances 也会被同步改动
                polygon[:, 0] = np.clip(polygon[:, 0], 0, output_w - 1)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, output_h - 1)

            # 第三道过滤：按 shapely 多边形面积剔除小碎片（面积 <= 5 就丢）
            # 面积单位是"输出网格格子"的平方，1 格² 约等于输入图 4×4 像素；
            # 随机 crop 后椎体常被切出零碎小片，它们既不是有效轮廓，
            # 又会给 Flow 匹配引入噪声监督
            polygons = snake_voc_utils.filter_tiny_polys(polygons)

            # 第四道处理：统一成顺时针方向
            # 内部判断 shapely 的 exterior.is_ccw，逆时针的多边形会被 [::-1] 反转。
            # 原因：演化监督是逐点对应的 Δ* = C_GT - C_cur，如果 GT 与初始化轮廓
            # 绕行方向相反，同一个索引会指向轮廓的"对面"，监督信号完全错误；
            # 另外 polygon 点的遍历顺序也会影响重采样后起点的语义
            polygons = snake_voc_utils.get_cw_polys(polygons)

            # 第五道处理：去掉重复点，同时保持原有的绕行顺序
            # np.unique(polygon, axis=0, return_index=True) 做两件事：
            #   1. 返回去重后的行（但它按字典序排序，顺序被打乱了）
            #   2. return_index 给出每个唯一行在"原数组"中首次出现的下标
            # 因此用 np.sort(return_index) 把这些下标重新升序排列，
            # 就能按原始遍历顺序取出"每个点第一次出现"的版本。
            # 这样既删掉了闭合多边形首尾重复的收尾点，也删掉了裁剪/取整产生的
            # 连续重合点——它们会让弧长重采样出现零长度边，进而除零
            polygons = [
                polygon[
                    np.sort(np.unique(polygon, axis=0, return_index=True)[1])
                ]
                for polygon in polygons
            ]

            # 该实例清洗完毕（可能是空列表，表示这片切片上它的轮廓已全部失效）
            valid_instances.append(polygons)

        # 返回结构与输入同构：list[instance] -> list[polygon] -> (N, 2) 浮点数组
        return valid_instances

    @staticmethod
    def get_extreme_points(instance_polys):
        """逐多边形提取上/左/下/右四个"稳定极值点"。

        本方法只是个双层广播壳：外层遍历实例、内层遍历多边形，
        真正的算法在 snake_voc_utils.get_extreme_points 里。

        底层算法做的事（以"最上点"为例）：
        1. 先用 np.argmin 找到 y 坐标最小的顶点作为种子；
        2. 但"最上"往往不止一个点——比如轮廓顶部有一条水平边，
           或重采样后若干点几乎同高。若直接取 argmin 那一个点，
           结果会随微小扰动（噪声、重采样误差）跳变，不稳定；
        3. 因此 plateau() 会从种子出发沿闭合轮廓双向扩展，
           把所有与极值差在容差内（宽/高的 2%）的"平台"点都收进来；
        4. 最终返回的极值点取平台中点：y 用极值本身，
           x 取平台内所有点 x 的 (max+min)/2——确定且居中。

        返回结构与输入同构：
            list[instance] -> list[(4, 2) 数组]
        每个 (4, 2) 数组行序固定为 [top, left, bottom, right]，
        与 get_octagon 的下标约定严格对应：
            extreme[0]=top, extreme[1]=left, extreme[2]=bottom, extreme[3]=right

        下游用途：
        - prepare_init / prepare_evolution 用它的 min/max 反推包围盒
          （x_min=extreme[:,0].min() 等），即这 4 个点同时编码了 box；
        - get_evolution_init 用它们直接搭出初始八边形，是演化起点。
        """
        return [
            # 内层：单个实例的每个多边形各得一个 (4, 2) 极值点数组
            [snake_voc_utils.get_extreme_points(polygon) for polygon in instance]
            # 外层：切片上的每个实例各得一个列表
            for instance in instance_polys
        ]

    @staticmethod
    def prepare_box_target(box, cls_id, output_width, wh, ct_cls, ct_ind):
        """写入 Route-B 主线消费的 GT 框中心、宽高和类别。

        这是典型的 CenterNet 风格检测 GT：不存框的对角点，而是把每个框
        拆成三条信息，分别追加到调用方传入的三个累加列表里：
            ct_cls[i] : 第 i 个 GT 中心的类别 id
            wh[i]     : 第 i 个 GT 框的 (宽, 高)
            ct_ind[i] : 第 i 个 GT 中心在输出网格上展平后的线性索引

        调用约定（见 snake.py 的循环）：
        - 本函数是"追加"语义，每处理一个实例调用一次，无返回值；
        - 三个列表由调用方逐样本新建，样本级数量由 len(ct_ind) 记录
          （snake.py 里的 'ct_num'），collate_batch 再按 max_count
          填充成 (B, max) 的 batch 张量；
        - 因此三个列表的下标天然按实例顺序对齐，训练侧可以按同一
          位置同时取出"哪个中心、什么类别、框多大"。

        下游消费（Route-B 主线）：
        - networks/mainline.py 用 ct_ind 当特征图索引，把 GT 中心/宽高/
          类别 scatter 到检测输出上算监督（center heatmap + wh 回归 + cls）；
        - 推理时检测头预测出中心索引和宽高，再经 snake_decode 把 xyxy
          框转成八边形 —— 即 Route-B 的"检测框 → 八边形 → 演化"起点。
        """
        # 类别独立存一份：中心索引定位"在哪"，类别回答"是什么"
        ct_cls.append(cls_id)

        # 拆出框的四个角（坐标系已是 128×128 输出网格，见上游 transform）
        x_min, y_min, x_max, y_max = box

        # 框中心 = 对角点平均。此时还是浮点（如 62.3），
        # 但检测 GT 要落在离散网格上，所以下一步取整
        center = np.asarray(
            [(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32
        )
        # 四舍五入到整格格点再转 int32。
        # 注意 np.round 是"银行家舍入"（.5 取偶数），与常规 round-half-up
        # 有细微差别；对训练影响可忽略，但复现对比时要知道这一点。
        # 取整后 center[0]=x 列号、center[1]=y 行号，可直接当索引用
        center = np.round(center).astype(np.int32)

        # 框尺寸：height 取 y 跨度、width 取 x 跨度（注意变量名与来源轴的对应）
        height, width = y_max - y_min, x_max - x_min
        # 按 (宽, 高) 顺序存入，与 wh 命名一致；下游回归头同序对齐
        wh.append([width, height])

        # 二维格点 (y, x) 展平成一维线性索引：idx = y * W + x（行优先）。
        # 等价于把 H×W 特征图拉平后按位置寻址，mainline.py 拿它直接
        # scatter 到 (B, H*W) 展平的检测输出上。
        # int(output_width) 是防御性转换：上游传入的可能是 numpy 标量或浮点
        ct_ind.append(center[1] * int(output_width) + center[0])

    @staticmethod
    def prepare_init(
        box,
        extreme_point,
        image_init_extremes,
        canonical_init_extremes,
        image_gt_extremes,
        canonical_gt_extremes,
        height,
        width,
    ):
        """四极值点初始化目标（图像坐标 + canonical 坐标）。

        为单个实例构造"初始化阶段"的两组输入/GT，与 prepare_box_target
        相同的追加式约定：每实例调用一次，无返回值，写入四个累加列表。

        产出内容：
        1. image_init —— 40 点初始轮廓：
           get_init(box) = get_octagon(get_quadrangle(box))，几何链是
           检测框 → 菱形四边形（四条边中点连线）→ 内切八边形；
           再经 uniformsample 按弧长重采样成固定 40 点
           （snake_config.init_poly_num=40）。
           这是检测框到初始轮廓的确定式映射，推理时无需学习即可得到起点。
        2. image_gt —— 4 个稳定极值点（上游 get_extreme_points 的输出）：
           训练"从框预测极值点"的头；推理时用预测出的极值点搭建
           比裸框更贴合的初始轮廓（原版 Deep Snake 的两段式设计）。
        3. canonical_* —— 把上述两个轮廓各自减去自身 min(x,y)，即把原点
           平移到自身包围盒左上角。注意本仓库的 img_poly_to_can_poly
           会忽略传入的 box 参数（内部 del 掉），只用多边形自身的 min。
           平移不变后网络学的是"形状"而非"绝对位置"，同一形状
           在不同位置共享同一回归目标。

        读代码时的两个陷阱：
        - 调用处列表名为 i_it_4pys/c_it_4pys/i_gt_4pys/c_gt_4pys，
          "4py" 是原版 Deep Snake 的遗留命名（本指 4 个极值点），
          而这里的 image_init 实际是 40 点八边形，勿被名字误导；
        - snake.py 会把刚 append 进去的 image_init 传给 prepare_evolution，
          但后者立即 del 掉 —— 演化主线用的是它自己从极值点重建的
          128 点初始轮廓，与本函数的 40 点初始化是两条独立分支。
        """
        # 输出网格尺寸（调用处传的 output_h/output_w）未参与计算，
        # 占位删除，仅为保持调用签名兼容
        del height, width
        # 用极值点的 min/max 反推实例包围盒：
        # 4 个极值点本身就把 box 编码在自己的坐标范围里
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])
        # 检测框 → 菱形 → 八边形 → 按弧长均匀重采样成 40 点初始轮廓
        image_init = snake_voc_utils.uniformsample(
            snake_voc_utils.get_init(box), snake_config.init_poly_num
        )
        # 初始轮廓平移到自身包围盒左上角，得到 canonical 坐标版本
        canonical_init = snake_voc_utils.img_poly_to_can_poly(
            image_init, x_min, y_min, x_max, y_max
        )
        # 初始化阶段的 GT 就是这 4 个极值点（直接引用，不拷贝）
        image_gt = extreme_point
        # 极值点同样做 canonical 平移，与初始轮廓保持同一坐标口径
        canonical_gt = snake_voc_utils.img_poly_to_can_poly(
            image_gt, x_min, y_min, x_max, y_max
        )
        # 四个累加列表下标按实例顺序对齐，与 prepare_box_target 同约定；
        # collate_batch 按 max_count 填充成 (B, max) 的 batch 张量
        image_init_extremes.append(image_init)
        canonical_init_extremes.append(canonical_init)
        image_gt_extremes.append(image_gt)
        canonical_gt_extremes.append(canonical_gt)

    def prepare_evolution(
        self,
        polygon,
        extreme_point,
        image_init_extremes,
        image_init_polys,
        canonical_init_polys,
        image_gt_polys,
        canonical_gt_polys,
    ):
        """构造 128 点初始轮廓和对齐后的 GT：统一点数、循环起点后再给 Flow 当 Δ*。

        这是演化主线（Flow Matching）的目标构造，与 prepare_init 的 40 点
        初始化分支相互独立。核心难点是让"初始轮廓"和"GT 轮廓"满足两个前提：

        前提一：点数相同（都是 128，snake_config.poly_num）。
        前提二：第 i 个点在两个轮廓上指的是轮廓的同一处。
            否则 Δ* = C_GT - C_cur 这个逐点差分会把"上终板"的位移
            错配给"前缘"的点，监督信号完全错误。

        流程：
        1. image_init：极值点 → 八边形 → 弧长重采样成 128 点。
        2. image_gt：原始多边形 → 重采样成 len(polygon)*128 点。
           为什么先放大到 N 倍再多抽？因为原始多边形点数任意（可能 37 个点），
           直接重采样到 128 点只能保证点数对，无法保证"点 i 落在同一相位上"；
           先加密到 128 的整数倍，就能用 [::N] 等间隔抽取，均匀且不丢形状。
        3. 对齐循环起点：轮廓是闭合环，重采样后的"第 0 个点"只是个任意起点。
           必须把 GT 环旋转到与 image_init[0] 最近的顶点作为起点，
           否则虽然点数对了，但整体绕行相位差了一半圈。
        """
        # 上一个方法产出的 40 点初始化在这里用不到，占位删除；
        # 说明演化主线不用 40 点轮廓，而是用自己的八边形重建 128 点起点
        del image_init_extremes
        # 用极值点的 min/max 反推包围盒（与 prepare_init 同一手法）
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])
        box = [x_min, y_min, x_max, y_max]
        # 演化轮廓的固定点数：128（snake_config.poly_num）
        point_count = snake_config.poly_num
        # 极值点 → 八边形作为初始形状的骨架（box 参数内部被忽略）
        base_init = snake_voc_utils.get_evolution_init(extreme_point, box)
        # 弧长均匀重采样到 128 点：保证点沿周长等间隔分布，
        # 网络学到的位移才是"法向/切向"这种几何上均匀的量
        image_init = snake_voc_utils.uniformsample(base_init, point_count)
        # 初始轮廓的 canonical 版本：平移不变，只保留形状
        canonical_init = snake_voc_utils.img_poly_to_can_poly(
            image_init, x_min, y_min, x_max, y_max
        )
        # GT 先加密到 (原始点数 × 128) 点，为下一步等间隔抽取做准备
        image_gt = snake_voc_utils.uniformsample(
            polygon, len(polygon) * point_count
        )
        # 找对齐起点：在加密后的 GT 环上，找出与 image_init[0] 欧氏距离
        # 平方和最小的那个顶点下标（np.argmin 返回单个整数下标）
        start_index = np.argmin(
            np.power(image_gt - image_init[0], 2).sum(axis=1)
        )
        # 两步合一：
        # np.roll(image_gt, -start_index, axis=0) —— 把环旋转，使原来的
        #   start_index 挪到位置 0，即"离初始轮廓首点最近的那个 GT 点"成为新起点；
        # [:: len(polygon)] —— 在 N*128 个点上每隔 N 个取一个，
        #   恰好得到 128 个点，且保持加密后的均匀间隔与绕行方向。
        image_gt = np.roll(image_gt, -start_index, axis=0)[:: len(polygon)]
        # GT 的 canonical 版本，与 canonical_init 同一坐标口径
        canonical_gt = snake_voc_utils.img_poly_to_can_poly(
            image_gt, x_min, y_min, x_max, y_max
        )
        # 四个累加列表按实例顺序追加，供 collate_batch 填充成 (B, N, 128, 2)
        image_init_polys.append(image_init)
        canonical_init_polys.append(canonical_init)
        image_gt_polys.append(image_gt)
        canonical_gt_polys.append(canonical_gt)
