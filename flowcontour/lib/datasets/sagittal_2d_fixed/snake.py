# -*- coding: utf-8 -*-
"""FlowContour 矢状位 2D 切片数据集与 MoonViT 特征加载模块 (Sagittal 2D Snake Dataset)

本模块定义了用于 FlowContour 轮廓演化训练与评估的 Dataset 类：
1. 矢状位 2D 切片管理：负责读取 VerSe 数据集的切片清单 (manifest)，管理训练/验证集划分，
   严格隔离禁入病例 (forbidden cases) 与独立验证病例 (dev cases)。
2. MoonViT 离线 特征对齐：
   - 读取预先抽取的 MoonViT layer_18 视觉 Token (.npz 格式)
   - 验证特征维度、输入尺寸 (448)、Patch 大小 (14)、均值标准差归一化等严格契约
   - 仅加载中心切片特征 (center-slice)，避免跨切片误差扩散
3. Route-B 几何轮廓初始化与抖动增强 (Box Jitter)：
   - 从 GT 标注框生成八边形 (Octagon) 初始轮廓，并重采样为 128 点闭合多边形
   - 训练阶段支持中心偏移、对数尺度缩放、四边独立扰动等多维度几何抖动数据增强
4. 轮廓目标构造：
   - 继承 ContourTargetMixin 构建位移残差监督目标与强化学习基准
"""

import csv
import os
import math

import cv2
import numpy as np
import torch.utils.data as data

from lib.config import cfg
from lib.datasets.contour_targets import ContourTargetMixin
from lib.utils.snake import snake_config, snake_voc_utils


_MANIFEST_SPLITS = {
    'train': 'training',
    'val': 'validation',
    'dev': 'training',
    'mini': 'training',
    'test': 'test',
}
_CACHE_SPLITS = dict(_MANIFEST_SPLITS)
_REQUIRED_COLUMNS = {'split', 'case_id', 'slice_idx', 'image_path', 'mask_path'}
_MOONVIT_FEATURE_KEY = 'layer_18'
_MOONVIT_LAYER_CHANNELS = 1152
_MOONVIT_INPUT_SIZE = 448
_MOONVIT_PATCH_SIZE = 14
_MOONVIT_NORMALIZATION = 'moonvit_pretrained_rgb_mean_std'


def _resample_boundary(poly, count=512):
    """Uniform arc-length sampling, matching the tuned RAOS target builder."""
    lengths = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    total = float(lengths.sum())
    if total < 1e-6:
        raise ValueError('Degenerate raw-mask boundary')
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    positions = np.arange(count) * total / count
    indices = np.clip(np.searchsorted(arc, positions, side='right') - 1, 0, len(poly) - 1)
    fraction = (positions - arc[indices]) / np.maximum(lengths[indices], 1e-8)
    return (poly[indices] + fraction[:, None] *
            (poly[(indices + 1) % len(poly)] - poly[indices])).astype(np.float32)


def _boundary_curvature(poly):
    from scipy.ndimage import gaussian_filter1d
    spacing = max(float(np.linalg.norm(np.roll(poly, -1, axis=0) - poly,
                                       axis=1).mean()), 1e-3)
    smooth = gaussian_filter1d(poly, sigma=max(2.0 / spacing, 0.5), axis=0, mode='wrap')
    span = min(max(round(5.0 / spacing), 1), max(len(poly) // 8, 1))
    before = smooth - np.roll(smooth, span, axis=0)
    after = np.roll(smooth, -span, axis=0) - smooth
    cosine = (before * after).sum(-1) / np.maximum(
        np.linalg.norm(before, axis=-1) * np.linalg.norm(after, axis=-1), 1e-6)
    return np.arccos(np.clip(cosine, -1, 1)).astype(np.float32)


def _dense_mask_targets(mask, classes, anchors, transform_output, flipped):
    """Match raw-mask components by GT-anchor votes before extracting 512 points.

    The dense curve comes from the mask's half-pixel isocontour, never from the
    simplified 128-point polygon. Curvature weights use raw-image pixel units,
    exactly as in resized_data_10.18/tools/geometry.py.
    """
    from skimage import measure
    components = {}
    curves, weights = [], []
    for cls_id, anchor in zip(classes, anchors):
        cls_id = int(cls_id)
        if cls_id not in components:
            count, labels, stats, centers = cv2.connectedComponentsWithStats(
                (mask == cls_id).astype(np.uint8), connectivity=8)
            if count <= 1:
                raise ValueError('GT contour class {} is absent from raw mask'.format(cls_id))
            components[cls_id] = (labels, centers)
        labels, centers = components[cls_id]
        xy = np.rint(anchor).astype(np.int32)
        xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)
        votes = labels[xy[:, 1], xy[:, 0]]
        votes = votes[votes > 0]
        if len(votes):
            component_id = int(np.bincount(votes).argmax())
        else:
            component_id = 1 + int(np.linalg.norm(
                centers[1:] - anchor.mean(0), axis=1).argmin())
        component = (labels == component_id).astype(np.uint8)
        candidates = measure.find_contours(np.pad(component, 1), 0.5)
        raw = max(candidates, key=len)[:, ::-1].astype(np.float32) - 1
        dense = _resample_boundary(raw)
        curve_weight = 1 + 2 * np.minimum(_boundary_curvature(dense) / math.radians(60), 1)
        curve_weight /= curve_weight.mean()
        if flipped:
            dense[:, 0] = mask.shape[1] - dense[:, 0] - 1
        dense = dense @ transform_output[:, :2].T + transform_output[:, 2]
        curves.append(dense.astype(np.float32))
        weights.append(curve_weight.astype(np.float32))
    return curves, weights


class Dataset(ContourTargetMixin, data.Dataset):
    """矢状位 2D 切片数据集核心类 (纯 2D 轮廓演化 + 中心切片 MoonViT 离线特征)。

    继承自 ContourTargetMixin 与 PyTorch Dataset：
    - 读取切片清单 CSV，构建严格的数据划分与案例隔离
    - 加载灰度图像、分割 Mask 以及 MoonViT layer_18 缓存特征
    - 空间仿射数据增强 (平移、缩放、旋转、翻转)
    - 构造 128 点闭合轮廓与 Route-B 八边形初始轮廓
    """

    @staticmethod
    def _config_value(names, default=None):
        """从候选配置键名列表中依序查找第一个存在的非空配置项值。

        Args:
            names: 候选配置属性名称列表或元组。
            default: 若所有属性皆不存在或为空时的默认返回值。

        Returns:
            获取到的配置属性值。
        """
        for name in names:
            value = getattr(cfg, name, None)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return default

    @staticmethod
    def _canonical_int(value, description):
        """规范化并严格校验单标量正整数。

        校验输入必须为单一数值，且能精确表示为大于 0 的正整数，防御浮点精度截断或非数异常。

        Args:
            value: 待校验的输入值。
            description: 字段描述信息（用于抛出明确的异常提示）。

        Returns:
            int: 规范化后的正整数。
        """
        if value is None:
            return None
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(
                '{} must contain one integer, got shape {}'.format(
                    description, array.shape
                )
            )
        scalar = array.reshape(-1)[0]
        try:
            parsed = int(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('{} must be an integer, got {!r}'.format(
                description, scalar
            )) from exc
        try:
            numeric = float(scalar)
        except (TypeError, ValueError, OverflowError):
            numeric = float(parsed)
        if not np.isfinite(numeric) or numeric != float(parsed) or parsed <= 0:
            raise ValueError('{} must be a positive integer, got {!r}'.format(
                description, scalar
            ))
        return parsed

    @classmethod
    def _canonical_size(cls, value, description):
        """规范化正方形空间分辨率尺寸。

        若输入为单整数则直接返回；若为长度为 2 的序列，则要求宽高必须严格相等。

        Args:
            value: 空间分辨率尺寸（整数或 [H, W] 序列）。
            description: 字段描述信息。

        Returns:
            int: 规范化后的单边分辨率大小。
        """
        if value is None:
            return None
        array = np.asarray(value)
        if array.size == 2:
            values = [
                cls._canonical_int(item, description) for item in array.reshape(-1)
            ]
            if values[0] != values[1]:
                raise ValueError(
                    '{} must be square, got {}'.format(description, values)
                )
            return values[0]
        return cls._canonical_int(value, description)

    @staticmethod
    def _canonical_text(value, description, lower=False):
        """规范化并严格校验字符串文本。

        支持 bytes 解码为 UTF-8，可选择转换为小写，禁止空字符串。

        Args:
            value: 待校验的文本值。
            description: 字段描述。
            lower: 是否转换为小写。

        Returns:
            str: 规范化后的非空字符串。
        """
        if value is None:
            return None
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(
                '{} must contain one string, got shape {}'.format(
                    description, array.shape
                )
            )
        scalar = array.reshape(-1)[0]
        if isinstance(scalar, bytes):
            try:
                scalar = scalar.decode('utf-8')
            except UnicodeDecodeError as exc:
                raise ValueError('{} is not valid UTF-8'.format(description)) from exc
        text = str(scalar).strip()
        if not text:
            raise ValueError('{} must not be empty'.format(description))
        return text.lower() if lower else text

    @classmethod
    def _canonical_normalization(cls, value, description):
        """规范化图像特征归一化策略名称，必须严格匹配预设标识符。"""
        text = cls._canonical_text(value, description, lower=True)
        aliases = {
            'moonvit_rgb_mean_0.5_std_0.5': _MOONVIT_NORMALIZATION,
        }
        return aliases.get(text, text)

    @classmethod
    def _canonical_checkpoint(cls, value, description):
        """规范化 MoonViT 检查点文件名标识符。"""
        text = cls._canonical_text(value, description)
        if text is None:
            return None
        return os.path.realpath(os.path.abspath(os.path.expanduser(text)))

    @staticmethod
    def _canonical_case_ids(value, description):
        """规范化病例 ID 集合（解析列表或元组，去除空格与重复项）。"""
        if value is None:
            return ()
        if isinstance(value, str):
            raw_values = value.split(',')
        else:
            try:
                raw_values = list(value)
            except TypeError as exc:
                raise ValueError('{} must be a string or sequence'.format(
                    description
                )) from exc
        parsed = []
        for raw in raw_values:
            case_id = str(raw).strip()
            if not case_id:
                continue
            if (
                case_id in ('.', '..')
                or os.path.basename(case_id) != case_id
                or '/' in case_id
                or '\\' in case_id
            ):
                raise ValueError('{} contains unsafe case_id={!r}'.format(
                    description, case_id
                ))
            parsed.append(case_id)
        if len(parsed) != len(set(parsed)):
            raise ValueError('{} contains duplicate case IDs'.format(description))
        return tuple(parsed)

    def _apply_case_partition(self, rows):
        """执行严格的数据集案例划分与安全隔离。

        规则约束：
        1. 排除 dev8 案例不进入训练集 (train_exclude_case_ids)
        2. 仅允许 dev8 案例进入评估集 (eval_include_case_ids)
        3. 绝对禁止 forbidden 案例进入任何流程 (forbidden_case_ids)
        4. 校验划分后的案例总数与切片行数是否完全符合预设契约 (fail-closed)

        Args:
            rows: 从清单中读取的原始记录行列表。

        Returns:
            list[dict]: 过滤且校验通过后的目标样本记录列表。
        """
        available = {row['case_id'] for row in rows}
        train_exclude = self._canonical_case_ids(
            getattr(cfg, 'sagittal_train_exclude_case_ids', ()),
            'cfg.sagittal_train_exclude_case_ids',
        )
        eval_include = self._canonical_case_ids(
            getattr(cfg, 'sagittal_eval_include_case_ids', ()),
            'cfg.sagittal_eval_include_case_ids',
        )
        forbidden = set(self._canonical_case_ids(
            getattr(cfg, 'sagittal_forbidden_case_ids', ()),
            'cfg.sagittal_forbidden_case_ids',
        ))

        if self.split == 'train':
            missing = set(train_exclude).difference(available)
            if missing:
                raise ValueError('Training exclusion cases are absent from the '
                                 'manifest split: {}'.format(sorted(missing)))
            selected = [
                row for row in rows if row['case_id'] not in set(train_exclude)
            ]
            expected_cases = int(getattr(
                cfg, 'sagittal_expected_train_case_count', 0
            ))
            expected_rows = int(getattr(
                cfg, 'sagittal_expected_train_row_count', 0
            ))
            policy = 'training-minus-exact-dev-cases'
        elif self.split == 'dev':
            if not eval_include:
                raise ValueError(
                    'The dev split requires cfg.sagittal_eval_include_case_ids'
                )
            missing = set(eval_include).difference(available)
            if missing:
                raise ValueError('Development cases are absent from the '
                                 'training manifest split: {}'.format(sorted(missing)))
            selected = [
                row for row in rows if row['case_id'] in set(eval_include)
            ]
            expected_cases = int(getattr(
                cfg, 'sagittal_expected_eval_case_count', 0
            ))
            expected_rows = int(getattr(
                cfg, 'sagittal_expected_eval_row_count', 0
            ))
            policy = 'exact-development-case-allowlist'
        else:
            selected = rows
            expected_cases = 0
            expected_rows = 0
            policy = 'manifest-split-unfiltered'

        selected_cases = sorted({row['case_id'] for row in selected})
        forbidden_selected = forbidden.intersection(selected_cases)
        if forbidden_selected:
            raise ValueError('Forbidden locked cases selected: {}'.format(
                sorted(forbidden_selected)
            ))
        if expected_cases and len(selected_cases) != expected_cases:
            raise ValueError('Expected {} cases for {}, got {}'.format(
                expected_cases, policy, len(selected_cases)
            ))
        if expected_rows and len(selected) != expected_rows:
            raise ValueError('Expected {} rows for {}, got {}'.format(
                expected_rows, policy, len(selected)
            ))
        self.case_partition_audit = {
            'policy': policy,
            'manifest_split': self.manifest_split,
            'selected_case_ids': selected_cases,
            'selected_case_count': len(selected_cases),
            'selected_row_count': len(selected),
            'train_exclude_case_ids': list(train_exclude),
            'eval_include_case_ids': list(eval_include),
            'forbidden_case_ids': sorted(forbidden),
            'forbidden_selected': [],
        }
        return selected

    def __init__(self, ann_file, data_root, split):
        """初始化矢状位 2D 轮廓演化数据集。

        执行步骤：
        1. 初始化基类与配置参数读取
        2. 解析图像尺寸、下采样率、轮廓点数 (128点) 等核心超参
        3. 加载并严格核验离线 MoonViT layer_18 缓存配置
        4. 配置 Route-B 边界框几何抖动数据增强参数
        5. 读取清单文件，执行案例划分筛选与前景切片有效性核查

        Args:
            ann_file: 切片标注清单文件路径。
            data_root: 数据集根目录路径。
            split: 数据集划分标识 ('train', 'dev', 'val', 'test')。
        """
        data.Dataset.__init__(self)
        if split not in _MANIFEST_SPLITS:
            raise ValueError(
                'Unsupported catalog split {!r}; expected one of {}'.format(
                    split, sorted(_MANIFEST_SPLITS)
                )
            )

        self.ann_file = os.path.abspath(os.fspath(ann_file))
        self.data_root = os.path.abspath(os.fspath(data_root))
        self.split = split
        self.manifest_split = _MANIFEST_SPLITS[split]
        self.locate_feat_enabled = not bool(cfg.raos_image_encoder) and bool(
            getattr(cfg, 'locate_feat_inject', False)
            or getattr(cfg, 'locate_feat_replace', False)
        )
        if self.locate_feat_enabled:
            cache_root = self._config_value(
                ('sagittal_moonvit_cache_root', 'locate_feat_cache_root'),
                None,
            )
            if cache_root is None:
                raise ValueError(
                    'MoonViT sagittal cache is enabled, but cfg.sagittal_moonvit_cache_root '
                    'and cfg.locate_feat_cache_root are both empty'
                )
            self.locate_feat_cache_root = os.path.abspath(os.fspath(cache_root))
            self.sagittal_moonvit_cache_root = self.locate_feat_cache_root
            self.locate_feat_split_map = dict(_CACHE_SPLITS)
            self.locate_feat_split = self.locate_feat_split_map[split]
            self.locate_feat_cache_split = self.locate_feat_split
            self.sagittal_moonvit_cache_split = self.locate_feat_split
            self.sagittal_moonvit_cache_split_map = dict(self.locate_feat_split_map)

            feature_key = str(self._config_value(
                (
                    'sagittal_moonvit_feature_key',
                    'sagittal_moonvit_feat_key',
                    'locate_feat_feature_key',
                    'locate_feat_key',
                ),
                _MOONVIT_FEATURE_KEY,
            )).strip()
            parsed_keys = [k.strip() for k in feature_key.split(',') if k.strip()]
            if parsed_keys != [_MOONVIT_FEATURE_KEY]:
                raise ValueError(
                    'The mainline requires only MoonViT {!r}; got {!r}'.format(
                        _MOONVIT_FEATURE_KEY, feature_key
                    )
                )
            self.locate_feat_key = parsed_keys[0]
            self.locate_feat_keys = parsed_keys
            self.sagittal_moonvit_feature_key = parsed_keys[0]

            fusion_mode = str(self._config_value(
                (
                    'sagittal_moonvit_fusion_mode',
                    'sagittal_moonvit_fusion',
                    'locate_feat_fusion_mode',
                ),
                'center_only',
            )).strip().lower()
            if fusion_mode != 'center_only':
                raise ValueError(
                    "The mainline requires locate_feat_fusion_mode='center_only'; "
                    'got {!r}'.format(fusion_mode)
                )
            self.locate_feat_fusion_mode = fusion_mode
            self.sagittal_moonvit_fusion_mode = fusion_mode

            expected_input_size = self._config_value(
                (
                    'sagittal_moonvit_expected_input_size',
                    'sagittal_moonvit_input_size',
                    'locate_feat_expected_input_size',
                    'locate_feat_input_size',
                ),
                _MOONVIT_INPUT_SIZE,
            )
            expected_normalization = self._config_value(
                (
                    'sagittal_moonvit_expected_normalization',
                    'sagittal_moonvit_normalization',
                    'sagittal_moonvit_norm',
                    'locate_feat_expected_normalization',
                    'locate_feat_normalization',
                ),
                _MOONVIT_NORMALIZATION,
            )
            expected_patch_size = self._config_value(
                (
                    'sagittal_moonvit_expected_patch_size',
                    'sagittal_moonvit_patch_size',
                    'locate_feat_expected_patch_size',
                    'locate_feat_patch_size',
                ),
                _MOONVIT_PATCH_SIZE,
            )
            expected_checkpoint = self._config_value(
                (
                    'sagittal_moonvit_expected_checkpoint',
                    'sagittal_moonvit_checkpoint',
                    'sagittal_moonvit_checkpoint_path',
                    'locate_feat_expected_checkpoint',
                    'locate_feat_checkpoint',
                ),
                None,
            )
            self.locate_feat_expected_input_size = self._canonical_size(
                expected_input_size, 'expected input_size'
            )
            self.locate_feat_expected_normalization = self._canonical_normalization(
                expected_normalization, 'expected normalization'
            )
            self.locate_feat_expected_patch_size = self._canonical_int(
                expected_patch_size, 'expected patch_size'
            )
            self.locate_feat_expected_checkpoint = self._canonical_checkpoint(
                expected_checkpoint, 'expected checkpoint'
            )
            self.sagittal_moonvit_expected_input_size = self.locate_feat_expected_input_size
            self.sagittal_moonvit_expected_normalization = self.locate_feat_expected_normalization
            self.sagittal_moonvit_expected_patch_size = self.locate_feat_expected_patch_size
            self.sagittal_moonvit_expected_checkpoint = self.locate_feat_expected_checkpoint
            self.sagittal_moonvit_input_size = self.locate_feat_expected_input_size
            self.sagittal_moonvit_normalization = self.locate_feat_expected_normalization
            self.sagittal_moonvit_patch_size = self.locate_feat_expected_patch_size
            self.sagittal_moonvit_checkpoint = self.locate_feat_expected_checkpoint
            self.locate_feat_input_size = self.locate_feat_expected_input_size
            self.locate_feat_normalization = self.locate_feat_expected_normalization
            self.locate_feat_patch_size = self.locate_feat_expected_patch_size
            self.locate_feat_checkpoint = self.locate_feat_expected_checkpoint

        mean = float(getattr(cfg, 'image_mean', 0.5))
        std = float(getattr(cfg, 'image_std', 0.5))
        if std <= 0:
            raise ValueError('cfg.image_std must be positive')
        self.mean = snake_config.mean.copy()
        self.std = snake_config.std.copy()
        self.color_aug = bool(getattr(cfg, 'image_color_aug', False))
        self.lr_flip = bool(getattr(cfg, 'image_lr_flip', False))
        self.random_crop = bool(getattr(cfg, 'image_random_crop', True))

        if int(cfg.anatomical_class_count) != 19:
            raise ValueError(
                'RAOS requires anatomical_class_count=19, got {}'.format(
                    cfg.anatomical_class_count
                )
            )

        rows = self._apply_case_partition(self._read_manifest())
        self._case_rows = self._group_case_rows(rows)
        self.records = [row for case_rows in self._case_rows.values() for row in case_rows]
        if split == 'mini' and self.records:
            first_case = next(iter(self._case_rows))
            self.records = self._case_rows[first_case][:100]
        if not self.records:
            raise ValueError(
                'No rows for manifest split {!r} in {}'.format(
                    self.manifest_split, self.ann_file
                )
            )

        self.bbox_2d_manifest = self._configured_bbox_manifest()
        foreground_keys = self._read_foreground_keys(self.bbox_2d_manifest)
        self.foreground_flags = [
            (self.manifest_split, row['case_id'], row['slice_idx']) in foreground_keys
            for row in self.records
        ]

    def _resolve_path(self, path):
        path = os.fspath(path)
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self.data_root, path))

    @staticmethod
    def _configured_bbox_manifest():
        train_cfg = getattr(cfg, 'train', None)
        for config_node in (train_cfg, cfg):
            if config_node is None:
                continue
            for name in ('bbox_2d_manifest', 'bbox_manifest'):
                value = getattr(config_node, name, None)
                if value is None:
                    continue
                path = os.fspath(value).strip()
                if path:
                    return os.path.abspath(os.path.expanduser(path))
        return None

    @staticmethod
    def _read_foreground_keys(manifest_path):
        if manifest_path is None:
            return set()
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                '2D bbox manifest not found: {}'.format(manifest_path)
            )

        required_columns = {'split', 'case_id', 'slice_idx', 'mask_pixel_count'}
        foreground_keys = set()
        with open(manifest_path, 'r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            missing = required_columns.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    '2D bbox manifest is missing columns: {}'.format(sorted(missing))
                )
            for row_number, raw in enumerate(reader, start=2):
                split = raw['split']
                case_id = raw['case_id']
                if not split or not case_id:
                    raise ValueError(
                        'Empty split or case_id at 2D bbox manifest row {}'.format(
                            row_number
                        )
                    )
                try:
                    slice_idx = int(raw['slice_idx'])
                    count_text = str(raw['mask_pixel_count'] or '').strip()
                    mask_pixel_count = int(count_text) if count_text else 0
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        'Invalid slice_idx or mask_pixel_count at 2D bbox manifest '
                        'row {}: slice_idx={!r}, mask_pixel_count={!r}'.format(
                            row_number, raw['slice_idx'], raw['mask_pixel_count']
                        )
                    ) from exc
                if mask_pixel_count > 0:
                    foreground_keys.add((split, case_id, slice_idx))
        return foreground_keys

    def _read_manifest(self):
        if not os.path.isfile(self.ann_file):
            raise FileNotFoundError('Slice manifest not found: {}'.format(self.ann_file))

        rows = []
        with open(self.ann_file, 'r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            missing = _REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    'Slice manifest is missing columns: {}'.format(sorted(missing))
                )
            for row_number, raw in enumerate(reader, start=2):
                if raw['split'] != self.manifest_split:
                    continue
                case_id = raw['case_id']
                if not case_id:
                    raise ValueError('Empty case_id at manifest row {}'.format(row_number))
                try:
                    slice_idx = int(raw['slice_idx'])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        'Invalid slice_idx at manifest row {}: {!r}'.format(
                            row_number, raw['slice_idx']
                        )
                    ) from exc
                rows.append({
                    'case_id': case_id,
                    'slice_idx': slice_idx,
                    'image_path': self._resolve_path(raw['image_path']),
                    'mask_path': self._resolve_path(raw['mask_path']),
                    'row_number': row_number,
                })
        return rows

    @staticmethod
    def _group_case_rows(rows):
        grouped = {}
        for row in rows:
            grouped.setdefault(row['case_id'], []).append(row)

        for case_id, case_rows in grouped.items():
            case_rows.sort(key=lambda row: row['slice_idx'])
            indices = [row['slice_idx'] for row in case_rows]
            if len(indices) != len(set(indices)):
                raise ValueError(
                    'Duplicate slice_idx values for case {!r}'.format(case_id)
                )
        return grouped

    def _sagittal_moonvit_cache_path(self, row):
        """构建指定病例及切片的离线 MoonViT 特征缓存文件 (.npz) 完整路径。"""
        case_id = str(row['case_id'])
        if (
            not case_id
            or case_id in ('.', '..')
            or os.path.basename(case_id) != case_id
            or '/' in case_id
            or '\\' in case_id
        ):
            raise ValueError(
                'Unsafe case_id={!r}; MoonViT cache case IDs must be one path component'.format(
                    case_id
                )
            )
        slice_idx = int(row['slice_idx'])
        if slice_idx < 0:
            raise ValueError(
                'Invalid slice_idx={}; MoonViT cache filenames require non-negative indices'.format(
                    slice_idx
                )
            )
        return os.path.join(
            self.locate_feat_cache_root,
            self.locate_feat_split,
            case_id,
            'x{:04d}.npz'.format(slice_idx),
        )

    @staticmethod
    def _cache_required(npz, path, key):
        if key not in npz.files:
            raise ValueError(
                'MoonViT cache {} is missing required metadata {!r}; available keys={}'.format(
                    path, key, list(npz.files)
                )
            )
        return npz[key]

    @classmethod
    def _cache_integer(cls, value, path, key, minimum=0):
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(
                'MoonViT cache {} metadata {} must be scalar, got shape {}'.format(
                    path, key, array.shape
                )
            )
        scalar = array.reshape(-1)[0]
        try:
            parsed = int(scalar)
            numeric = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                'MoonViT cache {} metadata {} must be an integer, got {!r}'.format(
                    path, key, scalar
                )
            ) from exc
        if not np.isfinite(numeric) or numeric != float(parsed) or parsed < minimum:
            raise ValueError(
                'MoonViT cache {} metadata {} must be an integer >= {}, got {!r}'.format(
                    path, key, minimum, scalar
                )
            )
        return parsed

    @classmethod
    def _cache_integer_vector(cls, npz, path, key, size, minimum=0):
        value = np.asarray(cls._cache_required(npz, path, key))
        if value.size != size:
            raise ValueError(
                'MoonViT cache {} metadata {} must contain {} integers, got shape {}'.format(
                    path, key, size, value.shape
                )
            )
        parsed = [
            cls._cache_integer(item, path, key, minimum=minimum)
            for item in value.reshape(-1)
        ]
        return np.asarray(parsed, dtype=np.int32)

    @staticmethod
    def _cache_text(npz, path, key, lower=False):
        value = np.asarray(Dataset._cache_required(npz, path, key))
        if value.size != 1:
            raise ValueError(
                'MoonViT cache {} metadata {} must be one string, got shape {}'.format(
                    path, key, value.shape
                )
            )
        scalar = value.reshape(-1)[0]
        if isinstance(scalar, bytes):
            try:
                scalar = scalar.decode('utf-8')
            except UnicodeDecodeError as exc:
                raise ValueError(
                    'MoonViT cache {} metadata {} is not valid UTF-8'.format(path, key)
                ) from exc
        text = str(scalar).strip()
        if not text:
            raise ValueError(
                'MoonViT cache {} metadata {} must not be empty'.format(path, key)
            )
        return text.lower() if lower else text

    @classmethod
    def _cache_normalization(cls, npz, path):
        value = cls._cache_text(npz, path, 'normalization', lower=True)
        aliases = {
            'moonvit_rgb_mean_0.5_std_0.5': _MOONVIT_NORMALIZATION,
        }
        return aliases.get(value, value)

    @classmethod
    def _cache_checkpoint(cls, npz, path):
        checkpoint = cls._cache_text(npz, path, 'checkpoint')
        return os.path.realpath(os.path.abspath(os.path.expanduser(checkpoint)))

    @staticmethod
    def _cache_scale(npz, path):
        value = np.asarray(Dataset._cache_required(npz, path, 'scale'))
        if value.size != 1:
            raise ValueError(
                'MoonViT cache {} metadata scale must be scalar, got shape {}'.format(
                    path, value.shape
                )
            )
        try:
            scale = float(value.reshape(-1)[0])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                'MoonViT cache {} metadata scale must be finite and positive, got {!r}'.format(
                    path, value
                )
            ) from exc
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                'MoonViT cache {} metadata scale must be finite and positive, got {!r}'.format(
                    path, scale
                )
            )
        return np.asarray([scale], dtype=np.float32)

    def _load_sagittal_moonvit_file(self, row):
        path = self._sagittal_moonvit_cache_path(row)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                'MoonViT cache missing for case_id={!r}, slice_idx={}; expected {}'.format(
                    row['case_id'], row['slice_idx'], path
                )
            )

        with np.load(path, allow_pickle=False) as npz:
            # Load and concatenate all configured layer keys along the channel dim.
            # For a single key the output is [C, H, W]; for two keys [2C, H, W].
            layer_tensors = []
            for feature_key in self.locate_feat_keys:
                if feature_key not in npz.files:
                    raise ValueError(
                        'MoonViT cache {} is missing feature key {!r}; available keys={}'.format(
                            path, feature_key, list(npz.files)
                        )
                    )
                layer = np.asarray(npz[feature_key])
                if layer.dtype != np.float16:
                    raise ValueError(
                        'MoonViT cache {} feature {!r} has dtype {}; expected float16'.format(
                            path, feature_key, layer.dtype
                        )
                    )
                if layer.ndim != 3:
                    raise ValueError(
                        'MoonViT cache {} feature {!r} must be [C,H,W], got shape {}'.format(
                            path, feature_key, layer.shape
                        )
                    )
                if layer.shape[0] != _MOONVIT_LAYER_CHANNELS:
                    raise ValueError(
                        'MoonViT cache {} feature {!r} must have {} channels, got shape {}'.format(
                            path, feature_key, _MOONVIT_LAYER_CHANNELS, layer.shape
                        )
                    )
                layer_tensors.append(layer)
            feature = np.concatenate(layer_tensors, axis=0) if len(layer_tensors) > 1 else layer_tensors[0]
            feature_key = self.locate_feat_key  # primary key (for grid metadata checks)

            grid_hw = self._cache_integer_vector(npz, path, 'grid_hw', 2, minimum=1)
            if tuple(layer_tensors[0].shape[-2:]) != tuple(grid_hw.tolist()):
                raise ValueError(
                    'MoonViT cache {} feature {!r} spatial shape {} disagrees with grid_hw={}'.format(
                        path, feature_key, tuple(layer_tensors[0].shape[-2:]), grid_hw.tolist()
                    )
                )
            orig_hw = self._cache_integer_vector(npz, path, 'orig_hw', 2, minimum=1)
            resized_hw = self._cache_integer_vector(npz, path, 'resized_hw', 2, minimum=1)
            padded_hw = self._cache_integer_vector(npz, path, 'padded_hw', 2, minimum=1)
            pad = self._cache_integer_vector(npz, path, 'pad', 4, minimum=0)
            patch_size = self._cache_integer(
                self._cache_required(npz, path, 'patch_size'), path, 'patch_size', minimum=1
            )
            input_size = self._cache_integer(
                self._cache_required(npz, path, 'input_size'), path, 'input_size', minimum=1
            )
            scale = self._cache_scale(npz, path)
            normalization = self._cache_normalization(npz, path)
            checkpoint = self._cache_checkpoint(npz, path)
            case_id = self._cache_text(npz, path, 'case_id')
            slice_idx = self._cache_integer(
                self._cache_required(npz, path, 'slice_idx'), path, 'slice_idx', minimum=0
            )

            if case_id != str(row['case_id']):
                raise ValueError(
                    'MoonViT cache {} case_id={!r} does not match manifest case_id={!r}'.format(
                        path, case_id, row['case_id']
                    )
                )
            if slice_idx != int(row['slice_idx']):
                raise ValueError(
                    'MoonViT cache {} slice_idx={} does not match manifest slice_idx={}'.format(
                        path, slice_idx, row['slice_idx']
                    )
                )
            if 'manifest_split' in npz.files:
                manifest_split = self._cache_text(npz, path, 'manifest_split')
                if manifest_split != self.manifest_split:
                    raise ValueError(
                        'MoonViT cache {} manifest_split={!r} does not match {!r}'.format(
                            path, manifest_split, self.manifest_split
                        )
                    )
            if 'feature_key' in npz.files:
                stored_key = self._cache_text(npz, path, 'feature_key')
                if stored_key != feature_key:
                    raise ValueError(
                        'MoonViT cache {} feature_key={!r} does not match {!r}'.format(
                            path, stored_key, feature_key
                        )
                    )

            if input_size != self.locate_feat_expected_input_size:
                raise ValueError(
                    'MoonViT cache {} input_size={} does not match expected input_size={}'.format(
                        path, input_size, self.locate_feat_expected_input_size
                    )
                )
            if patch_size != self.locate_feat_expected_patch_size:
                raise ValueError(
                    'MoonViT cache {} patch_size={} does not match expected patch_size={}'.format(
                        path, patch_size, self.locate_feat_expected_patch_size
                    )
                )
            if normalization != self.locate_feat_expected_normalization:
                raise ValueError(
                    'MoonViT cache {} normalization={!r} does not match expected normalization={!r}'.format(
                        path, normalization, self.locate_feat_expected_normalization
                    )
                )
            if (
                self.locate_feat_expected_checkpoint is not None
                and checkpoint != self.locate_feat_expected_checkpoint
            ):
                raise ValueError(
                    'MoonViT cache {} checkpoint={!r} does not match expected checkpoint={!r}'.format(
                        path, checkpoint, self.locate_feat_expected_checkpoint
                    )
                )

            expected_scale = float(input_size) / float(max(orig_hw.tolist()))
            if not np.isclose(float(scale[0]), expected_scale, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    'MoonViT cache {} scale={} disagrees with orig_hw={} and input_size={}; expected {}'.format(
                        path, float(scale[0]), orig_hw.tolist(), input_size, expected_scale
                    )
                )
            expected_resized = np.maximum(
                np.rint(orig_hw.astype(np.float64) * expected_scale).astype(np.int32),
                1,
            )
            if not np.array_equal(resized_hw, expected_resized):
                raise ValueError(
                    'MoonViT cache {} resized_hw={} disagrees with orig_hw={}, scale={}; expected {}'.format(
                        path,
                        resized_hw.tolist(),
                        orig_hw.tolist(),
                        expected_scale,
                        expected_resized.tolist(),
                    )
                )

            expected_grid = np.ceil(
                resized_hw.astype(np.float64) / float(patch_size)
            ).astype(np.int32)
            if not np.array_equal(grid_hw, expected_grid):
                raise ValueError(
                    'MoonViT cache {} grid_hw={} is not the minimal patch grid for resized_hw={} and patch_size={}; expected {}'.format(
                        path,
                        grid_hw.tolist(),
                        resized_hw.tolist(),
                        patch_size,
                        expected_grid.tolist(),
                    )
                )
            expected_padded = expected_grid.astype(np.int64) * int(patch_size)
            if not np.array_equal(padded_hw, expected_padded.astype(np.int32)):
                raise ValueError(
                    'MoonViT cache {} padded_hw={} does not equal minimal grid*patch_size={}'.format(
                        path, padded_hw.tolist(), expected_padded.tolist()
                    )
                )
            expected_pad = np.asarray(
                [
                    0,
                    0,
                    int(padded_hw[1]) - int(resized_hw[1]),
                    int(padded_hw[0]) - int(resized_hw[0]),
                ],
                dtype=np.int32,
            )
            if np.any(expected_pad < 0) or not np.array_equal(pad, expected_pad):
                raise ValueError(
                    'MoonViT cache {} pad={} disagrees with resized_hw={} and padded_hw={}; expected {}'.format(
                        path,
                        pad.tolist(),
                        resized_hw.tolist(),
                        padded_hw.tolist(),
                        expected_pad.tolist(),
                    )
                )

            return {
                'feature': np.ascontiguousarray(feature),
                'grid_hw': np.asarray(grid_hw, dtype=np.int32),
                'orig_hw': np.asarray(orig_hw, dtype=np.int32),
                'resized_hw': np.asarray(resized_hw, dtype=np.int32),
                'padded_hw': np.asarray(padded_hw, dtype=np.int32),
                'pad': np.asarray(pad, dtype=np.int32),
                'scale': scale,
                'patch_size': np.asarray([patch_size], dtype=np.int32),
                'input_size': input_size,
                'normalization': normalization,
                'checkpoint': checkpoint,
                'path': path,
            }

    def _load_locate_feature(self, row, expected_orig_hw=None):
        """从磁盘加载当前样本切片的 MoonViT layer_18 离线特征。

        仅加载中心切片特征 (center-slice)，并校验特征张量尺寸、原始分辨率一致性。

        Args:
            row: 当前样本切片的元数据记录字典。
            expected_orig_hw: 期望的原始图像分辨率 (H, W)。

        Returns:
            list[torch.Tensor]: 包含中心切片特征的张量列表。
        """
        if not self.locate_feat_enabled:
            return {}
        center_entry = self._load_sagittal_moonvit_file(row)
        if expected_orig_hw is not None:
            expected_orig = tuple(int(value) for value in expected_orig_hw)
            if tuple(center_entry['orig_hw'].tolist()) != expected_orig:
                raise ValueError(
                    'MoonViT cache {} orig_hw={} does not match center image shape={}'.format(
                        center_entry['path'], center_entry['orig_hw'].tolist(), expected_orig
                    )
                )

        feature = center_entry['feature'].astype(np.float16, copy=False)
        return {
            'locate_feat': np.ascontiguousarray(feature),
            'locate_feat_grid_hw': center_entry['grid_hw'].copy(),
            'locate_feat_orig_hw': center_entry['orig_hw'].copy(),
            'locate_feat_resized_hw': center_entry['resized_hw'].copy(),
            'locate_feat_padded_hw': center_entry['padded_hw'].copy(),
            'locate_feat_pad': center_entry['pad'].copy(),
            'locate_feat_scale': center_entry['scale'].copy(),
            'locate_feat_patch_size': center_entry['patch_size'].copy(),
            'locate_feat_path': center_entry['path'],
        }

    @staticmethod
    def _read_grayscale_image(path):
        """从磁盘读取单通道灰度医学切片图像 (支持 8 位或 16 位 PNG)。"""
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError('Image not found or cannot be read: {}'.format(path))
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if image.ndim != 2:
            raise ValueError('Expected a 2D grayscale image at {}, got {}'.format(path, image.shape))
        if image.dtype != np.uint8:
            raise ValueError('Expected uint8 image at {}, got {}'.format(path, image.dtype))
        return image

    @staticmethod
    def _read_mask(path, expected_shape):
        """读取切片的金标准分割掩码图像，并严格断言尺寸与原图匹配。"""
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError('Mask not found or cannot be read: {}'.format(path))
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            raise ValueError('Expected a 2D label mask at {}, got {}'.format(path, mask.shape))
        if mask.dtype not in (np.uint8, np.uint16):
            raise ValueError('Expected uint8 or uint16 label mask at {}, got {}'.format(path, mask.dtype))
        if mask.shape != expected_shape:
            raise ValueError(
                'Image/mask shape mismatch for {}: image {}, mask {}'.format(
                    path, expected_shape, mask.shape
                )
            )
        labels = np.unique(mask)
        invalid = labels[(labels < 0) | (labels > 19)]
        if invalid.size:
            raise ValueError(
                'Mask labels must be in [0, 19] at {}; got {}'.format(
                    path, invalid.tolist()
                )
            )
        return mask

    @staticmethod
    def _mask_to_instances(mask):
        """将多类别分割掩码解析为独立的连通域多边形实例 (Polygon Instances)。

        根据面积过滤极微小伪影，按面积降序截取每个类别的主成分连通域。

        Args:
            mask: 2D 标签掩码数组。

        Returns:
            tuple: (多边形实例坐标列表, 对应的解剖结构类别 ID 列表)
        """
        component_mode = str(
            getattr(cfg, 'sagittal_component_mode', 'largest')
        ).strip().lower()
        if component_mode not in ('largest', 'significant'):
            raise ValueError(
                "cfg.sagittal_component_mode must be 'largest' or "
                "'significant', got {!r}".format(component_mode)
            )
        max_components_per_class = max(
            int(getattr(cfg, 'sagittal_max_components_per_class', 4)), 1
        )
        max_instances_per_slice = max(
            int(getattr(cfg, 'sagittal_max_instances_per_slice', 32)), 1
        )
        min_component_area = max(
            float(getattr(cfg, 'sagittal_min_component_area_raw', 2.0)), 0.0
        )
        candidates = []
        for label in np.unique(mask):
            label = int(label)
            if label == 0:
                continue
            binary = np.asarray(mask == label, dtype=np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            valid_contours = [c for c in contours if len(c.reshape(-1, 2)) >= 4]
            if not valid_contours:
                continue
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            if component_mode == 'largest':
                valid_contours = valid_contours[:1]
            else:
                valid_contours = [
                    contour
                    for contour in valid_contours[:max_components_per_class]
                    if float(cv2.contourArea(contour)) >= min_component_area
                ]
            for contour in valid_contours:
                candidates.append((
                    float(cv2.contourArea(contour)),
                    label,
                    contour.reshape(-1, 2).astype(np.float32),
                ))

        # A single polygon cannot represent disconnected anatomy.  Treat each
        # retained component as an instance of the same vertebra class.  The
        # global cap prevents noisy masks from recreating an unbounded
        # 51-contour OOM case.  Full-dataset audit for top-4/area>=2/global-32:
        # 99.52% train and 99.73% validation foreground-pixel coverage.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        candidates = candidates[:max_instances_per_slice]
        candidates.sort(key=lambda item: (item[1], -item[0]))
        instance_polys = []
        cls_ids = []
        for _, label, poly in candidates:
            instance_polys.append([poly.astype(np.float32)])
            cls_ids.append(label)
        return instance_polys, cls_ids

    def get_valid_polys(self, instance_polys, inp_out_hw):
        """Same as VocDataset.get_valid_polys but with a configurable area floor.

        The inherited implementation hard-codes ``Polygon(poly).area > 5`` in the
        1/4-resolution output space.  Sagittal vertebrae are small: a vertebra
        spanning ~100 px^2 in a 616x473 slice shrinks to ~3 px^2 at 128x128, so
        the default threshold silently drops entire slices worth of GT (measured:
        26 of 209 foreground val slices lose every instance).  Lowering the floor
        keeps them.
        """
        from shapely.geometry import Polygon

        min_area = float(getattr(cfg, 'min_poly_area_output', 5.0))
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = [poly for poly in instance if Polygon(poly).area > min_area]
            polys = snake_voc_utils.get_cw_polys(polys)
            polys = [
                poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])]
                for poly in polys
            ]
            instance_polys_.append(polys)
        return instance_polys_

    def __getitem__(self, index):
        """获取单个训练/评估样本数据。

        数据流水线：
        1. 读取切片灰度图、金标准分割 Mask、离线 MoonViT 缓存特征
        2. 计算空间仿射变换矩阵 (中心对齐、随机平移、尺度缩放、水平翻转)
        3. 对图像与 Mask 应用仿射变换并进行像素值标准化
        4. 提取多边形实例，并重采样为 128 点闭合多边形作为演化金标准
        5. 生成 Route-B 初始八边形轮廓 (若在训练集则应用多维度几何抖动增强)
        6. 打包构建包含图像输入、点坐标、特征图索引、初始轮廓与 GT 目标的完整 Batch 字典

        Args:
            index: 样本索引。

        Returns:
            dict: 包含网络前向与损失计算所需全部字段的数据字典。
        """
        row = self.records[index]
        center_image = self._read_grayscale_image(row['image_path'])

        locate_feat = {}
        if self.locate_feat_enabled:
            locate_feat = self._load_locate_feature(
                row, expected_orig_hw=center_image.shape
            )

        image = np.repeat(center_image[:, :, None], 3, axis=2)

        height, width = center_image.shape
        mask = self._read_mask(row['mask_path'], center_image.shape)
        instance_polys, cls_ids = self._mask_to_instances(mask)

        augmented, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_voc_utils.augment(
                image, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                self.mean, self.std, instance_polys,
                color_aug=self.color_aug,
                lr_flip=self.lr_flip,
                random_crop=self.random_crop,
            )
        orig_img = np.repeat(augmented[:, :, 1:2], 3, axis=2)
        instance_polys = self.transform_original_data(
            instance_polys, flipped, width, trans_output, inp_out_hw
        )
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        extreme_points = self.get_extreme_points(instance_polys)

        output_h, output_w = inp_out_hw[2:]
        wh = []
        ct_cls = []
        ct_ind = []

        i_it_4pys = []
        c_it_4pys = []
        i_gt_4pys = []
        c_gt_4pys = []
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []

        for cls_id, instance, instance_points in zip(
                cls_ids, instance_polys, extreme_points):
            for poly, extreme_point in zip(instance, instance_points):
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue
                bbox = [x_min, y_min, x_max, y_max]

                self.prepare_box_target(
                    bbox, cls_id, output_w, wh, ct_cls, ct_ind
                )
                self.prepare_init(
                    bbox, extreme_point,
                    i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys,
                    output_h, output_w,
                )
                self.prepare_evolution(
                    poly, extreme_point, i_it_4pys[-1],
                    i_it_pys, c_it_pys, i_gt_pys, c_gt_pys,
                )

        inv_trans_input = cv2.invertAffineTransform(trans_input).astype(np.float32)
        anchors = []
        for polygon in i_gt_pys:
            raw = np.asarray(polygon, dtype=np.float32) * float(snake_config.down_ratio)
            raw = raw @ inv_trans_input[:, :2].T + inv_trans_input[:, 2]
            if flipped:
                raw[:, 0] = width - raw[:, 0] - 1
            anchors.append(raw)
        dense_gt, dense_weights = _dense_mask_targets(
            mask, ct_cls, anchors, trans_output, flipped)

        ret = {
            'inp': inp,
            'orig_img': orig_img,
            'img_path': row['image_path'],
            'wh': wh,
            'ct_cls': ct_cls,
            'ct_ind': ct_ind,
            'i_it_4py': i_it_4pys,
            'c_it_4py': c_it_4pys,
            'i_gt_4py': i_gt_4pys,
            'c_gt_4py': c_gt_4pys,
            'i_it_py': i_it_pys,
            'c_it_py': c_it_pys,
            'i_gt_py': i_gt_pys,
            'c_gt_py': c_gt_pys,
            'dense_gt': dense_gt,
            'dense_weights': dense_weights,
        }
        if locate_feat:
            ret.update(locate_feat)

        ret['meta'] = {
            'is_train': self.split == 'train',
            'center': center,
            'scale': scale,
            'ct_num': len(ct_ind),
            'trans_input': trans_input.astype(np.float32),
            'inv_trans_input': inv_trans_input,
            'flipped': np.asarray([1 if flipped else 0], dtype=np.float32),
            'orig_hw': np.asarray([height, width], dtype=np.float32),
            'inp_out_hw': np.asarray(inp_out_hw, dtype=np.float32),
            'img_id': int(index),
            'case_id': row['case_id'],
            'slice_idx': int(row['slice_idx']),
        }
        return ret

    def __len__(self):
        """返回数据集样本切片总数。"""
        return len(self.records)
