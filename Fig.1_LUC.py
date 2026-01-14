import numpy as np
import rasterio
from rasterio import warp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
import matplotlib as mpl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import time
import pandas as pd
from osgeo import gdal
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from PIL import Image
import io
import warnings
from matplotlib.ticker import ScalarFormatter
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体

# 忽略无效值警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ====================== 第一部分：空间变化图数据处理 ======================
print("开始处理空间变化图数据...")
start_time_global = time.time()

# 定义情景路径
scenarios_spatial = [
    {
        "name": "SSP1-2.6",
        "code": "126",
        "path_2015": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\SSP126_Reclassified\PFT_2015_126_reclassified.tif",
        "path_2100": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\SSP126_Reclassified\PFT_2100_126_reclassified.tif"
    },
    {
        "name": "SSP2-4.5",
        "code": "245",
        "path_2015": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\SSP245_Reclassified\PFT_2015_245_reclassified.tif",
        "path_2100": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\SSP245_Reclassified\PFT_2100_245_reclassified.tif"
    },
    {
        "name": "SSP3-7.0",
        "code": "370",
        "path_2015": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\SSP370_Reclassified\PFT_2015_370_reclassified.tif",
        "path_2100": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\SSP370_Reclassified\PFT_2100_370_reclassified.tif"
    },
    {
        "name": "SSP5-8.5",
        "code": "585",
        "path_2015": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\SSP585_Reclassified\PFT_2015_585_reclassified.tif",
        "path_2100": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\SSP585_Reclassified\PFT_2100_585_reclassified.tif"
    }
]

# 矢量边界路径
vector_path = r"G:\Data\Landuse\Bound\world_dissolve1.shp"

# 1. 处理矢量边界
print("处理矢量边界...")
world_gdf = None
if os.path.exists(vector_path):
    try:
        # 读取矢量数据
        world_gdf = gpd.read_file(vector_path)
        # 转换为WGS84坐标系
        world_gdf = world_gdf.to_crs(epsg=4326)
        print("矢量边界处理完成")
    except Exception as e:
        print(f"处理矢量数据时出错: {e}")
        world_gdf = None
else:
    print(f"矢量文件不存在: {vector_path}")


# 2. 定义栅格处理函数
def read_and_reproject(raster_path):
    """读取栅格数据并确保其在WGS84坐标系中"""
    try:
        with rasterio.open(raster_path) as src:
            print(f"读取栅格: {raster_path}")
            print(f"原始坐标系: {src.crs}")

            # 如果已经是WGS84坐标系，直接读取
            if src.crs and src.crs.to_epsg() == 4326:
                data = src.read(1)
                extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
                return data, extent, src.transform

            # 重投影到WGS84
            print(f"重投影栅格数据到WGS84...")

            # 计算重投影后的变换和尺寸
            transform, width, height = warp.calculate_default_transform(
                src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)

            # 创建目标数组
            data = np.zeros((height, width), dtype=src.dtypes[0])

            # 执行重投影
            warp.reproject(
                source=src.read(1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs='EPSG:4326',
                resampling=warp.Resampling.nearest)

            # 计算地理范围
            left = transform[2]
            bottom = transform[5] + height * transform[4]
            right = transform[2] + width * transform[0]
            top = transform[5]
            extent = (left, right, bottom, top)

            return data, extent, transform

    except Exception as e:
        print(f"处理栅格数据时出错: {e}")
        return None, None, None


def align_rasters(data1, extent1, transform1, data2, extent2, transform2):
    """对齐两个栅格到相同的空间参考和分辨率"""
    # 确定共同的地理范围
    common_extent = (
        max(extent1[0], extent2[0]),
        min(extent1[1], extent2[1]),
        max(extent1[2], extent2[2]),
        min(extent1[3], extent2[3])
    )

    # 确定输出分辨率（使用第一个栅格的分辨率）
    res_x = transform1[0]
    res_y = transform1[4]

    # 计算输出栅格的尺寸
    width = int((common_extent[1] - common_extent[0]) / abs(res_x))
    height = int((common_extent[3] - common_extent[2]) / abs(res_y))

    # 创建目标变换
    dst_transform = rasterio.Affine(res_x, 0.0, common_extent[0],
                                    0.0, res_y, common_extent[3])

    # 创建目标数组
    aligned_data1 = np.zeros((height, width), dtype=data1.dtype)
    aligned_data2 = np.zeros((height, width), dtype=data2.dtype)

    # 重投影第一个栅格
    warp.reproject(
        source=data1,
        destination=aligned_data1,
        src_transform=transform1,
        src_crs='EPSG:4326',
        dst_transform=dst_transform,
        dst_crs='EPSG:4326',
        resampling=warp.Resampling.nearest
    )

    # 重投影第二个栅格
    warp.reproject(
        source=data2,
        destination=aligned_data2,
        src_transform=transform2,
        src_crs='EPSG:4326',
        dst_transform=dst_transform,
        dst_crs='EPSG:4326',
        resampling=warp.Resampling.nearest
    )

    return aligned_data1, aligned_data2, common_extent


# 3. 创建颜色映射 - 按目标地类分组
print("创建空间变化图颜色映射...")
color_dict = {
    0: '#FFFFFF',  # 无变化-白色
    21: '#A6D96A', 12: '#A6BDDB', 14: '#FEE08B', 13: '#C2A5CF',
    31: '#1A9641', 32: '#67A9CF', 34: '#D6604D', 23: '#9970AB',
    41: '#006837', 42: '#1C6DAB', 43: '#762A83', 24: '#FDAE61'
}

# 重新组织图例项和标签 - 按目标地类分组
legend_codes = [21, 31, 41, 12, 32, 42, 14, 24, 34, 13, 23, 43]

# 对应的标签名称
label_dict = {
    21: "Grassland → Forest",
    31: "Barren → Forest",
    41: "Cropland → Forest",
    12: "Forest → Grassland",
    32: "Barren → Grassland",
    42: "Cropland → Grassland",
    14: "Forest → Cropland",
    24: "Grassland → Cropland",
    34: "Barren → Cropland",
    13: "Forest → Barren",
    23: "Grassland → Barren",
    43: "Cropland → Barren"
}

# 创建图例元素和标签列表
legend_elements = [Rectangle((0, 0), 1, 1, fc=color_dict[code], ec='k', lw=0.5) for code in legend_codes]
legend_labels = [label_dict[code] for code in legend_codes]

# 存储空间变化图数据
spatial_data = {}

# 处理每个情景
for i, scenario in enumerate(scenarios_spatial):
    print(f"处理情景: {scenario['name']}")

    # 读取栅格数据
    data_2015, extent_2015, transform_2015 = read_and_reproject(scenario["path_2015"])
    data_2100, extent_2100, transform_2100 = read_and_reproject(scenario["path_2100"])

    # 对齐栅格
    data_2015_aligned, data_2100_aligned, common_extent = align_rasters(
        data_2015, extent_2015, transform_2015,
        data_2100, extent_2100, transform_2100
    )

    # 创建变化矩阵
    change_matrix = np.zeros_like(data_2015_aligned, dtype=np.int16)

    # 编码规则：十位数=2015年地类，个位数=2100年地类
    # 林地=1, 草地=2, 荒地=3, 耕地=4
    class_codes = [1, 2, 3, 4]
    for old_class in class_codes:
        for new_class in class_codes:
            if old_class != new_class:
                mask = (data_2015_aligned == old_class) & (data_2100_aligned == new_class)
                change_matrix[mask] = old_class * 10 + new_class

    # 存储结果
    spatial_data[scenario['code']] = {
        'change_matrix': change_matrix,
        'extent': common_extent,
        'name': scenario['name']
    }

print(f"空间变化图数据处理完成，耗时: {time.time() - start_time_global:.2f}秒")

# ====================== 第二部分：折线图和桑基图数据处理 ======================
print("开始处理折线图和桑基图数据...")


# 定义类别合并函数
def get_merged_class(cls):
    if 1 <= cls <= 10:
        return "Forest"
    elif 11 <= cls <= 12:
        return "Grassland"
    elif cls == 13:
        return "Barren"
    elif cls == 14:
        return "Cropland"
    elif cls == 15:
        return "Urban"
    else:
        return "Others"


# 定义情景和对应的颜色 (全部使用实线)
scenarios_line = {
    '126': {'name': 'SSP126', 'color': '#1f77b4', 'linestyle': '-'},
    '245': {'name': 'SSP245', 'color': '#ff7f0e', 'linestyle': '-'},
    '370': {'name': 'SSP370', 'color': '#2ca02c', 'linestyle': '-'},
    '585': {'name': 'SSP585', 'color': '#d62728', 'linestyle': '-'}
}

# 定义要分析的六个地类（补充Others类别）
land_classes = ['Forest', 'Grassland', 'Barren', 'Cropland', 'Urban', 'Others']
selected_classes = ['Forest', 'Grassland', 'Barren', 'Cropland']  # 重点关注的四个地类

# 使用更亮丽的颜色方案
class_colors = {
    'Forest':    '#A6D96A',
    'Grassland': '#A6BDDB',
    'Cropland':  '#FEE08B',
    'Barren':    '#C2A5CF',
    'Urban':     '#9C27B0',  # 保持原样
    'Others':    '#607D8B'   # 保持原样
}

# 基础路径
base_dir = r"G:\Data\Landuse\PFT_5KM"

# 像元面积 (5km分辨率)
pixel_area = 25  # km²

# 年份范围 (2015-2100，每5年一个数据点)
years = list(range(2015, 2101, 5))

# 初始化存储结构
results = {}
transition_matrices = {}
sankey_images = {}


def generate_sankey(
        trans_matrix,
        scenario_name=None,  # 不绘制标题
        class_order=None,  # 左列从上到下；右列自动镜像
        font_family="Times New Roman",
        font_size=100,  # 稍降字体以避免小图遮挡
        thickness_px=24,  # 节点厚度（像素）
        gap_px_desired=120,  # 希望的相邻间距（像素）→ 可调大
        node_pad_px=40,  # 让 Plotly 也强制留缝（arrangement='snap' 才生效）
        left_x=0.05, right_x=0.95,  # 两侧位置，留出一点边
        fig_width=1600, fig_height=1300,  # 导出尺寸
        margins=(10, 10, 5, 5),  # (l,r,t,b) 对称外边距
        y_is_center=True  # 大多数版本 y=中心；如发现偏移设 False
):
    import numpy as np
    import plotly.graph_objects as go
    import pandas as pd

    sel = ['Forest', 'Grassland', 'Barren', 'Cropland']
    F = trans_matrix.loc[sel, sel].fillna(0).astype(float)

    # 左右顺序（右侧镜像）
    if class_order is None:
        class_order = sel[:]
    nodes_left = class_order
    nodes_right = class_order[::-1]
    n = len(class_order)

    # 右列索引映射（target 用）
    right_pos_map = {c: i for i, c in enumerate(nodes_right)}

    # 颜色
    node_colors = [class_colors[c] for c in nodes_left] + [class_colors[c] for c in nodes_right]

    # 连线
    source, target, value, link_colors = [], [], [], []
    class_to_num = {'Forest': 1, 'Grassland': 2, 'Barren': 3, 'Cropland': 4}
    for i, fcls in enumerate(nodes_left):
        for tcls in sel:
            if fcls == tcls:
                continue
            flow = float(F.at[fcls, tcls])
            if flow > 0:
                source.append(i)
                target.append(n + right_pos_map[tcls])
                value.append(flow)
                code = class_to_num[fcls] * 10 + class_to_num[tcls]
                link_colors.append(color_dict.get(code, '#CCCCCC'))

    # ===== 用像素计算等距排布，再映射到 0–1（确保左右对齐） =====
    l, r, t, b = margins
    H = max(1, fig_height - t - b)

    total_needed = n * thickness_px + (n - 1) * gap_px_desired
    if total_needed <= H:
        node_th = float(thickness_px)
        gap_px = float(gap_px_desired)
        pad_tb = (H - total_needed) / 2.0
    else:
        s = H / total_needed
        node_th = max(10.0, thickness_px * s)
        gap_px = gap_px_desired * s
        pad_tb = 0.0

    a = 0.2
    y_tops_px = [pad_tb + i * (node_th + gap_px) for i in range(n)]
    y_norm = [((y + node_th / 2) / H) if y_is_center else (y / H) for y in y_tops_px]
    # ===== y 整体上移（单位：轴高的比例）=====
    y_offset_up = 0.16  # 例如整体上移 4%，你可改成 0.02~0.08 按需调

    if y_is_center:
        # node.y 是"中心"坐标时，允许范围需要预留半个厚度
        low = (node_th / 2.0) / H
        high = 1.0 - low
    else:
        # node.y 是"顶部"坐标时，允许范围要预留一个厚度
        low = 0.0
        high = 1.0 - (node_th / H)

    # 向上移动就是 "减去" 偏移量；并裁剪到合法范围 [low, high]
    y_norm = [min(max(y - y_offset_up, low), high) for y in y_norm]

    node_y = y_norm + y_norm
    node_x = [left_x] * n + [right_x] * n

    # 粗体标签
    labels = [f"<b>{c}</b>" for c in (nodes_left + nodes_right)]

    fig = go.Figure(go.Sankey(
        # 用 snap：保留我们给的 y，同时让 node.pad 生效，避免小图时"贴在一起"
        arrangement='snap',
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
        node=dict(
            pad=node_pad_px,  # ★ 生效于同列节点之间的最小缝
            thickness=node_th,
            line=dict(color="white", width=2),
            label=labels,
            color=node_colors,
            x=node_x, y=node_y
        ),
        link=dict(source=source, target=target, value=value, color=link_colors)
    ))

    fig.update_layout(
        width=fig_width, height=fig_height, autosize=False,
        margin=dict(l=l, r=r, t=t, b=b),
        font=dict(family=font_family, size=font_size, color="black"),
        plot_bgcolor="white", paper_bgcolor="white"
    )
    return fig


# 在进入循环之前放在外面
first_class_order = None

# 遍历所有情景
for scenario_code, scenario_info in scenarios_line.items():
    print(f"\nProcessing scenario: {scenario_info['name']}")

    # 存储面积结果
    results[scenario_code] = {lc: [] for lc in land_classes}

    # 读取2015年数据作为基准
    file_2015 = f"PFT_2015_{scenario_code}.tif"
    tif_path_2015 = os.path.join(base_dir, f"SSP{scenario_code}_Resampled", file_2015)

    if not os.path.exists(tif_path_2015):
        print(f"  Warning: File {tif_path_2015} not found, skipping scenario")
        continue

    # 读取数据
    dataset_2015 = gdal.Open(tif_path_2015)
    band_2015 = dataset_2015.GetRasterBand(1)
    data_2015 = band_2015.ReadAsArray()
    dataset_2015 = None

    # 初始化转换矩阵
    transition_matrices[scenario_code] = pd.DataFrame(
        0, index=land_classes, columns=land_classes
    )

    # 存储所有年份数据用于转换分析
    all_data = {2015: data_2015}

    # 处理2015年的面积统计
    for lc in land_classes:
        if lc == "Forest":
            mask = (data_2015 >= 1) & (data_2015 <= 10)
        elif lc == "Grassland":
            mask = (data_2015 >= 11) & (data_2015 <= 12)
        else:  # 使用字典统一处理其他类别
            mask = {
                'Barren': data_2015 == 13,
                'Cropland': data_2015 == 14,
                'Urban': data_2015 == 15,
                'Others': (data_2015 < 1) | (data_2015 > 15)
            }[lc]

        count = np.count_nonzero(mask)
        results[scenario_code][lc].append(count * pixel_area)

    # 处理其他年份数据
    for year in years[1:]:
        file_name = f"PFT_{year}_{scenario_code}.tif"
        tif_path = os.path.join(base_dir, f"SSP{scenario_code}_Resampled", file_name)

        if not os.path.exists(tif_path):
            print(f"  Warning: File {tif_path} not found, skipping")
            for lc in land_classes:
                results[scenario_code][lc].append(np.nan)
            continue

        dataset = gdal.Open(tif_path)
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        all_data[year] = data
        dataset = None

        # 统计各类面积
        for lc in land_classes:
            if lc == "Forest":
                mask = (data >= 1) & (data <= 10)
            elif lc == "Grassland":
                mask = (data >= 11) & (data <= 12)
            else:
                mask = {
                    'Barren': data == 13,
                    'Cropland': data == 14,
                    'Urban': data == 15,
                    'Others': (data < 1) | (data > 15)
                }[lc]

            count = np.count_nonzero(mask)
            results[scenario_code][lc].append(count * pixel_area)

    # 计算2015-2100转换矩阵
    if 2100 in all_data:
        start_classes = all_data[2015].flatten()
        end_classes = all_data[2100].flatten()

        # 向量化映射
        vectorized_map = np.vectorize(
            lambda x: "Forest" if (1 <= x <= 10) else
            "Grassland" if (11 <= x <= 12) else
            "Barren" if x == 13 else
            "Cropland" if x == 14 else
            "Urban" if x == 15 else "Others"
        )

        df = pd.DataFrame({
            'Start': vectorized_map(start_classes),
            'End': vectorized_map(end_classes)
        })

        # 计算转换矩阵
        trans_matrix = df.groupby(['Start', 'End']).size().unstack(fill_value=0)
        trans_matrix = trans_matrix.reindex(index=land_classes, columns=land_classes, fill_value=0)
        transition_matrices[scenario_code] = trans_matrix * pixel_area

    # ========= 创建桑基图 =========
    trans_matrix = transition_matrices[scenario_code]

    if scenario_code == '126':
        first_class_order = ['Forest', 'Grassland', 'Barren', 'Cropland']  # 你希望的左侧从上到下
        sankey_fig = generate_sankey(trans_matrix, scenarios_line[scenario_code]['name'],
                                     class_order=first_class_order)
    else:
        sankey_fig = generate_sankey(trans_matrix, scenarios_line[scenario_code]['name'],
                                     class_order=first_class_order)  # 右侧自动镜像
    img_bytes = sankey_fig.to_image(format="png", scale=2, engine="kaleido")
    pil_img = Image.open(io.BytesIO(img_bytes))
    sankey_images[scenario_code] = np.array(pil_img)


# 计算区间净变化量
def calculate_interval_changes(area_series, years, intervals):
    """
    计算每个区间的净变化量

    参数:
    area_series: 面积序列（与years对应）
    years: 年份列表
    intervals: 区间列表，如[(2015, 2035), (2035, 2050), (2050, 2070), (2070, 2090), (2090, 2100)]

    返回:
    interval_years: 区间中点年份
    interval_changes: 每个区间的净变化量
    """
    interval_years = []
    interval_changes = []

    for start_year, end_year in intervals:
        # 找到起始年和结束年的索引
        try:
            start_idx = years.index(start_year)
            end_idx = years.index(end_year)
        except ValueError:
            # 如果年份不在列表中，尝试找到最接近的年份
            start_idx = min(range(len(years)), key=lambda i: abs(years[i] - start_year))
            end_idx = min(range(len(years)), key=lambda i: abs(years[i] - end_year))

        # 计算净变化
        start_area = area_series[start_idx]
        end_area = area_series[end_idx]

        if not np.isnan(start_area) and not np.isnan(end_area):
            change = end_area - start_area
        else:
            change = np.nan

        # 计算区间中点年份
        mid_year = (start_year + end_year) / 2

        interval_years.append(mid_year)
        interval_changes.append(change)

    return interval_years, interval_changes


# 定义五个区间
intervals = [
    (2015, 2035),
    (2035, 2050),
    (2050, 2070),
    (2070, 2090),
    (2090, 2100)
]

# 定义区间标签（用于X轴标注）
interval_labels = ['2015-2035', '2035-2050', '2050-2070', '2070-2090', '2090-2100']

# 计算区间中点（用于绘图位置）
interval_mid_points = [(start + end) / 2 for start, end in intervals]

# 计算每个情景每个地类的区间净变化
interval_net_changes = {}
for scenario_code in scenarios_line:
    interval_net_changes[scenario_code] = {}
    for lc in selected_classes:  # 只计算我们关注的四个地类
        area_series = results[scenario_code][lc]
        interval_years, interval_changes = calculate_interval_changes(area_series, years, intervals)
        interval_net_changes[scenario_code][lc] = {
            'years': interval_years,
            'changes': interval_changes
        }

print(f"折线图和桑基图数据处理完成，总耗时: {time.time() - start_time_global:.2f}秒")

# ====================== 第三部分：创建综合图表 ======================
print("创建综合图表...")


# 创建科学计数法格式化器，但不显示乘幂标记
class SciFormatter(ScalarFormatter):
    def __init__(self, useMathText=True):
        super().__init__(useMathText=useMathText)
        self.set_scientific(True)
        self.set_powerlimits((-3, 4))

    def __call__(self, x, pos=None):
        # 不显示乘幂标记
        return super().__call__(x, pos)


# 创建科学计数法格式化器
sci_formatter = SciFormatter(useMathText=True)

# 创建大型图表 - 5行4列
fig = plt.figure(figsize=(26, 26), dpi=500)  # 增加高度以容纳所有内容

# 定义网格布局：5行4列
gs = GridSpec(5, 4, figure=fig,
              height_ratios=[1.1, 1.1, 1.3, 1.3, 0.3],
              hspace=0.18,  # 增加行间距
              wspace=0.15,  # 增加列间距
              top=0.95, bottom=0.15,
              left=0.02, right=0.97)

# 左上角折线图区域 (第0-1行, 第0-1列)
gs_right_top = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0:2, 0:2],
                                       hspace=0.12, wspace=0.2)  # 增加折线图区域内部间距

# 右上角桑基图区域 (第0-1行, 第2-3列)
gs_left_top = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0:2, 2:4],
                                      hspace=0.05, wspace=0.03)  # 增加桑基图区域内部间距

# ====================== 修改后的折线图部分：区间净变化 ======================
# 森林净变化 (左上角区域第0行第0列)
ax_forest = fig.add_subplot(gs_right_top[0, 0])

# 为每个情景绘制区间净变化
for scenario_code, scenario_info in scenarios_line.items():
    if scenario_code in interval_net_changes and 'Forest' in interval_net_changes[scenario_code]:
        data = interval_net_changes[scenario_code]['Forest']
        # 绘制折线
        ax_forest.plot(
            data['years'],
            data['changes'],
            color=scenario_info['color'],
            linestyle=scenario_info['linestyle'],
            label=scenario_info['name'],
            linewidth=2.5,  # 增加线宽
            marker='o',  # 添加点标记
            markersize=8,  # 标记大小
            markeredgecolor='black',  # 标记边缘颜色
            markeredgewidth=1  # 标记边缘宽度
        )

# 添加基准年垂直线
ax_forest.axvline(x=2015, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

# 设置森林图Y轴范围：-149到99
ax_forest.set_ylim(-1490000, 1190000)

# 添加区域填充
xlim = ax_forest.get_xlim()
ylim = ax_forest.get_ylim()

# 在y>0的区域填充浅绿色
ax_forest.fill_between([xlim[0], xlim[1]], 0, ylim[1],
                       color='lightgreen', alpha=0.2, zorder=0)
# 在y<0的区域填充浅红色
ax_forest.fill_between([xlim[0], xlim[1]], ylim[0], 0,
                       color='lightcoral', alpha=0.2, zorder=0)

# 添加标注 - 向左平移，放在图内
increase_pos = 750000  # Y轴位置
decrease_pos = -1250000  # Y轴位置
ax_forest.text(2082, increase_pos+150000 , 'Increase',
               fontsize=22, ha='center', va='center', color='darkgreen', fontname='Times New Roman')
ax_forest.text(2042, decrease_pos, 'Decrease',
               fontsize=22, ha='center', va='center', color='darkred', fontname='Times New Roman')

# 设置轴标签和格式
ax_forest.set_ylabel("Forest net change (Mha)", fontsize=20, fontname='Times New Roman')
ax_forest.grid(True, linestyle=':', alpha=0.7, zorder=0)
ax_forest.axhline(0, color='k', linewidth=1.0, zorder=0)

# 设置X轴 - 使用区间中点作为刻度位置，区间标签作为刻度标签

ax_forest.set_xticks(interval_mid_points)
ax_forest.set_xticklabels([])
# ax_forest.set_xticklabels(interval_labels, fontsize=16, fontname='Times New Roman', rotation=45, ha='right')

# 不显示X轴标签（去掉"Year"）
ax_forest.tick_params(axis='x', labelbottom=True)  # 显示X轴标签

# 设置x轴范围为2015-2100
ax_forest.set_xlim(2015, 2100)

# 设置y轴格式
ax_forest.tick_params(axis='both', labelsize=20)
ax_forest.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v / 1e4:.0f}'))
ax_forest.yaxis.get_offset_text().set_visible(False)  # 隐藏乘幂标记

# 添加图例
ax_forest.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 15}, ncol=2)

# 添加字母标签 (a)
ax_forest.text(-0.09, 1.06, 'a', transform=ax_forest.transAxes,
               fontsize=26, fontweight='bold', fontname='Times New Roman', va='top')

# 草地净变化 (左上角区域第0行第1列)
ax_grass = fig.add_subplot(gs_right_top[0, 1])

# 为每个情景绘制区间净变化
for scenario_code, scenario_info in scenarios_line.items():
    if scenario_code in interval_net_changes and 'Grassland' in interval_net_changes[scenario_code]:
        data = interval_net_changes[scenario_code]['Grassland']
        # 绘制折线
        ax_grass.plot(
            data['years'],
            data['changes'],
            color=scenario_info['color'],
            linestyle=scenario_info['linestyle'],
            linewidth=2.5,  # 增加线宽
            marker='o',  # 添加点标记
            markersize=8,  # 标记大小
            markeredgecolor='black',  # 标记边缘颜色
            markeredgewidth=1  # 标记边缘宽度
        )

# 添加基准年垂直线
ax_grass.axvline(x=2015, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

# 添加区域填充
xlim = ax_grass.get_xlim()
ylim = ax_grass.get_ylim()
ax_grass.fill_between([xlim[0], xlim[1]], 0, ylim[1],
                      color='lightgreen', alpha=0.2, zorder=0)
ax_grass.fill_between([xlim[0], xlim[1]], ylim[0], 0,
                      color='lightcoral', alpha=0.2, zorder=0)

# 添加标注 - 向左平移，放在图内
ylim = ax_grass.get_ylim()
increase_pos = ylim[1] * 0.8  # Y轴位置
decrease_pos = ylim[0] * 0.8  # Y轴位置
ax_grass.text(2042, increase_pos, 'Increase',
              fontsize=22, ha='center', va='center', color='darkgreen', fontname='Times New Roman')
ax_grass.text(2042, decrease_pos, 'Decrease',
              fontsize=22, ha='center', va='center', color='darkred', fontname='Times New Roman')

# 设置轴标签和格式
ax_grass.set_ylabel("Grassland net change (Mha)", fontsize=20, fontname='Times New Roman')
ax_grass.grid(True, linestyle=':', alpha=0.7, zorder=0)
ax_grass.axhline(0, color='k', linewidth=1.0, zorder=0)

# 设置X轴 - 使用区间中点作为刻度位置，区间标签作为刻度标签
ax_grass.set_xticks(interval_mid_points)
ax_grass.set_xticklabels([])
# ax_grass.set_xticklabels(interval_labels, fontsize=16, fontname='Times New Roman', rotation=45, ha='right')

# 不显示X轴标签（去掉"Year"）
ax_grass.tick_params(axis='x', labelbottom=True)  # 显示X轴标签
ax_grass.set_xlim(2015, 2100)

# 设置y轴格式
ax_grass.tick_params(axis='both', labelsize=20)
ax_grass.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v / 1e4:.0f}'))
ax_grass.yaxis.get_offset_text().set_visible(False)  # 隐藏乘幂标记

# 添加字母标签 (b)
ax_grass.text(-0.09, 1.06, 'b', transform=ax_grass.transAxes,
              fontsize=26, fontweight='bold', fontname='Times New Roman', va='top')

# 荒地净变化 (左上角区域第1行第0列)
ax_barren = fig.add_subplot(gs_right_top[1, 0])

# 为每个情景绘制区间净变化
for scenario_code, scenario_info in scenarios_line.items():
    if scenario_code in interval_net_changes and 'Barren' in interval_net_changes[scenario_code]:
        data = interval_net_changes[scenario_code]['Barren']
        # 绘制折线
        ax_barren.plot(
            data['years'],
            data['changes'],
            color=scenario_info['color'],
            linestyle=scenario_info['linestyle'],
            linewidth=2.5,  # 增加线宽
            marker='o',  # 添加点标记
            markersize=8,  # 标记大小
            markeredgecolor='black',  # 标记边缘颜色
            markeredgewidth=1  # 标记边缘宽度
        )

# 添加基准年垂直线
ax_barren.axvline(x=2015, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

# 添加区域填充
xlim = ax_barren.get_xlim()
ylim = ax_barren.get_ylim()
ax_barren.fill_between([xlim[0], xlim[1]], 0, ylim[1],
                       color='lightgreen', alpha=0.2, zorder=0)
ax_barren.fill_between([xlim[0], xlim[1]], ylim[0], 0,
                       color='lightcoral', alpha=0.2, zorder=0)

# 添加标注 - 向左平移，放在图内
ylim = ax_barren.get_ylim()
increase_pos = ylim[1] * 0.8  # Y轴位置
decrease_pos = ylim[0] * 0.8  # Y轴位置
ax_barren.text(2042, increase_pos, 'Increase',
               fontsize=22, ha='center', va='center', color='darkgreen', fontname='Times New Roman')
ax_barren.text(2042, decrease_pos, 'Decrease',
               fontsize=22, ha='center', va='center', color='darkred', fontname='Times New Roman')

# 设置轴标签和格式
ax_barren.set_ylabel("Barren land net change (Mha)", fontsize=20, fontname='Times New Roman')
ax_barren.grid(True, linestyle=':', alpha=0.7, zorder=0)
ax_barren.axhline(0, color='k', linewidth=1.0, zorder=0)

# 设置X轴 - 使用区间中点作为刻度位置，区间标签作为刻度标签
ax_barren.set_xticks(interval_mid_points)
ax_barren.set_xticklabels(interval_labels, fontsize=16, fontname='Times New Roman', rotation=12, ha='right')

# 不显示X轴标签（去掉"Year"）
ax_barren.tick_params(axis='x', labelbottom=True)  # 显示X轴标签
ax_barren.set_xlim(2015, 2100)

# 设置y轴格式
ax_barren.tick_params(axis='both', labelsize=20)
ax_barren.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v / 1e4:.0f}'))
ax_barren.yaxis.get_offset_text().set_visible(False)  # 隐藏乘幂标记

# 添加字母标签 (c)
ax_barren.text(-0.09, 1.06, 'c', transform=ax_barren.transAxes,
               fontsize=26, fontweight='bold', fontname='Times New Roman', va='top')

# 耕地净变化 (左上角区域第1行第1列)
ax_crop = fig.add_subplot(gs_right_top[1, 1])

# 为每个情景绘制区间净变化
for scenario_code, scenario_info in scenarios_line.items():
    if scenario_code in interval_net_changes and 'Cropland' in interval_net_changes[scenario_code]:
        data = interval_net_changes[scenario_code]['Cropland']
        # 绘制折线
        ax_crop.plot(
            data['years'],
            data['changes'],
            color=scenario_info['color'],
            linestyle=scenario_info['linestyle'],
            linewidth=2.5,  # 增加线宽
            marker='o',  # 添加点标记
            markersize=8,  # 标记大小
            markeredgecolor='black',  # 标记边缘颜色
            markeredgewidth=1  # 标记边缘宽度
        )

# 添加基准年垂直线
ax_crop.axvline(x=2015, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

# 添加区域填充
xlim = ax_crop.get_xlim()
ylim = ax_crop.get_ylim()
ax_crop.fill_between([xlim[0], xlim[1]], 0, ylim[1],
                     color='lightgreen', alpha=0.2, zorder=0)
ax_crop.fill_between([xlim[0], xlim[1]], ylim[0], 0,
                     color='lightcoral', alpha=0.2, zorder=0)

# 添加标注 - 向左平移，放在图内
ylim = ax_crop.get_ylim()
increase_pos = ylim[1] * 0.8  # Y轴位置
decrease_pos = ylim[0] * 0.8  # Y轴位置
ax_crop.text(2042, increase_pos, 'Increase',
             fontsize=22, ha='center', va='center', color='darkgreen', fontname='Times New Roman')
ax_crop.text(2042, decrease_pos, 'Decrease',
             fontsize=22, ha='center', va='center', color='darkred', fontname='Times New Roman')

# 设置轴标签和格式
# 不显示X轴标签（去掉"Year"）
ax_crop.set_ylabel("Cropland net change (Mha)", fontsize=20, fontname='Times New Roman')
ax_crop.grid(True, linestyle=':', alpha=0.7, zorder=0)
ax_crop.axhline(0, color='k', linewidth=1.0, zorder=0)

# 设置X轴 - 使用区间中点作为刻度位置，区间标签作为刻度标签
ax_crop.set_xticks(interval_mid_points)
ax_crop.set_xticklabels(interval_labels, fontsize=16, fontname='Times New Roman', rotation=12, ha='right')

ax_crop.tick_params(axis='x', labelbottom=True)  # 显示X轴标签
ax_crop.set_xlim(2015, 2100)

# 设置y轴格式
ax_crop.tick_params(axis='both', labelsize=20)
ax_crop.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v / 1e4:.0f}'))
ax_crop.yaxis.get_offset_text().set_visible(False)  # 隐藏乘幂标记

# 添加字母标签 (d)
ax_crop.text(-0.09, 1.06, 'd', transform=ax_crop.transAxes,
             fontsize=26, fontweight='bold', fontname='Times New Roman', va='top')

# ====================== 第3行：桑基图 ======================
# 创建桑基图（放在右上角区域）
sankey_positions = [
    (0, 0),  # SSP126 位置
    (0, 1),  # SSP245 位置
    (1, 0),  # SSP370 位置
    (1, 1)  # SSP585 位置
]
sankey_labels = ['e', 'f', 'g', 'h']  # 桑基图标签

for idx, (scenario_code, pos) in enumerate(zip(scenarios_line, sankey_positions)):
    row, col = pos
    ax = fig.add_subplot(gs_left_top[row, col])
    ax.imshow(sankey_images[scenario_code])
    ax.set_aspect('auto')  # 避免被压缩/裁切
    ax.axis('off')

    # 添加桑基图字母标签
    ax.text(-0.03, 1.05, sankey_labels[idx], transform=ax.transAxes,
            fontsize=26, fontweight='bold', fontname='Times New Roman', va='top')

from matplotlib.colors import ListedColormap, BoundaryNorm

# 构造 0–43 共44类的颜色映射数组（未定义的也写成白色）
max_code = 43
color_list = ['#FFFFFF'] * (max_code + 1)
for code, color in color_dict.items():
    color_list[code] = color

# 建立颜色映射和规范化器
cmap = ListedColormap(color_list)
norm = BoundaryNorm(boundaries=range(0, max_code + 2), ncolors=max_code + 1)

# 提取有效转换代码
legend_codes_sorted = sorted(legend_codes)

spatial_labels = ['i', 'j', 'k', 'l']  # 图注标签


def setup_spatial_ax(ax, extent,
                     draw_left=False, draw_bottom=False,
                     draw_right=False, draw_top=False,
                     label_left=None, label_bottom=None,
                     label_right=None, label_top=None):
    """
    设置空间变化图的通用配置

    参数：
    draw_*: 控制是否绘制刻度线
    label_*: 控制是否显示经纬度标签（默认与draw_*相同）
    """

    # 添加经纬度网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0, color='gray', alpha=0.5, linestyle='--')

    # 配置网格线标签
    gl.top_labels = label_top
    gl.right_labels = label_right
    gl.bottom_labels = label_bottom
    gl.left_labels = label_left

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 17, 'color': 'black', 'fontname': 'Times New Roman'}
    gl.ylabel_style = {'size': 17, 'color': 'black', 'fontname': 'Times New Roman'}
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(-90, 91, 30))

    # 设置轴范围
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 添加刻度线
    xmin, xmax, ymin, ymax = extent
    xticks = np.arange(np.floor(xmin / 60) * 60, np.ceil(xmax / 60) * 60 + 60, 60)
    yticks = np.arange(np.floor(ymin / 30) * 30, np.ceil(ymax / 30) * 30 + 30, 30)

    # 计算刻度线长度
    x_ticklen = (xmax - xmin) * 0.01
    y_ticklen = (ymax - ymin) * 0.01

    # 绘制经度刻度线
    for x in xticks:
        if xmin <= x <= xmax:
            if draw_bottom:
                ax.plot([x, x], [ymin, ymin - y_ticklen],
                        color='black', linewidth=1,
                        transform=ccrs.PlateCarree(),
                        clip_on=False)
            if draw_top:
                ax.plot([x, x], [ymax, ymax + y_ticklen],
                        color='black', linewidth=1,
                        transform=ccrs.PlateCarree(),
                        clip_on=False)

    # 绘制纬度刻度线
    for y in yticks:
        if ymin <= y <= ymax:
            if draw_left:
                # 朝外延伸5点（与分辨率无关）
                ax.annotate('',
                            xy=(xmin, y),
                            xytext=(-5, 0),  # 向左5点
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='-',
                                            color='black',
                                            linewidth=1),
                            transform=ccrs.PlateCarree(),
                            clip_on=False)

            if draw_right:
                # 朝外延伸5点
                ax.annotate('',
                            xy=(xmax, y),
                            xytext=(5, 0),  # 向右5点
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='-',
                                            color='black',
                                            linewidth=1),
                            transform=ccrs.PlateCarree(),
                            clip_on=False)

    return ax


# SSP126
if '126' in spatial_data:
    data = spatial_data['126']
    ax_spatial1 = fig.add_subplot(gs[2, 0:2], projection=ccrs.PlateCarree())
    img = ax_spatial1.imshow(
        data['change_matrix'],
        transform=ccrs.PlateCarree(),
        extent=data['extent'],
        cmap=cmap,
        norm=norm,
        interpolation='nearest',
        origin='upper'
    )
    if world_gdf is not None:
        world_gdf.boundary.plot(ax=ax_spatial1, color='black', linewidth=0.5, alpha=0.7)
    ax_spatial1 = setup_spatial_ax(ax_spatial1, data['extent'],
                                   draw_left=True, draw_bottom=True,
                                   draw_right=True, draw_top=False,
                                   label_left=True, label_bottom=None,
                                   label_right=None, label_top=None)
    ax_spatial1.text(-0.03, 1.02, spatial_labels[0], transform=ax_spatial1.transAxes,
                     fontsize=26, fontweight='bold', fontname='Times New Roman', va='top')

# SSP245
if '245' in spatial_data:
    data = spatial_data['245']
    ax_spatial2 = fig.add_subplot(gs[2, 2:4], projection=ccrs.PlateCarree())
    img = ax_spatial2.imshow(
        data['change_matrix'],
        transform=ccrs.PlateCarree(),
        extent=data['extent'],
        cmap=cmap,
        norm=norm,
        interpolation='nearest',
        origin='upper'
    )
    if world_gdf is not None:
        world_gdf.boundary.plot(ax=ax_spatial2, color='black', linewidth=0.5, alpha=0.7)
    ax_spatial2 = setup_spatial_ax(ax_spatial2, data['extent'],
                                   draw_left=True, draw_bottom=True,
                                   draw_right=True, draw_top=False,
                                   label_left=None, label_bottom=None,
                                   label_right=True, label_top=None)
    ax_spatial2.text(-0.03, 1.02, spatial_labels[1], transform=ax_spatial2.transAxes,
                     fontsize=26, fontweight='bold', fontname='Times New Roman', va='top')

# ====================== 第5行：空间变化图 (SSP370和SSP585) ======================

# SSP370
if '370' in spatial_data:
    data = spatial_data['370']
    ax_spatial3 = fig.add_subplot(gs[3, 0:2], projection=ccrs.PlateCarree())
    img = ax_spatial3.imshow(
        data['change_matrix'],
        transform=ccrs.PlateCarree(),
        extent=data['extent'],
        cmap=cmap,
        norm=norm,
        interpolation='nearest',
        origin='upper'
    )
    if world_gdf is not None:
        world_gdf.boundary.plot(ax=ax_spatial3, color='black', linewidth=0.5, alpha=0.7)
    ax_spatial3 = setup_spatial_ax(ax_spatial3, data['extent'],
                                   draw_left=True, draw_bottom=True,
                                   draw_right=True, draw_top=True,
                                   label_left=True, label_bottom=True,
                                   label_right=None, label_top=None)
    ax_spatial3.text(-0.03, 1.02, spatial_labels[2], transform=ax_spatial3.transAxes,
                     fontsize=26, fontweight='bold', fontname='Times New Roman', va='top')

# SSP585
if '585' in spatial_data:
    data = spatial_data['585']
    ax_spatial4 = fig.add_subplot(gs[3, 2:4], projection=ccrs.PlateCarree())
    img = ax_spatial4.imshow(
        data['change_matrix'],
        transform=ccrs.PlateCarree(),
        extent=data['extent'],
        cmap=cmap,
        norm=norm,
        interpolation='nearest',
        origin='upper'
    )
    if world_gdf is not None:
        world_gdf.boundary.plot(ax=ax_spatial4, color='black', linewidth=0.5, alpha=0.7)
    ax_spatial4 = setup_spatial_ax(ax_spatial4, data['extent'],
                                   draw_left=True, draw_bottom=True,
                                   draw_right=True, draw_top=True,
                                   label_left=None, label_bottom=True,
                                   label_right=True, label_top=None)
    ax_spatial4.text(-0.03, 1.02, spatial_labels[3], transform=ax_spatial4.transAxes,
                     fontsize=26, fontweight='bold', fontname='Times New Roman', va='top')

legend_ax = fig.add_subplot(gs[4, :])  # 占用最后一行所有列
legend_ax.axis('off')  # 不显示坐标轴

# 添加图例
legend = legend_ax.legend(
    legend_elements,
    legend_labels,
    loc='center',  # 锚点在 bbox 的中心
    ncol=4,  # 每行 4 个
    mode='expand',  # ★ 关键：横向拉伸，均匀铺满
    bbox_to_anchor=(0, 0, 1, 1),  # ★ 关键：图例铺满 legend_ax
    prop={'family': 'Times New Roman', 'size': 24},
    frameon=True,
    edgecolor='gray',
    framealpha=0.9,
    borderaxespad=0.0,  # 贴边一点
    columnspacing=1.6,  # 列间距可微调
    handlelength=1.4,  # 颜色块长度
    labelspacing=0.6  # 行内/行间距
)

# 保存图像
output_path = os.path.join(r"G:\Data\paper\paper1\pic", "Integrated_Landuse_Analysis_Interval3.png")
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"综合图表已保存至: {output_path}")
plt.close()
print(f"总处理时间: {time.time() - start_time_global:.2f}秒")
