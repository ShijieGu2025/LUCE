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
        "path_2015": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\SSP126_Reclassified\PFT_2015_126_reclassified.tif",
        "path_2100": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\SSP126_Reclassified\PFT_2100_126_reclassified.tif"
    },
    {
        "name": "SSP2-4.5",
        "code": "245",
        "path_2015": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\SSP245_Reclassified\PFT_2015_245_reclassified.tif",
        "path_2100": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\SSP245_Reclassified\PFT_2100_245_reclassified.tif"
    },
    {
        "name": "SSP3-7.0",
        "code": "370",
        "path_2015": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\SSP370_Reclassified\PFT_2015_370_reclassified.tif",
        "path_2100": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\SSP370_Reclassified\PFT_2100_370_reclassified.tif"
    },
    {
        "name": "SSP5-8.5",
        "code": "585",
        "path_2015": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\SSP585_Reclassified\PFT_2015_585_reclassified.tif",
        "path_2100": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\SSP585_Reclassified\PFT_2100_585_reclassified.tif"
    }
]

# 矢量边界路径
vector_path = r"F:\Data\Landuse\Bound\world_dissolve1.shp"

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
    31: "Wasteland → Forest",
    41: "Cropland → Forest",
    12: "Forest → Grassland",
    32: "Wasteland → Grassland",
    42: "Cropland → Grassland",
    14: "Forest → Cropland",
    24: "Grassland → Cropland",
    34: "Wasteland → Cropland",
    13: "Forest → Wasteland",
    23: "Grassland → Wasteland",
    43: "Cropland → Wasteland"
}

# 创建图例元素和标签列表
legend_elements = [Rectangle((0, 0), 1, 1, fc=color_dict[code], ec='k', lw=0.5) for code in legend_codes]
legend_labels = [label_dict[code] for code in legend_codes]
# 用于 colormap 和 norm 的排序版本

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
    'Forest': '#4CAF50',  # 亮绿色
    'Grassland': '#FFC107',  # 亮黄色
    'Barren': '#FF9800',  # 亮橙色
    'Cropland': '#E91E63',  # 亮粉色
    'Urban': '#9C27B0',  # 紫色
    'Others': '#607D8B'  # 蓝灰色
}

# 基础路径
base_dir = r"F:\Data\Landuse\PFT_5KM"

# 像元面积 (5km分辨率)
pixel_area = 25  # km²

# 年份范围 (2015-2100，每5年一个数据点)
years = list(range(2015, 2101, 5))

# 初始化存储结构
results = {}
transition_matrices = {}
sankey_images = {}


def generate_sankey(trans_matrix, scenario_name):
    """优化后的桑基图生成函数"""
    # 过滤四大地类转换
    selected_classes = ['Forest', 'Grassland', 'Barren', 'Cropland']
    filtered_matrix = trans_matrix.loc[selected_classes, selected_classes]

    # 转换关系处理
    source, target, value, link_colors = [], [], [], []
    class_to_num = {c: i + 1 for i, c in enumerate(selected_classes)}  # 森林=1,草地=2,荒地=3,耕地=4

    # 构建连接数据
    for i, src in enumerate(selected_classes):
        for j, tgt in enumerate(selected_classes):
            flow = filtered_matrix.at[src, tgt]
            if flow > 0 and src != tgt:
                source.append(i)  # 左侧节点索引
                target.append(j + len(selected_classes))  # 右侧节点索引
                value.append(flow)

                # 获取颜色编码（根据转换类型）
                trans_code = class_to_num[src] * 10 + class_to_num[tgt]
                link_colors.append(color_dict.get(trans_code, '#CCCCCC'))

    # 节点配置
    node_labels = selected_classes * 2
    node_colors = [class_colors[c] for c in selected_classes] * 2

    # 创建桑基图
    fig = go.Figure(go.Sankey(
        arrangement="snap",  # 使用自由布局
        node=dict(
            pad=60,
            thickness=40,
            line=dict(color="white", width=2),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    ))

    # 优化布局参数
    fig.update_layout(
        font=dict(family='Times New Roman', size=98, weight='bold',color='white'),
        height=1200,
        width=1600,
        margin=dict(t=0, b=0, l=0, r=0),
        plot_bgcolor='white'
    )

    return fig


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

    # 生成桑基图
    sankey_fig = generate_sankey(transition_matrices[scenario_code],
                                 scenarios_line[scenario_code]['name'])
    # 转换为图像并进行维度处理
    img_bytes = sankey_fig.to_image(format="png", scale=2, engine="kaleido")
    pil_img = Image.open(io.BytesIO(img_bytes))
    sankey_images[scenario_code] = np.array(pil_img)

# 计算净变化量
net_changes = {}
for scenario_code in scenarios_line:
    net_changes[scenario_code] = {}
    for lc in land_classes:
        area_series = results[scenario_code][lc]
        changes = [0]  # 2015年无变化
        for i in range(1, len(area_series)):
            if np.isnan(area_series[i]) or np.isnan(area_series[i - 1]):
                changes.append(np.nan)
            else:
                changes.append(area_series[i] - area_series[i - 1])
        net_changes[scenario_code][lc] = changes

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
fig = plt.figure(figsize=(24, 24), dpi=500)  # 增加高度以容纳所有内容

# 定义网格布局：5行4列
gs = GridSpec(5, 4, figure=fig,
              height_ratios=[1.1, 1.1, 1.3, 1.3, 0.2],
              hspace=0.1,  # 增加行间距
              wspace=0.1,  # 增加列间距
              top=0.95, bottom=0.15,
              left=0.02, right=0.97)
# 左上角折线图区域 (第0-1行, 第0-1列)
gs_right_top = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0:2, 0:2],
                                       hspace=0.1, wspace=0.2)  # 增加折线图区域内部间距

# 右上角桑基图区域 (第0-1行, 第2-3列)
gs_left_top = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0:2, 2:4],
                                      hspace=0.1, wspace=0.1)  # 增加桑基图区域内部间距
# ====================== 第1行：净变化折线图 (森林和草地) ======================
# 森林净变化 (左上角区域第0行第0列)
ax_forest = fig.add_subplot(gs_right_top[0, 0])
for scenario_code, scenario_info in scenarios_line.items():
    if scenario_code in net_changes:
        # 直接绘制折线图
        ax_forest.plot(
            years,
            net_changes[scenario_code]['Forest'],
            color=scenario_info['color'],
            linestyle=scenario_info['linestyle'],
            label=scenario_info['name'],
            linewidth=2.5  # 增加线宽
        )

# 添加基准年垂直线
ax_forest.axvline(x=2015, color='red', linestyle='--', alpha=0.7)

# 添加区域填充
ylim = ax_forest.get_ylim()
ax_forest.fill_between(years, 0, ylim[1], color='lightgreen', alpha=0.2)
ax_forest.fill_between(years, ylim[0], 0, color='lightcoral', alpha=0.2)

# 添加标注
ax_forest.text(2090, ylim[1] * 0.9, 'Increase',
               fontsize=20, ha='center', va='center', color='darkgreen', fontname='Times New Roman')
ax_forest.text(2090, ylim[0] * 0.9, 'Decrease',
               fontsize=20, ha='center', va='center', color='darkred', fontname='Times New Roman')

# 设置轴标签和格式
ax_forest.set_ylabel("Forest Net Change (10$^5$ km$^2$)", fontsize=20, fontname='Times New Roman')
ax_forest.grid(True, linestyle=':', alpha=0.7)
ax_forest.axhline(0, color='k', linewidth=0.8)
ax_forest.tick_params(axis='x', labelbottom=False)  # 不显示x轴标签
ax_forest.set_xlim(2015, 2101)
xticks = list(range(2015, 2101, 10))
ax_forest.set_xticks(xticks)
ax_forest.tick_params(axis='both', labelsize=15)
ax_forest.yaxis.set_major_formatter(sci_formatter)  # 使用科学计数法
ax_forest.yaxis.get_offset_text().set_visible(False)  # 隐藏乘幂标记
ax_forest.legend( loc='upper left', prop={'family': 'Times New Roman', 'size': 15}, ncol=2)

# 添加字母标签 (a)
ax_forest.text(-0.09, 1.04, 'a', transform=ax_forest.transAxes,
               fontsize=24, fontweight='bold', fontname='Times New Roman', va='top')

# 草地净变化 (左上角区域第0行第1列)
ax_grass = fig.add_subplot(gs_right_top[0, 1])
for scenario_code, scenario_info in scenarios_line.items():
    if scenario_code in net_changes:
        # 直接绘制折线图
        ax_grass.plot(
            years,
            net_changes[scenario_code]['Grassland'],
            color=scenario_info['color'],
            linestyle=scenario_info['linestyle'],
            label=scenario_info['name'],
            linewidth=2.5  # 增加线宽
        )

# 添加基准年垂直线
ax_grass.axvline(x=2015, color='red', linestyle='--', alpha=0.7)

# 添加区域填充
ylim = ax_grass.get_ylim()
ax_grass.fill_between(years, 0, ylim[1], color='lightgreen', alpha=0.2)
ax_grass.fill_between(years, ylim[0], 0, color='lightcoral', alpha=0.2)

# 添加标注
ax_grass.text(2090, ylim[1] * 0.9, 'Increase',
              fontsize=20, ha='center', va='center', color='darkgreen', fontname='Times New Roman')
ax_grass.text(2090, ylim[0] * 0.9, 'Decrease',
              fontsize=20, ha='center', va='center', color='darkred', fontname='Times New Roman')

# 设置轴标签和格式
ax_grass.set_ylabel("Grassland Net Change (10$^5$ km$^2$)", fontsize=20, fontname='Times New Roman')
ax_grass.grid(True, linestyle=':', alpha=0.7)
ax_grass.axhline(0, color='k', linewidth=0.8)
ax_grass.tick_params(axis='x', labelbottom=False)  # 不显示x轴标签
ax_grass.set_xlim(2015, 2101)
xticks = list(range(2015, 2101, 10))
ax_grass.set_xticks(xticks)
ax_grass.tick_params(axis='both', labelsize=20)
ax_grass.yaxis.set_major_formatter(sci_formatter)  # 使用科学计数法
ax_grass.yaxis.get_offset_text().set_visible(False)  # 隐藏乘幂标记

# 添加字母标签 (b)
ax_grass.text(-0.09, 1.04, 'b', transform=ax_grass.transAxes,
              fontsize=24, fontweight='bold', fontname='Times New Roman', va='top')

# ====================== 第2行：净变化折线图 (荒地和耕地) ======================
# 荒地净变化 (左上角区域第1行第0列)
ax_barren = fig.add_subplot(gs_right_top[1, 0])
for scenario_code, scenario_info in scenarios_line.items():
    if scenario_code in net_changes:
        # 直接绘制折线图
        ax_barren.plot(
            years,
            net_changes[scenario_code]['Barren'],
            color=scenario_info['color'],
            linestyle=scenario_info['linestyle'],
            linewidth=2.5  # 增加线宽
        )

# 添加基准年垂直线
ax_barren.axvline(x=2015, color='red', linestyle='--', alpha=0.7)

# 添加区域填充
ylim = ax_barren.get_ylim()
ax_barren.fill_between(years, 0, ylim[1], color='lightgreen', alpha=0.2)
ax_barren.fill_between(years, ylim[0], 0, color='lightcoral', alpha=0.2)

# 添加标注
ax_barren.text(2090, ylim[1] * 0.9, 'Increase',
               fontsize=20, ha='center', va='center', color='darkgreen', fontname='Times New Roman')
ax_barren.text(2090, ylim[0] * 0.9, 'Decrease',
               fontsize=20, ha='center', va='center', color='darkred', fontname='Times New Roman')

# 设置轴标签和格式
ax_barren.set_ylabel("Barren Land Net Change (10$^5$ km$^2$)", fontsize=20, fontname='Times New Roman')
ax_barren.grid(True, linestyle=':', alpha=0.7)
ax_barren.axhline(0, color='k', linewidth=0.8)
ax_barren.set_xlim(2015, 2101)
xticks = list(range(2015, 2101, 20))
ax_barren.set_xticks(xticks)
ax_barren.tick_params(axis='both', labelsize=18, rotation=10)
ax_barren.yaxis.set_major_formatter(sci_formatter)  # 使用科学计数法
ax_barren.yaxis.get_offset_text().set_visible(False)  # 隐藏乘幂标记

# 添加字母标签 (c)
ax_barren.text(-0.08, 1.04, 'c', transform=ax_barren.transAxes,
               fontsize=24, fontweight='bold', fontname='Times New Roman', va='top')

# 耕地净变化 (左上角区域第1行第1列)
ax_crop = fig.add_subplot(gs_right_top[1, 1])
for scenario_code, scenario_info in scenarios_line.items():
    if scenario_code in net_changes:
        # 直接绘制折线图
        ax_crop.plot(
            years,
            net_changes[scenario_code]['Cropland'],
            color=scenario_info['color'],
            linestyle=scenario_info['linestyle'],
            linewidth=2.5  # 增加线宽
        )

# 添加基准年垂直线
ax_crop.axvline(x=2015, color='red', linestyle='--', alpha=0.7)

# 添加区域填充
ylim = ax_crop.get_ylim()
ax_crop.fill_between(years, 0, ylim[1], color='lightgreen', alpha=0.2)
ax_crop.fill_between(years, ylim[0], 0, color='lightcoral', alpha=0.2)

# 添加标注
ax_crop.text(2090, ylim[1] * 0.9, 'Increase',
             fontsize=20, ha='center', va='center', color='darkgreen', fontname='Times New Roman')
ax_crop.text(2090, ylim[0] * 0.9, 'Decrease',
             fontsize=20, ha='center', va='center', color='darkred', fontname='Times New Roman')

# 设置轴标签和格式
ax_crop.set_ylabel("Cropland Net Change (10$^5$ km$^2$)", fontsize=20, fontname='Times New Roman')
ax_crop.grid(True, linestyle=':', alpha=0.7)
ax_crop.axhline(0, color='k', linewidth=0.8)
ax_crop.set_xlim(2015, 2101)
xticks = list(range(2015, 2101, 20))
ax_crop.set_xticks(xticks)
ax_crop.tick_params(axis='both', labelsize=18, rotation=10)
ax_crop.yaxis.set_major_formatter(sci_formatter)  # 使用科学计数法
ax_crop.yaxis.get_offset_text().set_visible(False)  # 隐藏乘幂标记

# 添加字母标签 (d)
ax_crop.text(-0.09, 1.04, 'd', transform=ax_crop.transAxes,
             fontsize=24, fontweight='bold', fontname='Times New Roman', va='top')

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
    ax.axis('off')

    # 添加桑基图字母标签
    ax.text(-0.04, 1.03, sankey_labels[idx], transform=ax.transAxes,
            fontsize=24, fontweight='bold', fontname='Times New Roman', va='top')


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
# all_codes_sorted = sorted([0] + legend_codes)
# cmap_colors_sorted = [color_dict[code] for code in all_codes_sorted]
# cmap_colors_sorted = [color_dict[code] for code in legend_codes_sorted]
# cmap = ListedColormap(cmap_colors_sorted)
bounds = legend_codes_sorted + [legend_codes_sorted[-1] + 1]
# norm = BoundaryNorm(bounds, ncolors=len(cmap_colors_sorted))


# ====================== 第4行：空间变化图 (SSP126和SSP245) ======================
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
    ax_spatial1.text(-0.02, 1.01, spatial_labels[0], transform=ax_spatial1.transAxes,
                     fontsize=24, fontweight='bold', fontname='Times New Roman', va='top')

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
    ax_spatial2.text(-0.02, 1.01, spatial_labels[1], transform=ax_spatial2.transAxes,
                     fontsize=24, fontweight='bold', fontname='Times New Roman', va='top')

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
    ax_spatial3.text(-0.02, 1.01, spatial_labels[2], transform=ax_spatial3.transAxes,
                     fontsize=24, fontweight='bold', fontname='Times New Roman', va='top')

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
    ax_spatial4.text(-0.02, 1.01, spatial_labels[3], transform=ax_spatial4.transAxes,
                     fontsize=24, fontweight='bold', fontname='Times New Roman', va='top')




legend_ax = fig.add_subplot(gs[4, :])  # 占用最后一行所有列
legend_ax.axis('off')  # 不显示坐标轴

# 添加图例
legend = legend_ax.legend(
    legend_elements,
    legend_labels,
    loc='center',
    ncol=6,  # 4列布局
    prop={'family': 'Times New Roman', 'size': 20},
    frameon=True,
    edgecolor='gray',
    framealpha=0.9
)

# 保存图像
output_path = os.path.join(r"F:\Data\paper\paper1\pic", "Integrated_Landuse_Analysis3.png")
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"综合图表已保存至: {output_path}")
plt.close()
# import os
# import pandas as pd
# import numpy as np
#
# # 输出目录
# out_dir = r"F:\Data\Landuse\PFT_5KM\result"
# os.makedirs(out_dir, exist_ok=True)
# for code, scen_info in scenarios_line.items():
#     scen_name = scen_info['name']
#     # 构造两个表
#     # 1. Area Change
#     area_rows = []
#     for lc in land_classes:
#         series = results[code][lc]
#         a2015 = series[0] if series else np.nan
#         a2100 = series[-1] if len(series) > 1 else np.nan
#         net   = a2100 - a2015 if not np.isnan(a2015) and not np.isnan(a2100) else np.nan
#         area_rows.append({
#             'LandClass':     lc,
#             'Area_2015_km2': a2015,
#             'Area_2100_km2': a2100,
#             'NetChange_km2': net
#         })
#     df_area = pd.DataFrame(area_rows)
#
#     # 2. Detailed Transitions
#     # 读取该场景的转换矩阵
#     # 读取该场景的转换矩阵
#     df_trans = transition_matrices[code]
#
#     # —— 将矩阵展开为三列：From, To, Area_km2 —— #
#     # 1. 重置索引，把旧索引变成一列
#     df_trans = df_trans.reset_index()
#
#     # 2. 找到刚刚新生成的那一列的名字（可能是 'index'，也可能是 'Start'）
#     id_col = df_trans.columns[0]
#
#     # 3. 重命名为统一的 'From'
#     df_trans = df_trans.rename(columns={id_col: 'From'})
#
#     # 4. melt 成长表
#     df_list = df_trans.melt(
#         id_vars='From',
#         var_name='To',
#         value_name='Area_km2'
#     )
#
#     # 5. 只保留不同地类之间的转换
#     df_list = df_list[df_list['From'] != df_list['To']]
#
#     # 写入同一个 CSV
#     csv_path = os.path.join(out_dir, f"{code}_landuse_summary.csv")
#     with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
#         # 写标题行
#         f.write(f"{scen_name} — Area Change (2015 vs 2100)\n")
#         df_area.to_csv(f, index=False, float_format="%.2f")
#         f.write("\n")  # 空一行
#         f.write(f"{scen_name} — Detailed Transitions (From, To, Area_km2)\n")
#         df_list.to_csv(f, index=False, float_format="%.2f")
#
#     print(f"已生成：{csv_path}")
