import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as ticker
import warnings
import os
import tempfile
import shutil
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
warnings.filterwarnings('ignore')

# ========= 路径与场景 =========
shapefile_path = r"F:\Data\Landuse\nineregion_shp\World_shp_byCountry2.shp"
scenario_paths = {
    "SSP126": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\SSP126totalcombined_carbon_emission_2015_2100.tif",
    "SSP245": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\SSP245totalcombined_carbon_emission_2015_2100.tif",
    "SSP370": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\SSP370totalcombined_carbon_emission_2015_2100.tif",
    "SSP585": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\SSP585totalcombined_carbon_emission_2015_2100.tif"
}
save_dir = r"F:\Data\paper\paper1\pic"
os.makedirs(save_dir, exist_ok=True)

# 国家简称映射
abbr_dict = {
    "Peoples Republic of China": "China",
    "United States Of America": "America",
    "Russian Federation": "Russia",
    "Federative Republic of Brazil": "Brazil",
    "Commonwealth of Australia": "Australia",
    "Republic of Indonesia": "Indonesia",
    "Democratic Republic of Congo": "DRC",
"Canada": "Canada"
}

# 情景名称映射（用于展示标准格式）
scenario_name_map = {
    "SSP126": "SSP1-2.6",
    "SSP245": "SSP2-4.5",
    "SSP370": "SSP3-7.0",
    "SSP585": "SSP5-8.5"
}

# ========= 读取 & 坐标检查 =========
print("\n=== 原始shapefile坐标检查 ===")
gdf_original = gpd.read_file(shapefile_path)
print(f"原始坐标系: {gdf_original.crs}")
print(f"原始边界范围: {gdf_original.total_bounds}")

# 转WGS84
gdf = gdf_original.to_crs(epsg=4326)
print("\n=== 转换后的shapefile坐标检查 ===")
print(f"转换后坐标系: {gdf.crs}")
minx, miny, maxx, maxy = gdf.total_bounds
print(f"转换后边界范围 (WGS84): minx={minx:.2f}, miny={miny:.2f}, maxx={maxx:.2f}, maxy={maxy:.2f}")
if maxx > 180 or minx < -180:
    print("警告: 经度范围超出[-180,180]范围!")
if maxy > 90 or miny < -90:
    print("警告: 纬度范围超出[-90,90]范围!")
if gdf.empty:
    raise ValueError("Vector file failed to load or is empty.")
if 'FENAME' not in gdf.columns:
    raise ValueError("Vector missing 'FENAME' field for country names.")

# ========= 计算各情景各国累计（GtC） =========
all_scenarios_df = pd.DataFrame()
all_emissions = []  # 用于统一色带范围
temp_dir = tempfile.mkdtemp()
print(f"\n临时目录创建于: {temp_dir}")

for scenario, raster_path in scenario_paths.items():
    print(f"\nProcessing scenario: {scenario}")
    reprojected_path = os.path.join(temp_dir, f"{scenario}_reprojected.tif")

    # 重投影到EPSG:4326
    with rasterio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({'crs': 'EPSG:4326', 'transform': transform, 'width': width, 'height': height})

        with rasterio.open(reprojected_path, 'w', **kwargs) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs='EPSG:4326',
                    resampling=Resampling.bilinear
                )

    # zonal stats
    stats = zonal_stats(gdf, reprojected_path, stats=['sum'], nodata=-9999)

    emissions = []
    for j, row in gdf.iterrows():
        val = stats[j]['sum'] if stats[j] and stats[j]['sum'] is not None else 0
        emissions.append(val)

    emissions_gtc = np.array(emissions) / 1e7  # 转 GtC
    all_emissions.extend(emissions_gtc)
    gdf[f'{scenario}_emission'] = emissions_gtc

    if all_scenarios_df.empty:
        all_scenarios_df = gdf[['FENAME', f'{scenario}_emission']].copy()
        all_scenarios_df.rename(columns={'FENAME': 'Country'}, inplace=True)
    else:
        all_scenarios_df[f'{scenario}_emission'] = emissions_gtc

# 清理临时目录
try:
    shutil.rmtree(temp_dir)
    print(f"\n临时目录已删除: {temp_dir}")
except Exception as e:
    print(f"删除临时目录时出错: {e}")

# ========= 颜色映射（地图） =========
colors = ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#ffffff",
          "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]
cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
max_abs = np.max(np.abs(all_emissions)) * 1.1
vmin, vmax = -max_abs, max_abs
norm = Normalize(vmin=vmin, vmax=vmax)

# ========= 图1：地图（2×2） =========
fig_map = plt.figure(figsize=(14, 6))
gs_map = fig_map.add_gridspec(2, 2, hspace=0.05, wspace=0.05)
axs_map = [
    fig_map.add_subplot(gs_map[0, 0]),
    fig_map.add_subplot(gs_map[0, 1]),
    fig_map.add_subplot(gs_map[1, 0]),
    fig_map.add_subplot(gs_map[1, 1]),
]

for i, scenario in enumerate(scenario_paths.keys()):
    ax = axs_map[i]
    gdf.boundary.plot(ax=ax, linewidth=0.3, color='gray')
    gdf.plot(column=f'{scenario}_emission',
             ax=ax, cmap=cmap, norm=norm, edgecolor='gray', linewidth=0.3,
             missing_kwds={'color': 'lightgrey'}, legend=False)

    ax.set_xlim(minx-5, maxx+5)
    ax.set_ylim(miny-5, maxy+5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.tick_params(axis='both', which='both', labelsize=12)

    # ==== 修改开始：第一行用顶部刻度，第二行用底部刻度 ====
    xticks = [x for x in ax.get_xticks() if minx <= x <= maxx]
    def fmt_lon(vals):
        out = []
        for x in vals:
            if x < 0: out.append(f'{abs(int(x))}°W')
            elif x > 0: out.append(f'{int(x)}°E')
            else: out.append('0°')
        return out

    if i in [0, 1]:   # 第一行 -> 顶部
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xticks(xticks)
        ax.set_xticklabels(fmt_lon(xticks))
        # 第二行不显示顶部标签，避免重复
    else:             # 第二行 -> 底部
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        ax.set_xticks(xticks)
        ax.set_xticklabels(fmt_lon(xticks))
    # ==== 修改结束 ====

    # 纵轴：左列显示，右列隐藏（保持你的原逻辑）
    if i in [0, 2]:
        yticks = [y for y in ax.get_yticks() if miny <= y <= maxy]
        ax.set_yticks(yticks)
        yticklabels = []
        for y in yticks:
            if y < 0: yticklabels.append(f'{abs(int(y))}°S')
            elif y > 0: yticklabels.append(f'{int(y)}°N')
            else: yticklabels.append('0°')
        ax.set_yticklabels(yticklabels)
    else:
        yticks = [y for y in ax.get_yticks() if miny <= y <= maxy]
        ax.set_yticks(yticks)
        ax.set_yticklabels([])

    # ax.set_title(scenario_name_map[scenario], fontsize=13, pad=6)
# —— 图1四个子图添加 a,b,c,d（左上角）
panel_labels = ['a', 'b', 'c', 'd']
for i, lab in enumerate(panel_labels):
    axs_map[i].text(
        0.02, 0.98, lab,
        transform=axs_map[i].transAxes,
        ha='left', va='top',
        fontsize=16, fontweight='bold', zorder=10,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
    )


# 颜色条（地图）
fig_map.subplots_adjust(right=0.9)
cbar_ax = fig_map.add_axes([0.92, 0.25, 0.02, 0.5])
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cbar = fig_map.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar.set_label('Carbon Emissions (GtC)', fontsize=13, labelpad=12)
cbar.ax.tick_params(labelsize=11)

# 保存地图图
map_path = os.path.join(save_dir, 'global_carbon_maps.png')
fig_map.savefig(map_path, dpi=400, bbox_inches='tight')
print(f"[OK] 地图图已保存：{map_path}")

# ========= 准备分析数据 =========
# 三个重点国家
focus_countries = ['Federative Republic of Brazil', 'United States Of America', 'Canada']
country_colors = {
    'Federative Republic of Brazil': '#1b9e77',
    'United States Of America': '#d95f02',
    'Canada': '#7570b3'
}
# 各情景全球总量
global_emissions = {sc: gdf[f'{sc}_emission'].sum() for sc in scenario_paths.keys()}

# 三国与“其他国家”的分量
country_emissions = {country: [] for country in focus_countries}
other_emissions = []

for sc in scenario_paths.keys():
    sums = []
    for country in focus_countries:
        v = gdf.loc[gdf['FENAME'] == country, f'{sc}_emission']
        v = float(v.values[0]) if len(v.values) > 0 else 0.0
        sums.append(v)
        country_emissions[country].append(v)
    other_val = global_emissions[sc] - sum(sums)
    other_emissions.append(other_val)

scenario_names = [scenario_name_map[sc] for sc in scenario_paths.keys()]

# ========= 发生源汇转换的国家（箱型图数据） =========
country_variability = []
for country in gdf['FENAME'].unique():
    es = []
    for sc in scenario_paths.keys():
        vv = gdf.loc[gdf['FENAME'] == country, f'{sc}_emission']
        if len(vv.values) > 0:
            es.append(float(vv.values[0]))
    if es:
        mn, mx = min(es), max(es)
        if (mn < 0 < mx):  # 跨越0，发生源汇转换
            country_variability.append({
                'country': country,
                'min': mn, 'max': mx, 'range': abs(mx - mn), 'sign_change': True
            })

if country_variability:
    variability_df = pd.DataFrame(country_variability).sort_values(by='range', ascending=False).reset_index(drop=True)
    if len(variability_df) > 3:
        selected_countries = variability_df.head(3)['country'].tolist()
        other_countries_count = len(variability_df) - 3
    else:
        selected_countries = variability_df['country'].tolist()
        other_countries_count = 0
else:
    variability_df = pd.DataFrame(columns=['country','min','max','range','sign_change'])
    selected_countries = []
    other_countries_count = 0
    print("\n警告: 没有国家在情景间发生碳源汇转换!")

# 箱型图数据
boxplot_data, boxplot_labels = [], []
for country in selected_countries:
    es = []
    for sc in scenario_paths.keys():
        vv = gdf.loc[gdf['FENAME'] == country, f'{sc}_emission']
        if len(vv.values) > 0:
            es.append(float(vv.values[0]))
    boxplot_data.append(es)
    boxplot_labels.append(abbr_dict.get(country, country))


# ========= 图2：分析（横向堆叠条 + 箱型图） =========
fig_pan = plt.figure(figsize=(16, 6))  # 高度略降
gs_pan = fig_pan.add_gridspec(1, 2, width_ratios=[0.9, 1.0], wspace=0.12)  # 横向间距更小

# --- 左：横向堆叠条（情景在 y 轴）
ax5 = fig_pan.add_subplot(gs_pan[0, 0])

scenario_names = [scenario_name_map[sc] for sc in scenario_paths.keys()]
y = np.arange(len(scenario_names))

# 条形更细（减小 height）
height = 0.38

left_accum = np.zeros_like(y, dtype=float)


# 依次堆叠三国
for country in focus_countries:
    vals = np.array(country_emissions[country], dtype=float)
    ax5.barh(y, vals, height, left=left_accum, color=country_colors[country],
             label=abbr_dict.get(country, country))
    left_accum = left_accum + vals

# 其他国家
other_vals = np.array(other_emissions, dtype=float)
ax5.barh(y, other_vals, height, left=left_accum, color='#cccccc', label='Other Countries')

# —— 百分比标注：放在“三国堆叠段”中间位置（而不是条形最右）
for i in range(len(scenario_names)):
    three_sum = sum(country_emissions[c][i] for c in focus_countries)
    total = three_sum + other_vals[i]
    pct = (three_sum / total) * 100 if total != 0 else 0.0
    x_center = three_sum / 2.0  # 三国段的中心
    ax5.text(x_center, y[i],
             f'{pct:.0f}%',
             va='center', ha='center', fontsize=12, color='black',
             bbox=dict(boxstyle="round,pad=0.25", facecolor='goldenrod', alpha=0.85, edgecolor='gray'))

# 轴与标签（情景名在 y 轴）
ax5.set_yticks(y)
ax5.set_yticklabels(scenario_names, fontsize=13)
ax5.set_xlabel('Carbon Emissions (GtC)', fontsize=13)
ax5.grid(axis='x', linestyle='--', alpha=0.3)
ax5.legend(loc='lower right', fontsize=12, frameon=False)
# —— 第二行两个子图左上角添加 a, b 标注




# 去掉标题
# ax5.set_title('Contributions by country groups (horizontal stacked)', fontsize=14, pad=6)


# --- 右：箱型图（仅显示前三大源汇转换国家）
ax6 = fig_pan.add_subplot(gs_pan[0, 1])

if boxplot_data:
    positions = np.arange(1, len(boxplot_data) + 1)
    bp = ax6.boxplot(boxplot_data, positions=positions, patch_artist=True,
                     showfliers=False, widths=0.6)
    colors_bp = ['#1b9e77', '#d95f02', '#7570b3']
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_bp[i % len(colors_bp)])
        patch.set_alpha(0.7)

    # 参考线
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.6)

    # 变化范围Δ标注
    for i, es in enumerate(boxplot_data):
        if es:
            rng = max(es) - min(es)
            ax6.text(positions[i], max(es) + 0.1, f'Δ={rng:.2f}',
                     ha='center', fontsize=12, color='black')

    # 省略号占位
    if other_countries_count > 0:
        other_pos = positions[-1] + 1.0
        # 放一个省略号提示（不加真实箱型）
        ax6.text(other_pos, 0.0, '...', ha='center', va='center', fontsize=28, color='gray')
        ax6.text(other_pos, 0.2, 'Other countries', ha='center', va='bottom',
                 fontsize=11, color='black',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
        # 扩展x轴范围以容纳省略号
        ax6.set_xlim(0.5, other_pos + 0.6)

    ax6.set_xticks(np.arange(1, len(boxplot_labels) + 1))
    ax6.set_xticklabels(boxplot_labels, fontsize=12)
    ax6.set_ylabel('Carbon Emissions (GtC)', fontsize=13)
    ax6.grid(axis='y', linestyle='--', alpha=0.3)
    ymin, ymax = ax6.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    ax6.set_ylim(ymin - 0.15 * span, ymax + 0.25 * span)
    # ax6.set_title('Countries with source–sink transitions (boxplot)', fontsize=14, pad=6)
else:
    ax6.text(0.5, 0.5, "No countries show source–sink transition across scenarios",
             ha='center', va='center', fontsize=12, color='red')
    ax6.set_axis_off()
# —— 图2两个子图添加 a,b（左上角）
ax5.text(
    0.92, 0.98, 'a',
    transform=ax5.transAxes,
    ha='left', va='top',
    fontsize=16, fontweight='bold', zorder=10,
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
)
ax6.text(
    0.92, 0.98, 'b',
    transform=ax6.transAxes,
    ha='left', va='top',
    fontsize=16, fontweight='bold', zorder=10,
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
)


fig_pan.tight_layout()
panel_path = os.path.join(save_dir, 'global_carbon_emissions_panels.png')
fig_pan.savefig(panel_path, dpi=400, bbox_inches='tight')
print(f"[OK] 分析图已保存：{panel_path}")

# ========= 导出CSV =========
csv_path = os.path.join(save_dir, 'global_carbon_emissions.csv')
all_scenarios_df.to_csv(csv_path, index=False)
print(f"[OK] CSV数据文件已保存：{csv_path}")
