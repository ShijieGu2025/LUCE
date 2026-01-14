
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm, rcParams
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from osgeo import gdal, osr, ogr
import os, time, warnings
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning, module="osgeo")
# 如仅保存图片且不需要弹窗，可解开下一行强制使用无交互后端：
# mpl.use("Agg")

# ================= 全局样式（字体放大） =================
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 20            # 原18 -> 20
rcParams['mathtext.default'] = 'regular'

# ================= 地图数据路径（四个情景） =================
raster_paths = [
    r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\SSP126totalcombined_carbon_emission_2015_2100.tif",
    r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\SSP245totalcombined_carbon_emission_2015_2100.tif",
    r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\SSP370totalcombined_carbon_emission_2015_2100.tif",
    r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\SSP585totalcombined_carbon_emission_2015_2100.tif"
]
vector_path = r"G:\Data\Landuse\Bound\world_dissolve_WGS84.shp"

# ================= 统计用 Excel（阶段净变化 + 累积图） =================
SCENARIO_PATHS = {
    "SSP1-2.6": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\out\out2\SSP126_combined_carbon_storage_and_emission_summary.xlsx",
    "SSP2-4.5": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\out\out3\SSP245_combined_carbon_storage_and_emission_summary.xlsx",
    "SSP3-7.0": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\out\out4\SSP370_combined_carbon_storage_and_emission_summary.xlsx",
    "SSP5-8.5": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\out\out3\SSP585_combined_carbon_storage_and_emission_summary.xlsx"
}
SCENARIOS = list(SCENARIO_PATHS.keys())
PERIODS = [(2015, 2035), (2035, 2050), (2050, 2070), (2070, 2090), (2090, 2100)]

# ================= "全球 CSV（15to6）"用于总源/总汇 =================
GLOBAL_CSV = {
    "SSP1-2.6": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_global_15to6.csv",
    "SSP2-4.5": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_global_15to6.csv",
    "SSP3-7.0": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_global_15to6.csv",
    "SSP5-8.5": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_global_15to6.csv",
}

# CSV 换算到 Gt：若 CSV 已是 Gt -> 1；若是 t -> 1e9；若是"10^7 t" -> 1e7（与你之前一致）
CSV_UNIT_SCALE = 1e7

# 输出路径
output_png = r"G:\Data\paper\paper1\pic\Combined_Carbon_Visualization_final6.png"

# 颜色
SCENARIO_COLORS = {
    'SSP1-2.6': '#1f77b4',
    'SSP2-4.5': '#ff7f0e',
    'SSP3-7.0': '#2ca02c',
    'SSP5-8.5': '#d62728'
}
COLOR_PALETTE = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#FFA500']


# ================= Excel -> 年均（Gt/yr） =================
def process_scenario_data(path):
    df = pd.read_excel(path, sheet_name='Combined_Carbon_Emission_Sums', engine='openpyxl')
    df[['start', 'end']] = df['Year_Range'].str.split('-', expand=True).astype(int)
    df['annual_emission'] = df['Combined_Carbon_Emission_Sum'] / 1e7 / 5.0
    return df


def calculate_cumulative(df, start, end):
    mask = (df['start'] >= start) & (df['end'] <= end)
    return df.loc[mask, 'annual_emission'].sum() * 5.0  # 总量（Gt）


def calculate_directional_stats(series):
    """
    定向误差：若均值>=0（净排），只给向上误差；若均值<0（净汇），只给向下误差。
    """
    if len(series) == 0:
        return {'mean': 0.0, 'lower_err': 0.0, 'upper_err': 0.0}
    mean = np.nanmean(series)
    std = np.nanstd(series, ddof=1) if series.size > 1 else 0.0
    return {'mean': mean, 'lower_err': std if mean < 0 else 0.0, 'upper_err': std if mean >= 0 else 0.0}


def plot_period_bars(ax, df, scenario_name, show_title=True, show_ylabel=False):
    stats = []
    for (s, e) in PERIODS:
        ser = df[(df['start'] >= s) & (df['end'] <= e)]['annual_emission']
        stats.append(calculate_directional_stats(ser))

    # 只保留均值
    means = [s['mean'] for s in stats]

    # —— 柱子参数 ——
    bar_width = 0.8

    # 只画柱子，不画误差线
    ax.bar(
        range(len(means)),
        means,
        color=COLOR_PALETTE,
        width=bar_width,
    )

    # 0 线
    ax.axhline(0, color='#444444', lw=1.5, zorder=0)

    # x 轴
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels(
        [f"{s}-{e}" for s, e in PERIODS],
        rotation=15,
        fontsize=19
    )

    # y 轴格式 & 网格
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    if show_ylabel:
        ax.set_ylabel('Annual net fluxes (Gt yr$^{-1}$)', fontsize=20)
    if show_title:
        ax.set_title(scenario_name, fontsize=18, pad=8)




# === 统一最后一行 y 轴范围辅助（基于定向误差计算） ===
def collect_period_stats(df):
    """返回每个阶段的 mean, lower_err, upper_err 列表（供统一 y 轴计算使用）"""
    stats = []
    for (s, e) in PERIODS:
        ser = df[(df['start'] >= s) & (df['end'] <= e)]['annual_emission']
        stats.append(calculate_directional_stats(ser))
    means = np.array([s['mean'] for s in stats], dtype=float)
    lowers = np.array([s['lower_err'] for s in stats], dtype=float)
    uppers = np.array([s['upper_err'] for s in stats], dtype=float)
    return means, lowers, uppers


# ================= Positive/Negative 计算总源/总汇 =================
def compute_sources_sinks_from_global_csv():
    """
    Sources = sum(Positive)
    Sinks   = sum(|Negative|)   # 逐行取绝对值
    Net     = Sources - Sinks
    转为 Gt。
    """
    rows = []
    for scen, path in GLOBAL_CSV.items():
        if not os.path.exists(path):
            print(f"[警告] 找不到文件：{path}")
            rows.append((scen, 0.0, 0.0, 0.0))
            continue
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if ('positive' in cols) and ('negative' in cols):
            pos = pd.to_numeric(df[cols['positive']], errors='coerce').fillna(0.0)
            neg = pd.to_numeric(df[cols['negative']], errors='coerce').fillna(0.0)
            sources = pos.sum() / CSV_UNIT_SCALE
            sinks = np.abs(neg).sum() / CSV_UNIT_SCALE
        else:
            key = cols.get('net', 'Net') if ('net' in cols or 'Net' in df.columns) else None
            if key is None:
                print(f"[警告] 文件缺少 Positive/Negative 与 Net 列：{path}")
                rows.append((scen, 0.0, 0.0, 0.0))
                continue
            net = pd.to_numeric(df[key], errors='coerce').fillna(0.0)
            sources = net[net > 0].sum() / CSV_UNIT_SCALE
            sinks = (-net[net < 0]).sum() / CSV_UNIT_SCALE
        net_total = sources - sinks
        rows.append((scen, sources, sinks, net_total))
    return (pd.DataFrame(rows, columns=['Scenario', 'Sources', 'Sinks', 'Net'])
            .set_index('Scenario').loc[SCENARIOS].reset_index())


# ================= 绘图 =================
def main():
    start_time = time.time()
    fig = plt.figure(figsize=(22, 18))

    # 调整 GridSpec：减小第一、二行之间的间距
    gs = GridSpec(4, 4, figure=fig,
                  height_ratios=[1.0, 1.0, 0.9, 0.9],
                  width_ratios=[1, 1, 1, 1],
                  hspace=0.15, wspace=0.12,
                  top=0.95, bottom=0.06, left=0.06, right=0.95)

    # ---------- 顶部两行：地图 ----------
    print("开始处理地图数据...")
    map_axes = [
        fig.add_subplot(gs[0, 0:2]),  # SSP1-2.6
        fig.add_subplot(gs[0, 2:4]),  # SSP2-4.5
        fig.add_subplot(gs[1, 0:2]),  # SSP3-7.0
        fig.add_subplot(gs[1, 2:4])   # SSP5-8.5
    ]
    global_vmin, global_vmax = float('inf'), float('-inf')
    all_arrs, all_extents = [], []
    vector_paths_list = []

    # 矢量边界
    if os.path.exists(vector_path):
        vds = ogr.Open(vector_path)
        layer = vds.GetLayer()
        for feat in layer:
            geom = feat.GetGeometryRef().Clone()
            if not geom:
                continue
            gtype = geom.GetGeometryType()
            if gtype in [ogr.wkbPolygon, ogr.wkbMultiPolygon]:
                polys = [geom] if gtype == ogr.wkbPolygon else [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())]
                for poly in polys:
                    ring = poly.GetGeometryRef(0)
                    if not ring:
                        continue
                    points = ring.GetPoints()
                    if not points:
                        continue
                    vertices, codes = [], []
                    for i, (x, y, *_) in enumerate(points):
                        if i % 10 == 0:
                            vertices.append((x, y))
                            codes.append(mpath.Path.MOVETO if i == 0 else mpath.Path.LINETO)
                    if vertices:
                        vertices.append(vertices[0])
                        codes.append(mpath.Path.CLOSEPOLY)
                        vector_paths_list.append(mpath.Path(vertices, codes))
        del vds

    # 读取四个 TIF，统一色标范围（这里将数值 ÷ 10）
    for rp in raster_paths:
        ds = gdal.Open(rp)
        if ds is None:
            print(f"[错误] 无法打开栅格：{rp}")
            continue
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(ds.GetProjection())
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(4326)

        if not src_srs.IsSame(dst_srs):
            wds = gdal.Warp('', ds, format='MEM', dstSRS=dst_srs.ExportToWkt())
            band = wds.GetRasterBand(1)
            arr = band.ReadAsArray()
            arr = arr / 10.0   # <<< 缩放为原来的 1/10
            gt = wds.GetGeoTransform()
            cols, rows = wds.RasterXSize, wds.RasterYSize
        else:
            band = ds.GetRasterBand(1)
            arr = band.ReadAsArray()
            arr = arr / 10.0   # <<< 缩放为原来的 1/10
            gt = ds.GetGeoTransform()
            cols, rows = ds.RasterXSize, ds.RasterYSize

        xmin = gt[0]
        xmax = gt[0] + cols * gt[1]
        ymin = gt[3] + rows * gt[5]
        ymax = gt[3]
        nodata = band.GetNoDataValue()
        if nodata is not None:
            arr = np.ma.masked_where((arr == nodata) | (arr == 0), arr)
        else:
            arr = np.ma.masked_where(arr == 0, arr)
        all_arrs.append(arr)
        all_extents.append((xmin, xmax, ymin, ymax))
        global_vmin = min(global_vmin, np.nanmin(arr))
        global_vmax = max(global_vmax, np.nanmax(arr))
        del ds
        if 'wds' in locals():
            del wds

    cmap = LinearSegmentedColormap.from_list(
        "custom_div_cmap",
        ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#fcbba1", "#fb6a4a", "#ef3b2c", "#a50f15"]
    )
    norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)

    for idx, ax in enumerate(map_axes):
        arr = all_arrs[idx]
        xmin, xmax, ymin, ymax = all_extents[idx]
        rows, cols = arr.shape
        x = np.linspace(xmin, xmax, cols + 1)
        y = np.linspace(ymax, ymin, rows + 1)
        X, Y = np.meshgrid(x, y)
        ax.pcolormesh(X, Y, arr, cmap=cmap, norm=norm, shading='auto')
        if vector_paths_list:
            for path in vector_paths_list:
                ax.add_patch(mpatches.PathPatch(path, facecolor='none', edgecolor='black',
                                                linewidth=0.5, alpha=0.8, zorder=3))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, p: f"{abs(v):.0f}°{'E' if v >= 0 else 'W'}" if v else "0°"))
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, p: f"{abs(v):.0f}°{'N' if v >= 0 else 'S'}" if v else "0°"))
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')

        if idx in [2, 3]:  # 第二行地图
            ax.tick_params(axis='x', labelbottom=False)
        else:  # 第一行地图
            ax.tick_params(axis='x', labeltop=(idx in [0, 1]), labelbottom=False)

        ax.tick_params(axis='y', labelleft=(idx in [0, 2]), labelright=False)

    # 创建颜色条，纵向占满第1、2行
    cax = inset_axes(
        map_axes[1],
        width="2.6%",
        height="180%",
        loc="center left",
        bbox_to_anchor=(1.02, -0.7, 1, 1.2),
        bbox_transform=map_axes[1].transAxes,
        borderpad=0
    )
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical', extend='both')
    # 单位标注说明已缩放 0.1
    cbar.set_label('Cumulative carbon fluxes (kt C per grid)', fontsize=22, rotation=270, va='bottom', labelpad=20)
    cax.tick_params(labelsize=18)

    print("地图完成。")

    # ---------- 第三行：左 = 总源/总汇；右 = 仅 2015–2100 累积净排放点 ----------
    gs = GridSpec(4, 4, figure=fig,
                  height_ratios=[1.0, 1.0, 1.0, 1.0],
                  width_ratios=[1, 1, 1, 1],
                  hspace=0.15, wspace=0.12,
                  top=0.9, bottom=0.06, left=0.06, right=1)

    ax_srcsink = fig.add_subplot(gs[2, 0:2])  # 左
    ax_cum = fig.add_subplot(gs[2, 2:4], sharey=ax_srcsink)  # 右，与左共享 y 轴（严格对齐）

    # ---- 左图：总源/总汇 ----
    df_ss = compute_sources_sinks_from_global_csv()
    y_base = np.arange(len(SCENARIOS)) * 3.0

    XMAX = max(1e-9, df_ss[['Sources', 'Sinks']].to_numpy().max())
    ax_srcsink.set_xlim(0, XMAX * 1.05)
    ax_srcsink.set_ylim(-1, y_base[-1] + 2)
    ax_srcsink.grid(axis='x', linestyle=':', alpha=0.6)

    # 设置 ytick（只在左图显示情景名）
    ax_srcsink.set_yticks(y_base)
    ax_srcsink.set_yticklabels(SCENARIOS, fontsize=18)

    # 顶/底部副轴设置
    ax_srcsink.spines['bottom'].set_visible(False)
    ax_top = ax_srcsink.secondary_xaxis('top')
    ax_bottom = ax_srcsink.secondary_xaxis('bottom')
    ax_top.xaxis.set_label_coords(0.0, 1.08)
    ax_top.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax_srcsink.tick_params(
        axis='x', which='both',
        bottom=True, labelbottom=True,
        top=False, labeltop=False
    )
    ax_top.xaxis.set_major_formatter(NullFormatter())
    ax_top.xaxis.set_minor_formatter(NullFormatter())
    ax_top.spines['top'].set_visible(True)
    ax_bottom.spines['bottom'].set_visible(True)
    ax_srcsink.spines['top'].set_visible(False)

    # 左图画线
    for i, row in df_ss.iterrows():
        scen = row['Scenario']
        col = SCENARIO_COLORS[scen]
        y_upper = y_base[i] + 0.6
        y_lower = y_base[i] - 0.6
        ax_srcsink.hlines(y=y_upper, xmin=0, xmax=row['Sources'], color=col, lw=5, alpha=0.95, linestyle='-')
        ax_srcsink.hlines(y=y_lower, xmin=0, xmax=row['Sinks'], color=col, lw=5, alpha=0.95, linestyle=':')

    # —— 图例字体增大 ——（由 12 调为 16）
    ax_srcsink.legend(handles=[
        plt.Line2D([0], [0], color='#444', lw=4, linestyle='-', label='Gross source (Gt)'),
        plt.Line2D([0], [0], color='#444', lw=4, linestyle=':', label='Gross absorb (Gt)'),
    ], bbox_to_anchor=(1.02, 1.04), loc='upper right', frameon=False, fontsize=19)

    # ---- 右图：仅 2015–2100 的累计净排放（单点） ----
    # ---- 右图：单线表示 2015–2100 的累计净排放 ----
    cumulative_full = []
    for scenario, path in SCENARIO_PATHS.items():
        df = process_scenario_data(path)
        cum_full = calculate_cumulative(df, 2015, 2100)
        cumulative_full.append((scenario, cum_full))

    # 与 SCENARIOS 保持同序
    cumulative_full = [(s, f) for s, f in cumulative_full if s in SCENARIOS]
    cumulative_full.sort(key=lambda t: SCENARIOS.index(t[0]))

    # 不显示 y 轴刻度文字，但与左图共享 y 对齐
    ax_cum.tick_params(axis='y', labelleft=False, labelright=False, left=False, right=False, length=0)

    xmax = max(full for _, full in cumulative_full)
    ax_cum.set_xlim(0, xmax * 1.05)

    # 画“单线”：每个情景在其 y 位置，从 0 画到累计值 full
    for i, (scenario, full) in enumerate(cumulative_full):
        y_i = y_base[i]
        col = SCENARIO_COLORS.get(scenario, '#444444')
        ax_cum.hlines(y=y_i, xmin=0, xmax=full, color=col, lw=5, alpha=0.95)

    # 轴与标题样式
    ax_cum.grid(axis='x', linestyle=':', alpha=0.6)
    ax_cum.axvline(0, color='#555', lw=1)
    ax_cum.tick_params(axis='x', bottom=True, labelbottom=True, top=False, labeltop=False, labelsize=20)

    # 图内标题
    ax_cum.text(
        0.61, 0.13, 'Cumulative Net Fluxes (Gt)',
        transform=ax_cum.transAxes, ha='left', va='top', fontsize=22, zorder=10,
        bbox=dict(facecolor='goldenrod', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
    )

    # 简洁图例：单线代表 2015–2100
    legend_handles = [
        Line2D([0], [0], color='#444', lw=5, linestyle='-', label='2015–2100')
    ]
    ax_cum.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=22)

    # 只保留一个图例：2015–2100
    # legend_handles = [Line2D([0], [0], marker='o', linestyle='None', mfc='#444', mec='white', mew=1.0, label='2015–2100')]
    # ax_cum.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=16)

    # —— 再写回左图情景名（防共享轴操作覆盖）
    ax_srcsink.set_yticks(y_base)
    ax_srcsink.set_yticklabels(SCENARIOS, fontsize=20)
    for t in ax_srcsink.get_yticklabels():
        t.set_horizontalalignment('right')
    ax_srcsink.yaxis.set_tick_params(pad=3)

    # ---------- 第四行：四个情景的阶段净变化柱（统一 y 轴范围） ----------
    ax_ssp1   = fig.add_subplot(gs[3, 0])
    ax_ssp245 = fig.add_subplot(gs[3, 1])
    ax_ssp370 = fig.add_subplot(gs[3, 2])
    ax_ssp585 = fig.add_subplot(gs[3, 3])

    # 先读四个场景数据
    df_126  = process_scenario_data(SCENARIO_PATHS["SSP1-2.6"])
    df_245  = process_scenario_data(SCENARIO_PATHS["SSP2-4.5"])
    df_370  = process_scenario_data(SCENARIO_PATHS["SSP3-7.0"])
    df_585  = process_scenario_data(SCENARIO_PATHS["SSP5-8.5"])

    # 统计所有子图的 (mean ± 定向err)，求全局 y 范围
    min_vals, max_vals = [], []
    for _df in [df_126, df_245, df_370, df_585]:
        means, lowers, uppers = collect_period_stats(_df)
        min_vals.append(np.min(means - lowers))
        max_vals.append(np.max(means + uppers))
    ymin = min(min_vals)
    ymax = max(max_vals)
    padding = 0.1 * max(abs(ymin), abs(ymax))
    ymin_u = ymin - padding
    ymax_u = ymax + padding

    # 画四个图
    plot_period_bars(ax_ssp1,   df_126, "SSP1-2.6", show_title=False, show_ylabel=True)
    plot_period_bars(ax_ssp245, df_245, "SSP2-4.5", show_title=False, show_ylabel=False)
    plot_period_bars(ax_ssp370, df_370, "SSP3-7.0", show_title=False, show_ylabel=False)
    plot_period_bars(ax_ssp585, df_585, "SSP5-8.5", show_title=False, show_ylabel=False)

    # 统一 y 轴范围与刻度（字体放大）
    for axp in [ax_ssp1, ax_ssp245, ax_ssp370, ax_ssp585]:
        axp.set_ylim(ymin_u, ymax_u)
    yticks = ticker.MaxNLocator(nbins=6).tick_values(ymin_u, ymax_u)
    for axp in [ax_ssp1, ax_ssp245, ax_ssp370, ax_ssp585]:
        axp.set_yticks(yticks)
        axp.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axp.tick_params(axis='y', labelsize=20)
        axp.tick_params(axis='x', labelsize=19)
    # 仅保留第一个子图的 y 轴标签，其余隐藏标签文本但保留网格
    for axp in [ax_ssp245, ax_ssp370, ax_ssp585]:
        axp.tick_params(axis='y', labelleft=False)

    # ---------- 角标（a-j） ----------
    for i, axm in enumerate(map_axes):
        axm.text(0.02, 0.02, chr(ord('a') + i), transform=axm.transAxes,
                 fontsize=24, fontweight='bold', va='bottom', ha='left',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    ax_srcsink.text(0.02, 0.98, 'e', transform=ax_srcsink.transAxes,
                    fontsize=24, fontweight='bold', va='top', ha='left',
                    # bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2' )
                    )
    ax_cum.text(0.02, 0.98, 'f', transform=ax_cum.transAxes,
                fontsize=24, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    for j, axp in enumerate([ax_ssp1, ax_ssp245, ax_ssp370, ax_ssp585]):
        axp.text(0.02, 0.98, chr(ord('g') + j), transform=axp.transAxes,
                 fontsize=24, fontweight='bold', va='top', ha='left',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # ---------- 保存 ----------
    outdir = os.path.dirname(output_png)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(output_png, dpi=500, bbox_inches='tight')
    print(f"[OK] 已保存至：{output_png}")
    print(f"总用时 {time.time() - start_time:.2f}s")
    # plt.show()


if __name__ == "__main__":
    main()
