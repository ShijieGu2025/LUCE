
# -*- coding: utf-8 -*-
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
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from osgeo import gdal, osr, ogr
import os, time, warnings
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning, module="osgeo")
# 如仅保存图片且不需要弹窗，可解开下一行强制使用无交互后端：
# mpl.use("Agg")

# ================= 全局样式 =================
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 18
rcParams['mathtext.default'] = 'regular'

# ================= 地图数据路径（四个情景） =================
raster_paths = [
    r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\SSP126totalcombined_carbon_emission_2015_2100.tif",
    r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\SSP245totalcombined_carbon_emission_2015_2100.tif",
    r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\SSP370totalcombined_carbon_emission_2015_2100.tif",
    r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\SSP585totalcombined_carbon_emission_2015_2100.tif"
]
vector_path = r"F:\Data\Landuse\Bound\world_dissolve_WGS84.shp"

# ================= 统计用 Excel（阶段净变化 + 累积图） =================
SCENARIO_PATHS = {
    "SSP1-2.6": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\out\out2\SSP126_combined_carbon_storage_and_emission_summary.xlsx",
    "SSP2-4.5": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\out\out3\SSP245_combined_carbon_storage_and_emission_summary.xlsx",
    "SSP3-7.0": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\out\out4\SSP370_combined_carbon_storage_and_emission_summary.xlsx",
    "SSP5-8.5": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\out\out3\SSP585_combined_carbon_storage_and_emission_summary.xlsx"
}
SCENARIOS = list(SCENARIO_PATHS.keys())
PERIODS = [(2015, 2035), (2035, 2050), (2050, 2070), (2070, 2090), (2090, 2100)]

# ================= "全球 CSV（15to6）"用于总源/总汇 =================
GLOBAL_CSV = {
    "SSP1-2.6": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_global_15to6.csv",
    "SSP2-4.5": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_global_15to6.csv",
    "SSP3-7.0": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_global_15to6.csv",
    "SSP5-8.5": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_global_15to6.csv",
}

# CSV 换算到 Gt：若 CSV 已是 Gt -> 1；若是 t -> 1e9；若是"10^7 t" -> 1e7（与你之前一致）
CSV_UNIT_SCALE = 1e7

# 输出路径
output_png = r"F:\Data\paper\paper1\pic\Combined_Carbon_Visualization_final3.png"

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
    means = [s['mean'] for s in stats]
    lower_errs = [s['lower_err'] for s in stats]
    upper_errs = [s['upper_err'] for s in stats]
    yerr = [lower_errs, upper_errs]

    ax.bar(range(5), means, yerr=yerr, color=COLOR_PALETTE,
           error_kw={'ecolor': '#2f2f2f', 'elinewidth': 2})
    ax.axhline(0, color='#444444', lw=1.5, zorder=0)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{s}-{e}" for s, e in PERIODS], rotation=15, fontsize=14)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    if show_ylabel:
        ax.set_ylabel('Annual Net Emissions (Gt/yr)', fontsize=14)
    if show_title:
        ax.set_title(scenario_name, fontsize=16, pad=8)


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

    # 读取四个 TIF，统一色标范围
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
            gt = wds.GetGeoTransform()
            cols, rows = wds.RasterXSize, wds.RasterYSize
        else:
            band = ds.GetRasterBand(1)
            arr = band.ReadAsArray()
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
                                                linewidth=0.4, alpha=0.7, zorder=3))
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
    cbar.set_label('Carbon Emission (×100 t C per pixel)', fontsize=16, rotation=270, va='bottom', labelpad=18)
    cax.tick_params(labelsize=14)

    print("地图完成。")

    # ---------- 第三行：左 = 总源/总汇；右 = 哑铃对比 ----------
    # （建议略微增大左边距，避免 y 轴情景名被裁剪）
    gs = GridSpec(4, 4, figure=fig,
                  height_ratios=[1.0, 1.0, 1.0,1.0],
                  width_ratios=[1, 1, 1, 1],
                  hspace=0.15, wspace=0.12,
                  top=0.9, bottom=0.06, left=0.06, right=1)  # <-- left 从 0.06 调到 0.08

    ax_srcsink = fig.add_subplot(gs[2, 0:2])  # 左
    ax_cum = fig.add_subplot(gs[2, 2:4], sharey=ax_srcsink)  # 右，与左共享 y 轴（严格对齐）

    # ---- 左图：总源/总汇 ----
    df_ss = compute_sources_sinks_from_global_csv()
    y_base = np.arange(len(SCENARIOS)) * 3.0

    XMAX = max(1e-9, df_ss[['Sources', 'Sinks']].to_numpy().max())
    ax_srcsink.set_xlim(0, XMAX * 1.05)
    ax_srcsink.set_ylim(-1, y_base[-1] + 2)
    ax_srcsink.grid(axis='x', linestyle=':', alpha=0.6)

    # 先设置刻度位置（只在左图显示情景名）
    ax_srcsink.set_yticks(y_base)
    ax_srcsink.set_yticklabels(SCENARIOS, fontsize=14)

    ax_srcsink.spines['bottom'].set_visible(False)
    # 用“右对齐 + 小 pad”，避免被左边界裁剪
    # for t in ax_srcsink.get_yticklabels():
    #     t.set_horizontalalignment('right')
    # ax_srcsink.yaxis.set_tick_params(pad=2)  # pad 小一点

    # 左图底部/顶部 x 轴标题（靠左对齐可保留；与是否裁剪无关）
    # ax_srcsink.tick_params(axis='x', which='both', labelbottom=False)
    # try:
    #     ax_srcsink.set_xlabel('Gross absorb (Gt)', fontsize=14, loc='left')
    # except Exception:
    #     ax_srcsink.set_xlabel('Gross absorb (Gt)', fontsize=14)
    # ax_srcsink.xaxis.set_label_coords(0.0, -0.08)
    ax_top = ax_srcsink.secondary_xaxis('top')
    ax_bottom = ax_srcsink.secondary_xaxis('bottom')
    # try:
    #     ax_top.set_xlabel('Gross Sources (Gt)', fontsize=14, loc='left')
    # except Exception:
    #     ax_top.set_xlabel('Gross Sources (Gt)', fontsize=14)
    ax_top.xaxis.set_label_coords(0.0, 1.08)
    ax_top.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # 底部主轴：保留刻度与刻度标签
    ax_srcsink.tick_params(
        axis='x', which='both',
        bottom=True, labelbottom=True,  # 显示底部
        top=False, labeltop=False  # 关闭主轴顶部（不影响次轴对象）
    )
    # 底部 xlabel（如需左对齐）
    # try:
    #     ax_srcsink.set_xlabel('Gross absorb (Gt)', fontsize=14, loc='left')
    # except Exception:
    #     ax_srcsink.set_xlabel('Gross absorb (Gt)', fontsize=14)
    ax_srcsink.xaxis.set_label_coords(0.0, -0.08)

    # 顶部次轴：轴线保留，但隐藏“刻度、刻度标签、轴标题”
    # ax_top.set_xlabel("")  # 不显示顶部轴标题
    # ax_top.tick_params(
    #     axis='x', which='both',
    #     top=True, labeltop=False,  # 轴线在上方保留
    #     bottom=False, labelbottom=False,
    #     length=0  # 刻度长度设为 0 -> 不画刻度
    # )
    # 不显示任何顶部刻度标签
    ax_top.xaxis.set_major_formatter(NullFormatter())
    ax_top.xaxis.set_minor_formatter(NullFormatter())

    # 明确保留顶部次轴的轴线（spine），避免被样式改没
    ax_top.spines['top'].set_visible(True)
    ax_bottom.spines['bottom'].set_visible(True)
    # 同时隐藏主坐标系的顶部边框，避免双重边框叠加（可选）
    ax_srcsink.spines['top'].set_visible(False)

    # 左图画线
    for i, row in df_ss.iterrows():
        scen = row['Scenario']
        col = SCENARIO_COLORS[scen]
        y_upper = y_base[i] + 0.6
        y_lower = y_base[i] - 0.6
        ax_srcsink.hlines(y=y_upper, xmin=0, xmax=row['Sources'], color=col, lw=5, alpha=0.95, linestyle='-')
        ax_srcsink.hlines(y=y_lower, xmin=0, xmax=row['Sinks'], color=col, lw=5, alpha=0.95, linestyle=':')

    ax_srcsink.legend(handles=[
        plt.Line2D([0], [0], color='#444', lw=4, linestyle='-', label='Gross source (Gt)'),
        plt.Line2D([0], [0], color='#444', lw=4, linestyle=':', label='Gross absorb (Gt)'),
    ], loc='upper right', frameon=False, fontsize=12)

    # ---- 右图：哑铃（2015–2100 vs 2025–2100），只隐藏“显示”，不清空共享标签 ----
    cumulative_data = []
    for scenario, path in SCENARIO_PATHS.items():
        df = process_scenario_data(path)
        cum_full = calculate_cumulative(df, 2015, 2100)
        cum_partial = calculate_cumulative(df, 2025, 2100)
        cumulative_data.append((scenario, cum_full, cum_partial))

    # 与 SCENARIOS 保持同序
    cumulative_data = [(s, f, p) for s, f, p in cumulative_data if s in SCENARIOS]
    cumulative_data.sort(key=lambda t: SCENARIOS.index(t[0]))

    # 不再调用 ax_cum.set_yticklabels([]) —— 这会清空共享轴的标签！
    # 仅隐藏右图的“显示”，保留刻度与标签对象在共享轴中
    ax_cum.tick_params(axis='y', labelleft=False, labelright=False, left=False, right=False, length=0)

    xmax = max(max(full, partial) for _, full, partial in cumulative_data)
    ax_cum.set_xlim(0, xmax * 1.05)

    for i, (scenario, full, partial) in enumerate(cumulative_data):
        y_i = y_base[i]
        col = SCENARIO_COLORS.get(scenario, '#444444')
        # 连线 + 两端点
        ax_cum.plot([partial, full], [y_i, y_i], '-', lw=4, color=col, alpha=0.9, zorder=1)
        ax_cum.plot(full, y_i, 'o', ms=8, mfc=col, mec='white', mew=0.8, zorder=2)  # 实心圆：2015–2100
        ax_cum.plot(partial, y_i, 'o', ms=8, mfc='white', mec=col, mew=1.6, zorder=2)  # 空心圆：2025–2100

        # —— 新增：三角符号 + 等号 + 差值
        diff = full - partial
        x_mid = (full + partial) / 2.0
        ax_cum.text(
            x_mid, y_i+0.1, fr"$\triangle$ = {diff:.1f}",
            ha='center', va='bottom', fontsize=14, color=col,
            zorder=3
        )

    ax_cum.grid(axis='x', linestyle=':', alpha=0.6)
    ax_cum.axvline(0, color='#555', lw=1)
    ax_cum.set_xlabel("")  # 外部 xlabel 不用
    ax_cum.tick_params(axis='x', bottom=True, labelbottom=True, top=False, labeltop=False)

    # 图内左上角的标题文本（改为中文）
    ax_cum.text(
        0.71, 0.13, 'Cumulative net emissions (Gt)',
        transform=ax_cum.transAxes, ha='left', va='top', fontsize=16, zorder=10,
        bbox=dict(facecolor='goldenrod', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
    )

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None', mfc='white', mec='#444', mew=1.2, label='2025–2100'),
        Line2D([0], [0], marker='o', linestyle='None', mfc='#444', mec='white', mew=0.8, label='2015–2100'),
    ]
    ax_cum.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=12)

    # —— 关键一步：在两图所有设置完成“之后”，最后再把左图的情景名写回一次（防止被共享轴的操作覆盖）
    ax_srcsink.set_yticks(y_base)
    ax_srcsink.set_yticklabels(SCENARIOS, fontsize=14)
    for t in ax_srcsink.get_yticklabels():
        t.set_horizontalalignment('right')
    ax_srcsink.yaxis.set_tick_params(pad=2)

    # ---------- 第四行：四个情景的阶段净变化柱 ----------
    ax_ssp1   = fig.add_subplot(gs[3, 0])
    ax_ssp245 = fig.add_subplot(gs[3, 1])
    ax_ssp370 = fig.add_subplot(gs[3, 2])
    ax_ssp585 = fig.add_subplot(gs[3, 3])

    plot_period_bars(
        ax_ssp1, process_scenario_data(SCENARIO_PATHS["SSP1-2.6"]), "SSP1-2.6",
        show_title=False, show_ylabel=True
    )
    plot_period_bars(
        ax_ssp245, process_scenario_data(SCENARIO_PATHS["SSP2-4.5"]), "SSP2-4.5",
        show_title=False, show_ylabel=False
    )
    plot_period_bars(
        ax_ssp370, process_scenario_data(SCENARIO_PATHS["SSP3-7.0"]), "SSP3-7.0",
        show_title=False, show_ylabel=False
    )
    plot_period_bars(
        ax_ssp585, process_scenario_data(SCENARIO_PATHS["SSP5-8.5"]), "SSP5-8.5",
        show_title=False, show_ylabel=False
    )

    # ---------- 角标（a-j） ----------
    for i, axm in enumerate(map_axes):
        axm.text(0.02, 0.02, chr(ord('a') + i), transform=axm.transAxes,
                 fontsize=18, fontweight='bold', va='bottom', ha='left',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    ax_srcsink.text(0.02, 0.98, 'e', transform=ax_srcsink.transAxes,
                    fontsize=18, fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    ax_cum.text(0.02, 0.98, 'f', transform=ax_cum.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    for j, axp in enumerate([ax_ssp1, ax_ssp245, ax_ssp370, ax_ssp585]):
        axp.text(0.02, 0.98, chr(ord('g') + j), transform=axp.transAxes,
                 fontsize=18, fontweight='bold', va='top', ha='left',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # ---------- 保存 ----------
    outdir = os.path.dirname(output_png)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(output_png, dpi=700, bbox_inches='tight')
    print(f"[OK] 已保存至：{output_png}")
    print(f"总用时 {time.time() - start_time:.2f}s")
    # plt.show()


if __name__ == "__main__":
    main()
