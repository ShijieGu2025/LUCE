
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FuncFormatter

# ================= 全局样式 =================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['xtick.major.pad'] = 2
plt.rcParams['ytick.major.pad'] = 2

# ================= 常量定义 =================
COUNTRIES = [
    "Global", "China", "Bolivia", "Russia", "United States",
    "Nigeria", "Australia", "Canada", "DRC", "Brazil", "Indonesia", "CAR"
]
SCENARIOS = ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]

FILE_TEMPLATES = {
    "SSP1-2.6": {
        "Global": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_global_15to6.csv",
        "China": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_Peoples_Republic_of_China_15to6.csv",
        "Bolivia": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_Republic_of_Bolivia_15to6.csv",
        "Russia": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_Russian_Federation_15to6.csv",
        "United States": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_United_States_Of_America_15to6.csv",
        "Nigeria": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_Federal_Republic_of_Nigeria_15to6.csv",
        "Australia": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_Commonwealth_of_Australia_15to6.csv",
        "DRC": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_Democratic_Republic_of_Congo_15to6.csv",
        "Brazil": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_Federative_Republic_of_Brazil_15to6.csv",
        "Indonesia": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_Republic_of_Indonesia_15to6.csv",
        "CAR": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_Central_African_Republic_15to6.csv",
        "Canada": r"G:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to52015_2100country11\SSP126_Canada_15to6.csv"
    },
    "SSP2-4.5": {
        "Global": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_global_15to6.csv",
        "China": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_Peoples_Republic_of_China_15to6.csv",
        "Bolivia": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_Republic_of_Bolivia_15to6.csv",
        "Russia": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_Russian_Federation_15to6.csv",
        "United States": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_United_States_Of_America_15to6.csv",
        "Nigeria": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_Federal_Republic_of_Nigeria_15to6.csv",
        "Australia": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_Commonwealth_of_Australia_15to6.csv",
        "DRC": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_Democratic_Republic_of_Congo_15to6.csv",
        "Brazil": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_Federative_Republic_of_Brazil_15to6.csv",
        "Indonesia": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_Republic_of_Indonesia_15to6.csv",
        "CAR": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_Central_African_Republic_15to6.csv",
        "Canada": r"G:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to52015_2100country11\SSP245_Canada_15to6.csv"
    },
    "SSP3-7.0": {
        "Global": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_global_15to6.csv",
        "China": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_Peoples_Republic_of_China_15to6.csv",
        "Bolivia": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_Republic_of_Bolivia_15to6.csv",
        "Russia": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_Russian_Federation_15to6.csv",
        "United States": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_United_States_Of_America_15to6.csv",
        "Nigeria": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_Federal_Republic_of_Nigeria_15to6.csv",
        "Australia": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_Commonwealth_of_Australia_15to6.csv",
        "DRC": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_Democratic_Republic_of_Congo_15to6.csv",
        "Brazil": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_Federative_Republic_of_Brazil_15to6.csv",
        "Indonesia": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_Republic_of_Indonesia_15to6.csv",
        "CAR": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_Central_African_Republic_15to6.csv",
        "Canada": r"G:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to52015_2100country11\SSP370_Canada_15to6.csv"
    },
    "SSP5-8.5": {
        "Global": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_global_15to6.csv",
        "China": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_Peoples_Republic_of_China_15to6.csv",
        "Bolivia": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_Republic_of_Bolivia_15to6.csv",
        "Russia": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_Russian_Federation_15to6.csv",
        "United States": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_United_States_Of_America_15to6.csv",
        "Nigeria": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_Federal_Republic_of_Nigeria_15to6.csv",
        "Australia": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_Commonwealth_of_Australia_15to6.csv",
        "DRC": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_Democratic_Republic_of_Congo_15to6.csv",
        "Brazil": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_Federative_Republic_of_Brazil_15to6.csv",
        "Indonesia": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_Republic_of_Indonesia_15to6.csv",
        "CAR": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_Central_African_Republic_15to6.csv",
        "Canada": r"G:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to52015_2100country11\SSP585_Canada_15to6.csv"
    }
}

# 偏好显示顺序（若文件无该转换则忽略）
PREFERRED_ORDER = [
    'Barren->Cropland', 'Barren->Forest', 'Barren->Grassland',
    'Cropland->Barren', 'Cropland->Forest', 'Cropland->Grassland',
    'Forest->Barren', 'Forest->Cropland', 'Forest->Grassland',
    'Grassland->Barren', 'Grassland->Cropland', 'Grassland->Forest',
    'Others'
]

# ================= 用户自定义配色与图例顺序/标签 =================


# 数字编码颜色（你提供）
USER_COLOR_BY_CODE = {
    21: '#A6D96A', 12: '#A6BDDB', 14: '#FEE08B', 13: '#C2A5CF',
    31: '#1A9641', 32: '#67A9CF', 34: '#D6604D', 23: '#9970AB',
    41: '#006837', 42: '#1C6DAB', 43: '#762A83', 24: '#FDAE61'
}
import string
PANEL_LABELS = list(string.ascii_lowercase)
# === 选一个与现有颜色不重复的 'Others' 颜色 ===
EXISTING_COLORS = {c.lower() for c in USER_COLOR_BY_CODE.values()}
OTHERS_CANDIDATES = ['#FF1493', '#00FFFF', '#000000']  # 深粉 > 青色 > 黑色（最后兜底）
OTHERS_COLOR = next((c for c in OTHERS_CANDIDATES if c.lower() not in EXISTING_COLORS), '#FF1493')
# 图例顺序（按目标地类分组）
LEGEND_CODES = [21, 31, 41, 12, 32, 42, 14, 24, 34, 13, 23, 43]
# 编码→标签
LABEL_BY_CODE = {
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

# 转换字符串→两位编码：1=Forest, 2=Grassland, 3=Barren, 4=Cropland
_CLASS_CODE = {'Forest': 1, 'Grassland': 2, 'Barren': 3, 'Cropland': 4}
def trans_to_code(trans: str):
    """'Forest->Cropland' / 'Forest → Cropland' -> 14；不符合规则则返回 None"""
    if not isinstance(trans, str):
        return None
    t = trans.replace('→', '->')
    if '->' not in t:
        return None
    src, dst = [s.strip() for s in t.split('->')]
    if src not in _CLASS_CODE or dst not in _CLASS_CODE:
        return None
    return _CLASS_CODE[src] * 10 + _CLASS_CODE[dst]

# ================= 读取并合并数据 =================
all_data = []
for scenario in SCENARIOS:
    for country in COUNTRIES:
        path = FILE_TEMPLATES[scenario][country]
        if not os.path.exists(path):
            print(f"[WARN] 缺少文件：{path}")
            continue
        df = pd.read_csv(path)
        df['Scenario'] = scenario
        df['Country'] = country

        # 统一正负：Net>0 为 Positive；Net<0 为 Negative（取绝对值）
        df['Positive'] = df['Net'].where(df['Net'] > 0, 0.0)
        df['Negative'] = df['Net'].where(df['Net'] < 0, 0.0).abs()

        all_data.append(df)

if not all_data:
    raise RuntimeError("没有成功读到任何 CSV，请检查路径与文件名。")

full_df = pd.concat(all_data, ignore_index=True)

# 所有转换类型（按偏好顺序 + 其余）
seen = set(full_df['merged_transition'].unique())
all_transitions = [t for t in PREFERRED_ORDER if t in seen] + [t for t in sorted(seen) if t not in PREFERRED_ORDER]

# ================= 颜色映射（按用户色卡） =================
# ================= 颜色映射（按用户色卡） =================
DEFAULT_COLOR = OTHERS_COLOR  # 未定义转换/“Others”统一用独特颜色
color_dict = {}
for trans in all_transitions:
    if trans == 'Others':
        color_dict[trans] = OTHERS_COLOR
        continue
    code = trans_to_code(trans)
    color_dict[trans] = USER_COLOR_BY_CODE.get(code, DEFAULT_COLOR)


# ================= y 轴与单位设置 =================
# 若数据为 Gt C -> GT=1.0；为 tC -> 1e9；为 MtC -> 1e3；为 ktC -> 1e6
GT =  1e7

def _nice_step(raw_step):
    """把原始步长取到 1/2/5×10^k 的漂亮值"""
    if raw_step <= 0:
        return 1.0
    exp = math.floor(math.log10(raw_step))
    base = raw_step / (10 ** exp)
    for m in (1, 2, 5, 10):
        if base <= m:
            return m * (10 ** exp)
    return 10 ** (exp + 1)

def apply_axis_in_gt(ax, up_max_raw, down_max_raw, target_ticks=6, pad=1.10):
    """根据上下堆叠极值设置 y 轴范围与主刻度（显示单位：Gt C）"""
    up = float(up_max_raw)
    dn = float(down_max_raw)
    yhi = up * pad
    ylo = -dn * pad
    if yhi == 0 and ylo == 0:
        yhi, ylo = 1.0, -1.0

    ax.set_ylim(ylo, yhi)

    # 选合适刻度步长（在 GtC 单位下）
    yrange_gt = (yhi - ylo) / GT
    raw_step_gt = yrange_gt / max(3, target_ticks)
    step_gt = _nice_step(raw_step_gt)
    ax.yaxis.set_major_locator(MultipleLocator(step_gt * GT))

    def fmt(y, _):
        val = y / GT
        return f"{val:.1f}" if step_gt < 1 else f"{val:.0f}"
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))

# ================= 可视化布局（5x3，最后一行图例） =================
fig = plt.figure(figsize=(22, 22))
gs = fig.add_gridspec(
    5, 3, wspace=0.15, hspace=0.12, left=0.1, right=0.95, bottom=0.08, top=0.98,
    height_ratios=[1, 1, 1, 1, 0.4]
)

# ================= 主绘图循环 =================
for idx, country in enumerate(COUNTRIES):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])

    country_df = full_df[full_df['Country'] == country]

    # 透视表：按 (Scenario, transition) 聚合 Positive/Negative
    pivot_df = (
        country_df.groupby(['Scenario', 'merged_transition'])
                  .agg({'Positive': 'sum', 'Negative': 'sum'})
                  .reindex(pd.MultiIndex.from_product([SCENARIOS, all_transitions],
                                                      names=['Scenario', 'merged_transition']),
                           fill_value=0)
                  .unstack(level='Scenario')
    )

    scenarios_order = SCENARIOS
    x_pos = np.arange(len(scenarios_order))
    bar_width = 0.8

    # 堆叠基准
    stack_base_neg = np.zeros(len(scenarios_order))  # 底部（负向）
    stack_base_pos = np.zeros(len(scenarios_order))  # 顶部（正向）

    # —— 逐个转换堆叠 —— #
    for trans in all_transitions:
        neg_values = pivot_df['Negative'].loc[trans].reindex(scenarios_order).fillna(0).values
        pos_values = pivot_df['Positive'].loc[trans].reindex(scenarios_order).fillna(0).values

        # 负值（向下绘制）
        ax.bar(x_pos, -neg_values, bottom=stack_base_neg,
               width=bar_width, color=color_dict[trans],
               edgecolor='white', linewidth=0.5, zorder=3)

        # 正值（向上绘制）
        ax.bar(x_pos, pos_values, bottom=stack_base_pos,
               width=bar_width, color=color_dict[trans],
               edgecolor='white', linewidth=0.5, zorder=3)

        stack_base_neg -= neg_values
        stack_base_pos += pos_values

    # —— 子图 y 轴自适应（单位 Gt C） —— #
    pos_sum_by_scen = pivot_df['Positive'].sum(axis=0).reindex(scenarios_order).fillna(0).values
    neg_sum_by_scen = pivot_df['Negative'].sum(axis=0).reindex(scenarios_order).fillna(0).values
    up_max = float(np.max(pos_sum_by_scen))
    down_max = float(np.max(neg_sum_by_scen))
    apply_axis_in_gt(ax, up_max, down_max)

    # 仅对 Bolivia / CAR 强制用 1 的整数刻度（单位=Gt C）
    if country in ("Bolivia", "CAR"):
        ax.yaxis.set_major_locator(MultipleLocator(1 * GT))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(round(y / GT))}"))
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(np.floor(ylo / GT) * GT, np.ceil(yhi / GT) * GT)

    # 总净排放（白点）
    scenario_totals = country_df.groupby('Scenario')['Net'].sum().reindex(scenarios_order, fill_value=0)
    ax.scatter(x_pos, scenario_totals.values, s=70, c='white', edgecolor='white', linewidth=1.5, zorder=4)
    ax.tick_params(axis='y', labelsize=23)

    # 坐标轴与样式
    ax.axhline(0, color='black', linewidth=1, zorder=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 左侧国家标签
    ax.text(-0.10, 0.5, country, transform=ax.transAxes, rotation=90,
            va='center', ha='center', fontsize=24, weight='bold')
    # ================= 添加子图字母标注 =================
    # 只在Global图上添加字母'a'
    # ================= 添加子图字母标注（放在 y 轴外侧） =================
    ax.text(0.02, 0.02, PANEL_LABELS[idx], transform=ax.transAxes,
            fontsize=30, fontweight='bold', fontname='Times New Roman',
            va='top', ha='left', clip_on=False)

    # 顶部情景标签（仅第一行）
    if row == 0:
        ax.xaxis.set_ticks_position('top')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios_order, fontsize=24, rotation=0)
        ax.tick_params(axis='x', which='major', pad=3, length=0)
    else:
        ax.set_xticks([])

# ================= 最底部图例（按 LEGEND_CODES 顺序，三行） =================
legend_ax = fig.add_subplot(gs[4, :])   # 占据最底下一整行
legend_ax.axis('off')

# 数据中出现的编码
codes_present = set()
for t in seen:
    c = trans_to_code(t)
    if c is not None:
        codes_present.add(c)

legend_handles = [
    Patch(facecolor=USER_COLOR_BY_CODE[c], edgecolor='white', linewidth=0.5, label=LABEL_BY_CODE[c])
    for c in LEGEND_CODES if c in codes_present
]
# 若有 Others（未匹配到编码的类别），则追加
if 'Others' in all_transitions:
    legend_handles.append(Patch(facecolor=OTHERS_COLOR, edgecolor='white', linewidth=0.5, label='Others'))


n_items = len(legend_handles)
n_rows = 3
ncol = math.ceil(n_items / n_rows)

legend_ax.legend(
    handles=legend_handles,
    ncol=ncol,
    frameon=False,
    fontsize=23,
    handletextpad=1.0,
    columnspacing=1.0,
    labelspacing=0.6,
    loc='upper left',
    bbox_to_anchor=(0, 0, 1, 1),  # 占满整行
    mode='expand',
    borderaxespad=0.0
)

# ================= 保存 =================
fig.text(0.06, 0.6, 'Cumulative carbon fluxes  (Gt C)', rotation=90,
         va='center', ha='center', fontsize=26, weight='bold')

out_path = r'G:\Data\paper\paper1\pic\Globalemissionscomposement15.png'
plt.savefig(out_path, dpi=400, bbox_inches='tight')
print(f"Saved to: {out_path}")
# plt.show()
