# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import re
# from matplotlib.patches import Patch
#
# from matplotlib.ticker import ScalarFormatter
# from matplotlib.ticker import FuncFormatter
# import colorcet
# import cmasher
#
#
#
# # ================= 全局设置 =================
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.labelpad'] = 6
# plt.rcParams['xtick.major.pad'] = 2
# plt.rcParams['ytick.major.pad'] = 2
#
# # ================= 常量定义 =================
# COUNTRIES = ["Global", "China", "Bolivia", "Russia", "America",
#              "Canada", "Australia", "Congo", "Brazil"]
# SCENARIOS = ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
#
# FILE_TEMPLATES = {
#     "SSP1-2.6": {
#         "Global": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62020_2100\SSP126_global_15to6.csv",
#         "China": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62020_2100\SSP126_Peoples_Republic_of_China_15to6.csv",
#         "Bolivia": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62020_2100\SSP126_Republic_of_Bolivia_15to6.csv",
#         "Russia": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62020_2100\SSP126_Russian_Federation_15to6.csv",
#         "America": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62020_2100\SSP126_United_States_Of_America_15to6.csv",
#         "Canada": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62020_2100\SSP126_Canada_15to6.csv",
#         "Australia": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62020_2100\SSP126_Commonwealth_of_Australia_15to6.csv",
#         "Congo": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62020_2100\SSP126_Democratic_Republic_of_Congo_15to6.csv",
#         "Brazil": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62020_2100\SSP126_Federative_Republic_of_Brazil_15to6.csv"
#     },
#     "SSP2-4.5": {
#         "Global": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62020_2100\SSP245_global_15to6.csv",
#         "China": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62020_2100\SSP245_Peoples_Republic_of_China_15to6.csv",
#         "Bolivia": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62020_2100\SSP245_Republic_of_Bolivia_15to6.csv",
#         "Russia": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62020_2100\SSP245_Russian_Federation_15to6.csv",
#         "America": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62020_2100\SSP245_United_States_Of_America_15to6.csv",
#         "Canada": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62020_2100\SSP245_Canada_15to6.csv",
#         "Australia": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62020_2100\SSP245_Commonwealth_of_Australia_15to6.csv",
#         "Congo": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62020_2100\SSP245_Democratic_Republic_of_Congo_15to6.csv",
#         "Brazil": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62020_2100\SSP245_Federative_Republic_of_Brazil_15to6.csv"
#     },
#     "SSP3-7.0": {
#         "Global": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62020_2100\SSP370_global_15to6.csv",
#         "China": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62020_2100\SSP370_Peoples_Republic_of_China_15to6.csv",
#         "Bolivia": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62020_2100\SSP370_Republic_of_Bolivia_15to6.csv",
#         "Russia": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62020_2100\SSP370_Russian_Federation_15to6.csv",
#         "America": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62020_2100\SSP370_United_States_Of_America_15to6.csv",
#         "Canada": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62020_2100\SSP370_Canada_15to6.csv",
#         "Australia": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62020_2100\SSP370_Commonwealth_of_Australia_15to6.csv",
#         "Congo": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62020_2100\SSP370_Democratic_Republic_of_Congo_15to6.csv",
#         "Brazil": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62020_2100\SSP370_Federative_Republic_of_Brazil_15to6.csv"
#     },
#     "SSP5-8.5": {
#         "Global": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62020_2100\SSP585_global_15to6.csv",
#         "China": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62020_2100\SSP585_Peoples_Republic_of_China_15to6.csv",
#         "Bolivia": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62020_2100\SSP585_Republic_of_Bolivia_15to6.csv",
#         "Russia": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62020_2100\SSP585_Russian_Federation_15to6.csv",
#         "America": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62020_2100\SSP585_United_States_Of_America_15to6.csv",
#         "Canada": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62020_2100\SSP585_Canada_15to6.csv",
#         "Australia": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62020_2100\SSP585_Commonwealth_of_Australia_15to6.csv",
#         "Congo": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62020_2100\SSP585_Democratic_Republic_of_Congo_15to6.csv",
#         "Brazil": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62020_2100\SSP585_Federative_Republic_of_Brazil_15to6.csv"
#     }
# }
# # 读取并合并数据
# all_data = []
# for scenario in SCENARIOS:
#     for country in COUNTRIES:
#         df = pd.read_csv(FILE_TEMPLATES[scenario][country])
#         df['Scenario'] = scenario
#         df['Country'] = country
#         all_data.append(df)
#
# full_df = pd.concat(all_data)
#
# # 处理正负值
# full_df['Positive'] = full_df['Net'].where(full_df['Net'] > 0, 0)
# full_df['Negative'] = full_df['Net'].where(full_df['Net'] < 0, 0).abs()
#
# # 获取所有转换类型
# all_transitions = sorted(full_df['merged_transition'].unique())
#
# # 创建多级索引
# multi_index = pd.MultiIndex.from_product(
#     [SCENARIOS, all_transitions],
#     names=['Scenario', 'merged_transition']
# )
#
# # ================= 科学计数法设置 =================
# max_emission = max(full_df['Positive'].max(), full_df['Negative'].max())
# common_exponent = int(np.floor(np.log10(max_emission))) if max_emission > 0 else 0
# common_factor = 10 ** (common_exponent-1)
#
# # ================= 可视化设置 =================
# fig = plt.figure(figsize=(22, 12))
# gs = fig.add_gridspec(3, 3, wspace=0.15, hspace=0.1, left=0.1, right=0.82)
#
# # 创建颜色映射
# cmap = plt.colormaps['tab20b'].resampled(len(all_transitions))
# color_dict = {trans: cmap(i) for i, trans in enumerate(all_transitions)}
# # cividis是Nature官方推荐的色盲友好色阶
# # cmap = plt.colormaps['cividis'].resampled(len(all_transitions))
# # color_dict = {trans: cmap(i) for i, trans in enumerate(all_transitions)}
#
#
# # 统一格式化函数
# def unified_formatter(x, pos):
#     return f"{x / common_factor:.0f}" if abs(x) >= common_factor else f"{x / common_factor:.0f}"
#
#
# # ================= 主绘图循环 =================
# for idx, country in enumerate(COUNTRIES):
#     row, col = divmod(idx, 3)
#     ax = fig.add_subplot(gs[row, col])
#
#     # 数据处理
#     country_df = full_df[full_df['Country'] == country]
#     pivot_df = (
#         country_df.groupby(['Scenario', 'merged_transition'])
#         .agg({'Positive': 'sum', 'Negative': 'sum'})
#         .reindex(multi_index, fill_value=0)
#         .unstack(level='Scenario')
#     )
#
#     # 绘图参数
#     scenarios_order = SCENARIOS
#     x_pos = np.arange(len(scenarios_order))
#     bar_width = 0.8
#
#     # 堆叠基准
#     stack_base_neg = np.zeros(len(scenarios_order))
#     stack_base_pos = np.zeros(len(scenarios_order))
#
#     # 绘制堆叠柱状图
#     for trans in all_transitions:
#         neg_values = pivot_df['Negative'].loc[trans].reindex(scenarios_order).fillna(0).values
#         pos_values = pivot_df['Positive'].loc[trans].reindex(scenarios_order).fillna(0).values
#
#         # 负值部分
#         ax.bar(x_pos, -neg_values,
#                bottom=stack_base_neg,
#                width=bar_width,
#                color=color_dict[trans],
#                edgecolor='white',
#                linewidth=0.5,
#                zorder=3)
#
#         # 正值部分
#         ax.bar(x_pos, pos_values,
#                bottom=stack_base_pos,
#                width=bar_width,
#                color=color_dict[trans],
#                edgecolor='white',
#                linewidth=0.5,
#                zorder=3)
#
#         # 更新基准
#         stack_base_neg -= neg_values
#         stack_base_pos += pos_values
#
#     # ================= 坐标轴设置 =================
#     # ================= 添加总净排放边框 =================
#     scenario_totals = country_df.groupby('Scenario')['Net'].sum().reindex(scenarios_order, fill_value=0)
#     plt.scatter(x_pos, scenario_totals,
#                 s=70,  # 自定义圆点大小
#                 c='white',  # 设置圆点颜色为白色
#                 edgecolor='white',  # 设置圆点边缘颜色
#                 linewidth=1.5,  # 设置边缘线宽
#                 zorder=4)  # 设置图层顺序
#
#     # ================= 坐标轴设置 =================
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.axhline(0, color='black', linewidth=1, zorder=3)
#
#     # 国家标签
#     ax.text(-0.1, 0.5, country,
#             transform=ax.transAxes,
#             rotation=90,
#             va='center',
#             ha='center',
#             fontsize=22,
#             weight='bold')
#
#     # Y轴设置
#     ax.yaxis.set_major_formatter(FuncFormatter(unified_formatter))
#     ax.tick_params(axis='y', labelsize=15)
#
#     # X轴设置（仅第一行显示情景）
#     if row == 0:
#         ax.xaxis.set_ticks_position('top')
#         ax.set_xticks(x_pos)
#         # ax.set_xticks([])  # 不显示刻度位置
#         ax.set_xticklabels(scenarios_order, fontsize=20, rotation=0)
#         ax.tick_params(axis='x', which='major', pad=3, length=0)
#     else:
#         ax.set_xticks([])
#
# # ================= 动态计算图例位置 =================
# # 收集所有子图的坐标信息
# subplot_axes = [ax for ax in fig.get_axes() if ax.get_subplotspec() is not None]
# bboxes = [ax.get_position() for ax in subplot_axes]
#
# # 计算所有子图的上下边界
# top = max(bbox.y1 for bbox in bboxes)
# print(top)
# bottom = min(bbox.y0 for bbox in bboxes)
# print(bottom)
#
#
# # ================= 修改后的图例函数 =================
# def create_full_height_legend(fig, all_transitions, color_dict, vpos):
#     """创建与子图区域等高的图例"""
#     # 图例参数设置
#     legend_left = 0.83  # 图例左侧位置
#     legend_width = 0.18  # 图例宽度
#     legend_vpad = 0.015  # 垂直边距补偿
#
#     # 创建图例坐标系
#     legend_ax = fig.add_axes([
#         legend_left,
#         vpos['bottom'] - legend_vpad,
#         legend_width,
#         vpos['top'] - vpos['bottom'] + 2 * legend_vpad
#     ])
#     legend_ax.axis('off')
#
#     # 生成图例元素
#     legend_elements = [
#         Patch(facecolor=color_dict[trans],
#               edgecolor='white',
#               linewidth=0.5,
#               label=trans.replace("->", " → "))
#         for trans in all_transitions
#     ]
#     # 计算图例项数量和目标高度
#     num_items = len(legend_elements)
#     target_height =20  # 目标高度（相对坐标）
#
#     # 动态调整参数
#     fontsize = 18
#     labelspacing = 1 * (target_height / num_items)
#     borderaxespad = 1 * (target_height / num_items)
#
#     # 创建自适应图例
#     legend = legend_ax.legend(
#         handles=legend_elements,
#
#         fontsize=fontsize,
#         labelspacing=labelspacing,
#         borderaxespad=borderaxespad,
#         loc='upper center',
#         frameon=False,
#
#         ncol=1,
#
#         handletextpad=0.8,
#         bbox_to_anchor=(0.4, 1),  # 上方微调
#         markerscale=10
#     )
#
#
#
#     return legend
#
#
# # ================= 调用修改后的图例函数 =================
# legend = create_full_height_legend(fig, all_transitions, color_dict,
#                                    {'top': top, 'bottom': bottom})
#
# # ================= 保存输出 =================
# fig.text(0.06, 0.5, 'Carbon emissions  (Gt C)', rotation=90, va='center', ha='center', fontsize=28, weight='bold')
# plt.savefig(r'F:\Data\paper\paper1\pic\Globalemissionscomposement7.png', dpi=400, bbox_inches='tight')
# # plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import os
from matplotlib.ticker import FuncFormatter, MultipleLocator

# ================= 全局设置 =================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['xtick.major.pad'] = 2
plt.rcParams['ytick.major.pad'] = 2

# ================= 常量定义 =================
COUNTRIES = ["Global", "China", "Bolivia", "Russia", "America",
             "Canada", "Australia", "DRC", "Brazil", "Indonesia", "CAR"]
SCENARIOS = ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]

FILE_TEMPLATES = {
    "SSP1-2.6": {
        "Global": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_global_15to6.csv",
        "China": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_Peoples_Republic_of_China_15to6.csv",
        "Bolivia": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_Republic_of_Bolivia_15to6.csv",
        "Russia": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_Russian_Federation_15to6.csv",
        "America": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_United_States_Of_America_15to6.csv",
        "Canada": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_Canada_15to6.csv",
        "Australia": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_Commonwealth_of_Australia_15to6.csv",
        "DRC": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_Democratic_Republic_of_Congo_15to6.csv",
        "Brazil": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_Federative_Republic_of_Brazil_15to6.csv",
        "Indonesia": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_Republic_of_Indonesia_15to6.csv",
        "CAR": r"F:\Data\Landuse\PFT_5KM\SSP126_Resampled\New_Result\trans15to62015_2100tencountry\SSP126_Central_African_Republic_15to6.csv"
    },
    "SSP2-4.5": {
        "Global": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_global_15to6.csv",
        "China": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_Peoples_Republic_of_China_15to6.csv",
        "Bolivia": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_Republic_of_Bolivia_15to6.csv",
        "Russia": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_Russian_Federation_15to6.csv",
        "America": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_United_States_Of_America_15to6.csv",
        "Canada": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_Canada_15to6.csv",
        "Australia": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_Commonwealth_of_Australia_15to6.csv",
        "DRC": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_Democratic_Republic_of_Congo_15to6.csv",
        "Brazil": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_Federative_Republic_of_Brazil_15to6.csv",
        "Indonesia": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_Republic_of_Indonesia_15to6.csv",
        "CAR": r"F:\Data\Landuse\PFT_5KM\SSP245_Resampled\New_Result\trans15to62015_2100tencountry\SSP245_Central_African_Republic_15to6.csv"
    },
    "SSP3-7.0": {
        "Global": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_global_15to6.csv",
        "China": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_Peoples_Republic_of_China_15to6.csv",
        "Bolivia": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_Republic_of_Bolivia_15to6.csv",
        "Russia": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_Russian_Federation_15to6.csv",
        "America": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_United_States_Of_America_15to6.csv",
        "Canada": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_Canada_15to6.csv",
        "Australia": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_Commonwealth_of_Australia_15to6.csv",
        "DRC": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_Democratic_Republic_of_Congo_15to6.csv",
        "Brazil": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_Federative_Republic_of_Brazil_15to6.csv",
        "Indonesia": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_Republic_of_Indonesia_15to6.csv",
        "CAR": r"F:\Data\Landuse\PFT_5KM\SSP370_Resampled\New_Result\trans15to62015_2100tencountry\SSP370_Central_African_Republic_15to6.csv"
    },
    "SSP5-8.5": {
        "Global": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_global_15to6.csv",
        "China": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_Peoples_Republic_of_China_15to6.csv",
        "Bolivia": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_Republic_of_Bolivia_15to6.csv",
        "Russia": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_Russian_Federation_15to6.csv",
        "America": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_United_States_Of_America_15to6.csv",
        "Canada": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_Canada_15to6.csv",
        "Australia": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_Commonwealth_of_Australia_15to6.csv",
        "DRC": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_Democratic_Republic_of_Congo_15to6.csv",
        "Brazil": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_Federative_Republic_of_Brazil_15to6.csv",
        "Indonesia": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_Republic_of_Indonesia_15to6.csv",
        "CAR": r"F:\Data\Landuse\PFT_5KM\SSP585_Resampled\New_Result\trans15to62015_2100tencountry\SSP585_Central_African_Republic_15to6.csv"
    }
}

# 读取并合并数据
all_data = []
for scenario in SCENARIOS:
    for country in COUNTRIES:
        df = pd.read_csv(FILE_TEMPLATES[scenario][country])
        df['Scenario'] = scenario
        df['Country'] = country
        all_data.append(df)

full_df = pd.concat(all_data)

# 处理正负值
full_df['Positive'] = full_df['Net'].where(full_df['Net'] > 0, 0)
full_df['Negative'] = full_df['Net'].where(full_df['Net'] < 0, 0).abs()

# 获取所有转换类型
all_transitions = sorted(full_df['merged_transition'].unique())

# 创建多级索引
multi_index = pd.MultiIndex.from_product(
    [SCENARIOS, all_transitions],
    names=['Scenario', 'merged_transition']
)

# ================= 科学计数法设置 =================
max_emission = max(full_df['Positive'].max(), full_df['Negative'].max())
common_exponent = int(np.floor(np.log10(max_emission))) if max_emission > 0 else 0
common_factor = 10 ** (common_exponent-1)

# ================= 可视化设置 =================
fig = plt.figure(figsize=(22, 22))  # 增加高度以容纳4行
gs = fig.add_gridspec(4, 3, wspace=0.15, hspace=0.1, left=0.1, right=0.82)

# 创建颜色映射
cmap = plt.colormaps['tab20b'].resampled(len(all_transitions))
color_dict = {trans: cmap(i) for i, trans in enumerate(all_transitions)}

# 统一格式化函数
def unified_formatter(x, pos):
    return f"{x / common_factor:.0f}" if abs(x) >= common_factor else f"{x / common_factor:.0f}"

# 创建图例句柄
legend_handles = [
    Patch(facecolor=color_dict[trans],
          edgecolor='white',
          linewidth=0.5,
          label=trans.replace("->", " → "))
    for trans in all_transitions
]

# ================= 主绘图循环 =================
for idx, country in enumerate(COUNTRIES):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])

    # 数据处理
    country_df = full_df[full_df['Country'] == country]
    pivot_df = (
        country_df.groupby(['Scenario', 'merged_transition'])
        .agg({'Positive': 'sum', 'Negative': 'sum'})
        .reindex(multi_index, fill_value=0)
        .unstack(level='Scenario')
    )

    # 绘图参数
    scenarios_order = SCENARIOS
    x_pos = np.arange(len(scenarios_order))
    bar_width = 0.8

    # 堆叠基准
    stack_base_neg = np.zeros(len(scenarios_order))
    stack_base_pos = np.zeros(len(scenarios_order))

    # 绘制堆叠柱状图
    for trans in all_transitions:
        neg_values = pivot_df['Negative'].loc[trans].reindex(scenarios_order).fillna(0).values
        pos_values = pivot_df['Positive'].loc[trans].reindex(scenarios_order).fillna(0).values

        # 负值部分
        ax.bar(x_pos, -neg_values,
               bottom=stack_base_neg,
               width=bar_width,
               color=color_dict[trans],
               edgecolor='white',
               linewidth=0.5,
               zorder=3)

        # 正值部分
        ax.bar(x_pos, pos_values,
               bottom=stack_base_pos,
               width=bar_width,
               color=color_dict[trans],
               edgecolor='white',
               linewidth=0.5,
               zorder=3)

        # 更新基准
        stack_base_neg -= neg_values
        stack_base_pos += pos_values

    # ================= 坐标轴设置 =================
    # ================= 添加总净排放边框 =================
    scenario_totals = country_df.groupby('Scenario')['Net'].sum().reindex(scenarios_order, fill_value=0)
    ax.scatter(x_pos, scenario_totals,
               s=70,
               c='white',
               edgecolor='white',
               linewidth=1.5,
               zorder=4)

    # ================= 坐标轴设置 =================
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axhline(0, color='black', linewidth=1, zorder=3)

    # 国家标签
    ax.text(-0.1, 0.5, country,
            transform=ax.transAxes,
            rotation=90,
            va='center',
            ha='center',
            fontsize=22,
            weight='bold')

    # Y轴设置 - 特别处理中非国家
    ax.yaxis.set_major_formatter(FuncFormatter(unified_formatter))
    ax.tick_params(axis='y', labelsize=15)

    if country in ("Bolivia", "Central African"):
        ymin, ymax = ax.get_ylim()

        unit_scale = 10 ** 7  # 1 Gt 对应的原始数据单位

        min_val = np.floor(ymin / unit_scale) * unit_scale
        max_val = np.ceil(ymax / unit_scale) * unit_scale

        ax.set_ylim(min_val, max_val)
        ax.yaxis.set_major_locator(MultipleLocator(unit_scale))

        # 格式化标签显示为 Gt 单位（除以10^7）
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / unit_scale:.0f}"))

    # X轴设置（仅第一行显示情景）
    if row == 0:
        ax.xaxis.set_ticks_position('top')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios_order, fontsize=20, rotation=0)
        ax.tick_params(axis='x', which='major', pad=3, length=0)
    else:
        ax.set_xticks([])

# ================= 右下角图例子图 =================
legend_ax = fig.add_subplot(gs[3, 2])  # 第4行第3列（右下角）
legend_ax.axis('off')  # 关闭坐标轴

# 创建图例
legend = legend_ax.legend(
    handles=legend_handles,
    loc='center',
    frameon=False,
    fontsize=18,
    ncol=2,  # 单列显示
    handletextpad=1,
    columnspacing=0.5,
    bbox_to_anchor=(0.5, 0.5)  # 居中显示
)

# ================= 保存输出 =================
fig.text(0.06, 0.5, 'Carbon emissions  (Gt C)', rotation=90, va='center', ha='center', fontsize=28, weight='bold')
plt.savefig(r'F:\Data\paper\paper1\pic\Globalemissionscomposement9.png', dpi=400, bbox_inches='tight')

# plt.show()
