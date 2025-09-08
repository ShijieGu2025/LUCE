from matplotlib import pyplot as plt
from osgeo import gdal, osr
import pandas as pd
import numpy as np
import math
import rasterio.transform
import xarray as xr
import rioxarray
from rasterio import rio
from rasterio.transform import from_origin
import os
from tqdm import tqdm


# 定义函数 mature_year，将格网值映射为成熟年龄
def mature_year(grid_values):
    mature_year_mapping = {None: 0, 0: 0, 1: 40, 2: 50, 3: 35, 4: 50, 5: 50, 6: 50, 7: 50, 8: 50, 9: 20, 10: 50, 11: 10,
                           12: 30,
                           13: 50, 14: 1, 15: 1}
    mapped_values = np.vectorize(mature_year_mapping.get)(grid_values)
    return mapped_values


# 定义函数 Soil_time，将区域值映射为土壤时间
def Soil_time(region_values):
    region_values_mapping = {1: 50, 2: 25, 3: 50, 4: 50, 5: 50, 6: 25, 7: 25, 8: 50, 9: 200}
    return region_values_mapping.get(region_values, 0)


# 定义函数 soil_carbon_density，将格网值映射为土壤碳密度
def soil_carbon_density(grid_value):
    soil_density_mapping = {0: 0, 1: 98, 2: 71, 3: 98, 4: 134, 5: 93.5, 6: 166, 7: 93.5, 8: 109, 9: 75, 10: 69, 11: 130,
                            12: 95, 13: 38, 14: 90, 15: 58}
    return soil_density_mapping.get(grid_value, 0)  # Return 0 if grid_value is None


# 定义函数 vegetation_carbon_density，将格网值映射为植被碳密度
def vegetation_carbon_density(grid_value):
    vegetation_density_mapping = {0: 0, 1: 166, 2: 160, 3: 146, 4: 141, 5: 153.5, 6: 166, 7: 135.5, 8: 109, 9: 100,
                                  10: 120,
                                  11: 14, 12: 27, 13: 20, 14: 35, 15: 10}
    return vegetation_density_mapping.get(grid_value, 0)  # Return 0 if grid_value is None


# 定义函数 path_file，根据年份获取文件路径
def path_file(year):
    folder_path = r'E:\Data\Landuse\Try\SSP126\SSP126_INCLLUDE2020'
    files = os.listdir(folder_path)

    for file in files:
        if str(year) in file and file.endswith('.tif'):
            # print(file)

            return os.path.join(folder_path, file)


# 定义函数 area_pixel，获取单个像元面积
def area_pixel(year):
    file_path = path_file(year)
    dataset = gdal.Open(file_path)
    if dataset is None:
        print("Error: Unable to open dataset.")
        return None
    transform_coordinate = dataset.GetGeoTransform()
    # pixel_width = transform_coordinate[1]
    # pixel_height = -transform_coordinate[5]
    area_pixel = 2500.0
    return area_pixel


# 定义函数 vegetation_function_vectorized，植被生成曲线
def vegetation_function_vectorized(change_years, mature_years):
    if mature_years is None:
        return np.zeros_like(change_years)  # 返回与 change_years 形状相同的全零数组

    # mature_years_nonzero = np.where(mature_years == 0, 0.0001, mature_years).astype(float)  # 避免除零错误
    y = np.power(1.0 - np.exp(-3.0 * ((change_years + 1) / mature_years)), 2)
    y = np.where(np.isnan(y), 0, y)  # 将NaN值替换为0
    return y


# 定义函数 soil_function，土壤碳扩散曲线
def soil_function(change_years):
    soil_time = 50
    soil_time_fiction = soil_time / 10.0
    k = math.log(2) / soil_time_fiction
    y = 1.0 - np.exp((-1.0 * k * change_years))
    return y


input_raster_path = path_file(2030)


# 获取像元的属性信息
def get_attribute(year):
    file_path = path_file(year)

    if file_path is None:
        print(f"找不到 {year} 年的 TIFF 文件")
        return None

    dataset = gdal.Open(file_path)
    if dataset is None:
        print(f"无法打开 {year} 年的数据集")
        return None

    geotransform = dataset.GetGeoTransform()
    if geotransform is None:
        print(f"无法获取 {year} 年的地理转换信息")
        return None

    rows = dataset.RasterYSize
    cols = dataset.RasterXSize
    min_lon = geotransform[0]
    max_lat = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    max_lon = min_lon + cols * pixel_width
    min_lat = max_lat + rows * pixel_height

    data = dataset.ReadAsArray()
    lats, lons = np.meshgrid(np.linspace(min_lat, max_lat, rows), np.linspace(min_lon, max_lon, cols))
    df_attribute = pd.DataFrame({'Lat': lats.flatten(), 'Lon': lons.flatten(), 'value': data.flatten()})

    df_attribute['value'] = df_attribute['value'].replace(0, np.nan)

    return df_attribute


# def get_attribute(year):
#     file_path = path_file(year)
#     if file_path is None:
#         print(f"No TIFF file found for the year {year}")
#         return None
#
#     with rasterio.open(file_path) as src:
#         data = src.read(1, masked=True)
#         lats, lons = zip(*[src.xy(i, j) for i in range(src.height) for j in range(src.width)])
#         df_attribute = pd.DataFrame({'Lat': lats, 'Lon': lons, 'value': data.flatten()})
#
#     # Check for missing values
#     print("Number of missing values in Lat column:", df_attribute['Lat'].isnull().sum())
#     print("Number of missing values in Lon column:", df_attribute['Lon'].isnull().sum())
#
#     # Print min and max values for debugging
#     print("Min Lat:", df_attribute['Lat'].min())
#     print("Max Lat:", df_attribute['Lat'].max())
#     print("Min Lon:", df_attribute['Lon'].min())
#     print("Max Lon:", df_attribute['Lon'].max())
#
#     return df_attribute

# 保存为栅格地图
# def save_as_geotiff(df, output_path, input_raster_path):
#     # 读取输入栅格以获取空间信息
#     raster_ds = rioxarray.open_rasterio(input_raster_path)
#     data_2030 = get_attribute(2030)
#
#     merged_df = pd.merge(data_2030, df, on=['Lon', 'Lat'])
#     # print(merged_df)
#
#     # 创建DataArray并写入GeoTIFF文件
#     data_array = xr.DataArray(
#         merged_df['CarbonEmission'].values.reshape(raster_ds.sizes['y'], raster_ds.sizes['x']),
#         coords={'y': raster_ds.coords['y'], 'x': raster_ds.coords['x']},
#
#         dims=['y', 'x']
#     )
#     # print(data_array)
#
#     data_array.rio.set_spatial_dims('x', 'y', inplace=True)
#     data_array.rio.write_crs(raster_ds.rio.crs, inplace=True)
#     data_array.rio.write_transform(raster_ds.rio.transform(), inplace=True)
#
#     data_array.rio.to_raster(output_path)


def save_as_geotiff(df, output_path, input_raster_path, column_name):
    """
    保存 DataFrame 中的碳储存或碳排放数据为 GeoTIFF 文件。

    :param df: DataFrame，包含要保存的数据（碳储存或碳排放）
    :param output_path: 输出路径，保存为 .tif 文件
    :param input_raster_path: 输入的栅格文件路径，用于获取空间信息
    :param column_name: 要保存的列名，可以是 'Carbon_Storage' 或 'Emission'
    """
    # 读取输入栅格以获取空间信息
    raster_ds = rioxarray.open_rasterio(input_raster_path)

    # 合并输入数据和空间信息
    merged_df = pd.merge(get_attribute(2030), df, on=['Lon', 'Lat'])

    # 创建DataArray并写入GeoTIFF文件
    data_array = xr.DataArray(
        merged_df[column_name].values.reshape(raster_ds.sizes['y'], raster_ds.sizes['x']),
        coords={'y': raster_ds.coords['y'], 'x': raster_ds.coords['x']},
        dims=['y', 'x']
    )

    data_array.rio.set_spatial_dims('x', 'y', inplace=True)
    data_array.rio.write_crs(raster_ds.rio.crs, inplace=True)
    data_array.rio.write_transform(raster_ds.rio.transform(), inplace=True)

    # 保存为 GeoTIFF 文件
    data_array.rio.to_raster(output_path)


#
# import numpy as np
#
# def get_pixel_coordinates_with_m_changes(start_year, end_year, m, n):
#     """
#     获取在从 start_year 到 end_year 过程中属性值恰好变化了 m 次的像元的经纬度，
#     并在筛选出前 n 个像元后打印变化前后的值和发生变化的年份，且变化后的植被碳密度大于变化前，
#     且变化后的 value 值不能为空。
#
#     :param start_year: 开始年份 (如1990年)
#     :param end_year: 结束年份 (如2100年)
#     :param m: 像元值变化的次数
#     :param n: 只保留前 n 个变化了 m 次的像元
#     :return: DataFrame，包含前 n 个变化了 m 次的像元的经纬度
#     """
#     pixel_change_tracker = {}  # 用来跟踪每个像元值变化的字典
#     pixel_year_tracker = {}  # 用来记录每个像元值变化的年份
#
#     # 遍历每一年，记录像元值
#     for year in range(start_year, end_year + 1, 5):
#         attributes = get_attribute(year)
#         if attributes is not None:
#             for idx, row in attributes.iterrows():
#                 lat, lon, value = row['Lat'], row['Lon'], row['value']
#                 pixel_id = (lat, lon)
#
#                 # 如果该像元没有被记录过，初始化其值变化记录
#                 if pixel_id not in pixel_change_tracker:
#                     pixel_change_tracker[pixel_id] = []
#                     pixel_year_tracker[pixel_id] = []  # 初始化年份记录
#
#                 # 记录每年的像元值和年份
#                 pixel_change_tracker[pixel_id].append(value)
#                 pixel_year_tracker[pixel_id].append(year)
#
#     # 过滤出值恰好变化了 m 次的像元
#     pixels_with_m_changes = {
#         pixel_id: values for pixel_id, values in pixel_change_tracker.items()
#         if len(set(values)) == m + 1 and values[0] != values[-1] and values[-1] is not None and not np.isnan(values[-1])
#     }
#
#     filtered_pixels_list = []
#
#     # 打印出这些像元的变化前后值及其发生变化的年份，并且确保变化后的植被碳密度大于变化前的碳密度
#     for pixel_id, values in pixels_with_m_changes.items():
#         years = pixel_year_tracker[pixel_id]
#
#         if values[0] is not None and not np.isnan(values[0]):
#             initial_density = vegetation_carbon_density(values[0])
#         else:
#             continue
#
#         # 初始化一个标志位，确保至少有一次碳密度增加的情况
#         density_increased = False
#
#         # 遍历变化值，计算和打印植被碳密度
#         for i in range(1, len(values)):
#             if values[i] is not None and not np.isnan(values[i]):
#                 previous_density = vegetation_carbon_density(values[i - 1])
#                 current_density = vegetation_carbon_density(values[i])
#
#                 if current_density > previous_density:
#                     density_increased = True
#                     print(f"Changed from {values[i - 1]} to {values[i]} between {years[i - 1]} and {years[i]}")
#                     print(f"Vegetation Carbon Density increased from {previous_density} to {current_density}")
#
#         # 只有当至少有一次碳密度增加时，才将像元的经纬度添加到过滤后的列表中
#         if density_increased:
#             print(f"Final value: {values[-1]} in {years[-1]}, Vegetation Carbon Density: {vegetation_carbon_density(values[-1])}\n")
#             filtered_pixels_list.append((pixel_id[0], pixel_id[1]))
#
#         # 如果已经有了足够的像元，则停止
#         if len(filtered_pixels_list) >= n:
#             break
#
#     # 将这些像元转换为 DataFrame
#     filtered_pixels = pd.DataFrame(filtered_pixels_list, columns=['Lat', 'Lon'])
#
#     return filtered_pixels


def calculate_carbon_storage_for_selected_pixels_with_emissions(end_year):
    """
    根据给定的像元经纬度，计算每年的碳储存，保存每年像元的属性值。

    :param pixel_coordinates: 包含变化了 m 次的像元的经纬度的 DataFrame
    :param start_year: 开始年份
    :param end_year: 结束年份
    :return: 年份数据字典，包含每个年份的碳储存和属性值
    """
    global data_list_df
    yearly_dataframes = {}  # 保存每年的数据
    data_list = []  # 用于存储所有年份所有像元的碳储存
    previous_year_storage = {}  # 用字典存储每个年份的每个像元的碳储存，键为年份，值为(lat, lon)的字典
    previous_year_value = {}  # 用字典存储每个年份的每个像元的 value
    # 获取1990年的数据并计算像素面积
    attributes_1990 = get_attribute(1990)
    pixel_area_1990 = area_pixel(1990)
    temp_current_year = []

    # 初始化previous_year_storage和previous_year_value的1990年数据
    previous_year_storage[1990] = {}
    previous_year_value[1990] = {}

    # 处理1990年的数据
    for idx, row in attributes_1990.iterrows():
        lat, lon = row['Lat'], row['Lon']  # 获取像元的经纬度

        # 获取1990年的碳密度值
        pixel_data_1990 = attributes_1990.loc[
            (attributes_1990['Lat'] == lat) & (attributes_1990['Lon'] == lon), 'value'
        ]

        # 确保 value 为 NaN 时，碳储存也为 NaN
        if len(pixel_data_1990) > 0 and not pd.isna(pixel_data_1990.values[0]):
            value_1990 = pixel_data_1990.values[0]  # 取出1990年的value值
            vegetation_density_1990 = vegetation_carbon_density(value_1990)
            carbon_storage_1990 = vegetation_density_1990 * pixel_area_1990
        else:
            value_1990 = np.nan
            carbon_storage_1990 = np.nan  # 如果找不到数据或value是NaN，设置为 NaN

        # 保存1990年的碳储存和value值到字典中
        previous_year_storage[1990][(lat, lon)] = carbon_storage_1990
        previous_year_value[1990][(lat, lon)] = value_1990

        # 保存1990年的数据到 temp_current_year 和 data_list
        temp_current_year.append([1990, lat, lon, carbon_storage_1990, value_1990])
        data_list.append([1990, lat, lon, carbon_storage_1990, value_1990])

    yearly_dataframes[1990] = pd.DataFrame(temp_current_year, columns=['Year', 'Lat', 'Lon', 'Carbon_Storage', 'value'])

    # 遍历后续的年份（以5年为间隔），计算当前年份的碳储存和保存属性值
    last_year = 1990  # 初始化上个年份为1990年
    for year in tqdm(range(1995, end_year + 1, 5)):
        attributes_current = get_attribute(year)  # 获取当前年份的属性数据
        pixel_area_current = area_pixel(year)  # 获取当前年份的像元面积
        temp_current_year = []
        attributes_previous = get_attribute(year - 5)

        # 初始化当前年份的 storage 和 value
        previous_year_storage[year] = {}
        previous_year_value[year] = {}

        #for idx, row in pixel_coordinates.iterrows():

        for idx, row in attributes_1990.iterrows():
            lat, lon = row['Lat'], row['Lon']  # 获取像元的经纬度

            # 获取当前年份的 value 值
            value_current = attributes_current.loc[
                (attributes_current['Lat'] == lat) & (attributes_current['Lon'] == lon), 'value'
            ]
            value_current = value_current.values[0] if len(value_current) > 0 else np.nan

            value_previous = attributes_previous.loc[
                (attributes_previous['Lat'] == lat) & (attributes_previous['Lon'] == lon), 'value'
            ]
            value_previous = value_previous.values[0] if len(value_previous) > 0 else np.nan

            # 从 previous_year_storage 或者 yearly_dataframes 获取上个年份的碳储存值
            if year - 5 in yearly_dataframes:
                previous_carbon_storage_row = yearly_dataframes[year - 5].loc[
                    (yearly_dataframes[year - 5]['Lat'] == lat) & (
                            yearly_dataframes[year - 5]['Lon'] == lon), 'Carbon_Storage'
                ]
                previous_carbon_storage = previous_carbon_storage_row.values[0] if len(
                    previous_carbon_storage_row) > 0 else np.nan
            else:
                previous_carbon_storage = np.nan

            if pd.isna(value_current) or pd.isna(previous_carbon_storage):
                carbon_storage_current = np.nan
                temp_current_year.append([year, lat, lon, carbon_storage_current, value_current])
                data_list.append([year, lat, lon, carbon_storage_current, value_current])
            else:
                if value_current != value_previous:
                    vegetation_density_current = vegetation_carbon_density(value_current)
                    carbon_storage_current_unrevised = vegetation_density_current * pixel_area_current

                    if previous_carbon_storage >= carbon_storage_current_unrevised:
                        carbon_storage_current = carbon_storage_current_unrevised
                        temp_current_year.append([year, lat, lon, carbon_storage_current, value_current])
                        data_list.append([year, lat, lon, carbon_storage_current, value_current])
                    else:
                        mature_years = mature_year(value_current)
                        if mature_years == 1:
                            carbon_storage_current = carbon_storage_current_unrevised
                            temp_current_year.append([year, lat, lon, carbon_storage_current, value_current])
                            data_list.append([year, lat, lon, carbon_storage_current, value_current])
                        else:
                            for change_years in range(5, 101, 5):
                                future_year = year + change_years - 5
                                additional_carbon_storage_current = vegetation_function_vectorized(
                                    change_years, mature_years) * (
                                                                            carbon_storage_current_unrevised - previous_carbon_storage)


                                carbon_storage_current = previous_carbon_storage + additional_carbon_storage_current
                                # print(
                                #     f"Year: {future_year}, Lat: {lat}, Lon: {lon}, Carbon_Storage: {carbon_storage_current}, Change_Years: {change_years}")

                                if future_year == year:
                                    temp_current_year.append(
                                        [year, lat, lon, carbon_storage_current, value_current])

                                if future_year>2020 and year== 2020:
                                    continue
                                else:
                                # if future_year <=2200:
                                    data_list.append([future_year, lat, lon, carbon_storage_current, value_current])


                else:
                    existing_value = next(
                        (row[3] for row in data_list if row[0] == year and row[1] == lat and row[2] == lon), None)
                    carbon_storage_current = existing_value if existing_value is not None else vegetation_carbon_density(
                        value_current) * pixel_area_current
                    temp_current_year.append([year, lat, lon, carbon_storage_current, value_current])
                    data_list.append([year, lat, lon, carbon_storage_current, value_current])



        yearly_dataframes[year] = pd.DataFrame(temp_current_year,
                                               columns=['Year', 'Lat', 'Lon', 'Carbon_Storage', 'value'])
        data_list_df = pd.DataFrame(data_list, columns=['Year', 'Lat', 'Lon', 'Carbon_Storage', 'value'])
        # unique_years = data_list_df['Year'].unique()
        # print("唯一年份列表：", unique_years)


        last_year = year  # 更新 last_year 为当前年份

    return data_list_df


# 保存碳储存和碳排放的 GeoTIFF 文件到指定文件夹
def save_all_as_geotiff(yearly_dataframes, emissions_dataframes, input_raster_path, output_dir):
    """
    将所有按年份分组的碳储存和碳排放数据保存为 .tif 文件。

    :param yearly_dataframes: 每个年份的碳储存 DataFrame
    :param emissions_dataframes: 每两个年份之间的碳排放 DataFrame
    :param input_raster_path: 输入的栅格文件路径，用于获取空间信息
    :param output_dir: 输出的根文件夹，保存所有的 .tif 文件
    """
    # 创建碳储存和碳排放的文件夹
    carbon_storage_dir = os.path.join(output_dir, 'carbon_storage')
    carbon_emission_dir = os.path.join(output_dir, 'carbon_emission')
    os.makedirs(carbon_storage_dir, exist_ok=True)
    os.makedirs(carbon_emission_dir, exist_ok=True)

    # 保存每个年份的碳储存数据为 GeoTIFF 文件
    for year, df in yearly_dataframes.items():
        output_path = os.path.join(carbon_storage_dir, f"carbon_storage_{year}.tif")  # 文件名中包含年份
        save_as_geotiff(df, output_path, input_raster_path, 'Carbon_Storage')
        print(f"Saved Carbon Storage GeoTIFF for year {year} to {output_path}")

    # 保存每两个年份之间的碳排放数据为 GeoTIFF 文件
    for year_range, df in emissions_dataframes.items():
        output_path = os.path.join(carbon_emission_dir, f"carbon_emission_{year_range}.tif")  # 文件名中包含年份范围
        save_as_geotiff(df, output_path, input_raster_path, 'Emission')
        print(f"Saved Carbon Emission GeoTIFF for year range {year_range} to {output_path}")

end_year1=2100
# 计算碳储存和碳排放，并保存为 GeoTIFF 文件
data_list_2015 = calculate_carbon_storage_for_selected_pixels_with_emissions(end_year1)

# 按年份进行排序
data_list_df_sorted = data_list_2015.sort_values(by='Year', kind='mergesort')

# 存储每年碳储存和碳排放的字典
yearly_dataframes = {}
emissions_dataframes = {}

# 存储上一个年份的碳储存数据
previous_year_data = None

# 遍历按年份分组的数据
import pandas as pd

# 创建用于存储每年碳储存总和和每两个年份之间碳排放总和的列表
carbon_storage_sums = []
carbon_emission_sums = []

# 遍历按年份分组的数据
for year, group in data_list_df_sorted.groupby('Year'):
    if year > end_year1:
        continue

    # 去重，确保每个像元的经纬度唯一
    group_unique = group.drop_duplicates(subset=['Lat', 'Lon'], keep='last')

    # 将当前年份的碳储存数据保存到 yearly_dataframes
    yearly_dataframes[year] = group_unique.copy()

    # 计算当前年份的碳储存总和
    carbon_storage_sum = group_unique['Carbon_Storage'].sum()
    carbon_storage_sums.append({'Year': year, 'Carbon_Storage_Sum': carbon_storage_sum})

    # 输出每个年份的碳储存总和
    print(f"\nYear: {year}, Carbon Storage Sum: {carbon_storage_sum}")

    # 如果 previous_year_data 不为空，则计算碳排放
    if previous_year_data is not None:
        # 合并当前年份和上一个年份的数据，基于经纬度
        merged = pd.merge(previous_year_data, group_unique, on=['Lat', 'Lon'], suffixes=('_prev', '_curr'))

        # 计算碳排放量：上一个年份的碳储存减去当前年份的碳储存
        merged['Emission'] = merged['Carbon_Storage_prev'] - merged['Carbon_Storage_curr']

        # 将当前年份之间的碳排放数据保存到 emissions_dataframes
        year_range = f"{merged['Year_prev'].iloc[0]}-{merged['Year_curr'].iloc[0]}"
        emissions_dataframes[year_range] = merged[['Lat', 'Lon', 'Emission', 'value_prev', 'value_curr']].copy()

        # 计算当前年份和上一个年份之间的碳排放总和
        carbon_emission_sum = merged['Emission'].sum()
        carbon_emission_sums.append({'Year_Range': f"{merged['Year_prev'].iloc[0]}-{merged['Year_curr'].iloc[0]}", 'Carbon_Emission_Sum': carbon_emission_sum})

        # 输出每两个年份的碳排放总和
        print(f"\nYear Range: {merged['Year_prev'].iloc[0]}-{merged['Year_curr'].iloc[0]}, Carbon Emission Sum: {carbon_emission_sum}")

    # 更新 previous_year_data 为当前年份的 DataFrame
    previous_year_data = group_unique

# 最后将碳储存和碳排放的总和保存到Excel
with pd.ExcelWriter(r'E:\Data\Landuse\Try\SSP126\SSP126_INCLLUDE2020\out1\ssp126except2020carbon_storage_and_emission_summary1.xlsx') as writer:
    # 将每年碳储存总和保存到一个DataFrame
    carbon_storage_sums_df = pd.DataFrame(carbon_storage_sums)
    carbon_storage_sums_df.to_excel(writer, sheet_name='Carbon_Storage_Sums', index=False)

    # 将每两个年份的碳排放总和保存到一个DataFrame
    carbon_emission_sums_df = pd.DataFrame(carbon_emission_sums)
    carbon_emission_sums_df.to_excel(writer, sheet_name='Carbon_Emission_Sums', index=False)

    print("碳储存和碳排放总和数据已保存到Excel文件")


# 确保 input_raster_path 指向具体的栅格文件（例如 .tif 文件），而不是文件夹
input_raster_path = r'E:\Data\Landuse\Try\SSP126\SSP126_INCLLUDE2020\SSP126_50KM_2060.tif'
out_raster_path=r'E:\Data\Landuse\Try\SSP126\SSP126_INCLLUDE2020\out1'


# 调用保存函数，将碳储存和碳排放数据保存为 GeoTIFF
save_all_as_geotiff(yearly_dataframes, emissions_dataframes, input_raster_path, out_raster_path)


# 示例用法：
# m = 1  # 例如，选择那些变化了 3 次的像元
# n = 2  # 只保留前 5 个像元
# pixel_coordinates = get_pixel_coordinates_with_m_changes(1990, 2100, m, n)

# # 根据经纬度计算这些像元的碳储存和排放
# data_list_2015 = calculate_carbon_storage_for_selected_pixels_with_emissions(2000)
# #print("调用后年份列表：",data_list_2015['Year'].unique())
# # 调用计算函数，得到 data_list_df
#
# # 第一步：按年份进行排序，保持数据顺序不变
# data_list_df_sorted = data_list_2015.sort_values(by='Year', kind='mergesort')
#
# # 第二步：创建一个字典来存储每个年份的碳储存数据
# yearly_dataframes = {}
# # 创建一个字典来存储每两个年份之间的碳排放数据
# emissions_dataframes = {}
#
# # 第三步：按年份进行分组，并处理每个年份的数据
# previous_year_data = None  # 用于存储上一个年份的 DataFrame
#
# for year, group in data_list_df_sorted.groupby('Year'):
#     # 对每个年份的分组数据，按 ['Lat', 'Lon'] 进行去重，只保留最后一个
#     group_unique = group.drop_duplicates(subset=['Lat', 'Lon'], keep='last')
#
#     # 保存每个年份的去重后的数据到字典中
#     yearly_dataframes[year] = group_unique
#
#     # 输出每个年份的碳储存数据
#     print(f"\nCarbon Storage DataFrame for {year}:")
#     print(group_unique)
#     print(group_unique.head())  # 打印前几行数据
#
#     # 如果 previous_year_data 不为空，则开始计算排放量
#     if previous_year_data is not None:
#         # 合并当前年份和上一个年份的数据，基于经纬度
#         merged = pd.merge(previous_year_data, group_unique, on=['Lat', 'Lon'], suffixes=('_prev', '_curr'))
#
#         # 计算碳排放量：上一个年份的碳储存减去当前年份的碳储存
#         merged['Emission'] = merged['Carbon_Storage_prev'] - merged['Carbon_Storage_curr']
#
#         # 保存经纬度、排放量以及 value 值到新的 DataFrame
#         emission_df = merged[['Lat', 'Lon', 'Emission', 'value_prev', 'value_curr']]
#
#         # 用上一个年份和当前年份命名 DataFrame
#         emissions_dataframes[f"{merged['Year_prev'].iloc[0]}-{merged['Year_curr'].iloc[0]}"] = emission_df
#
#         # 输出每两个年份的碳排放数据
#         print(f"\nEmission DataFrame for {merged['Year_prev'].iloc[0]}-{merged['Year_curr'].iloc[0]}:")
#         print(emission_df)
#         print(emission_df.head())  # 打印前几行数据示例
#
#     # 更新 previous_year_data 为当前年份的 DataFrame
#     previous_year_data = group_unique

    # 输出每个年份的 DataFrame 结构
    # print(group_unique)
    # print(f"\nYear: {year}")
    # print("DataFrame Format:")
    # print(f"Columns: {group_unique.columns.tolist()}")
    # print("Sample Data:")
    # print(group_unique.head())  # 打印前几行数据作为示例

#     # 计算该年份的碳储存总值
#     total_carbon_storage = group_unique['Carbon_Storage'].sum()
#     print(f"Year {year}: Total Carbon Storage = {total_carbon_storage}")
#
#     # 将年份和碳储存总值存储到列表中
#     total_storage_per_year.append((year, total_carbon_storage))
#
# # 第五步：计算每年的碳排放量（即每年与上一年碳储存的差异）
# for i in range(1, len(total_storage_per_year)):
#     year, current_storage = total_storage_per_year[i]
#     prev_year, prev_storage = total_storage_per_year[i - 1]
#
#     # 计算碳排放量（上一年的碳储存减去当前年的碳储存）
#     emissions = prev_storage - current_storage
#     total_emissions_per_year.append((year, emissions))
#
# # 将总碳储存值和碳排放量转换为 DataFrame，方便后续处理或输出
# total_storage_df = pd.DataFrame(total_storage_per_year, columns=['Year', 'Total_Carbon_Storage'])
#
# # 碳排放量 DataFrame
# emissions_df = pd.DataFrame(total_emissions_per_year, columns=['Year', 'Carbon_Emissions'])
#
# # 合并两个 DataFrame，按年份对齐
# result_df = pd.merge(total_storage_df, emissions_df, on='Year', how='left')
#
# # 输出合并后的 DataFrame
# #print(result_df)
# #
# # # 保存结果到 Excel
# # result_df.to_excel(r'D:\Works\DATA\try\50KM\SSP370carbon_storage_and_emissions_1990_2100.xlsx', index=False)
#
# # 绘制图表
# plt.figure(figsize=(10, 6))
#
# # 绘制碳储存曲线
# plt.plot(total_storage_df['Year'], total_storage_df['Total_Carbon_Storage'], marker='o', linestyle='-', color='b', label='Total Carbon Storage')
#
# # 绘制碳排放曲线
# plt.plot(emissions_df['Year'], emissions_df['Carbon_Emissions'], marker='o', linestyle='--', color='r', label='Carbon Emissions')
#
# # 图表标题和轴标签
# plt.title('Total Carbon Storage and Carbon Emissions Over Time')
# plt.xlabel('Year')
# plt.ylabel('Amount')
#
# # 显示网格
# plt.grid(True)
#
# # 添加图例
# plt.legend()
#
# # 展示图表
# plt.show()
#
#
#
