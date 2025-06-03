import os
import pandas
import pandas as pd
import pymysql
import matplotlib
import rasterio
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import dates as mdates
from pymysql.cursors import DictCursor
from rasterio.transform import Affine
from dotenv import load_dotenv
from sqlalchemy import create_engine


# 明确指定要导入的CSV列及其对应的数据库键
FIELD_MAPPING = {
    # csv列名: 数据库键名
    '编号': '编号',
    '站点': '站点',
    'date': 'date',
    'SO2': 'SO2',
    'NO2': 'NO2',
    'O3': 'O3'
}

load_dotenv()

config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'charset': 'utf8mb4',
    'cursorclass': DictCursor
}

# 设置字体为 SimHei（黑体），可以显示中文
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False    # 避免负号'-'显示成方块


def _init_vars():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_pth = os.path.join(current_dir, 'data')
    vars_dict = {
        'csv_pth': os.path.join(data_pth, 'dataset1.csv'),
        'save_pth': os.path.join(data_pth, 'pollutants.csv'),
        'pollutants': ['SO2', 'NO2', 'O3'],
        # 改成你想要的名字
        'db_name': 'csv_import_db',
        'table_name': 'imported_data'
    }

    return vars_dict


def select_pollutants(csv_pth, save_pth, pollutants):
    """
    :param csv_pth: csv文件路径
    :param save_pth: 提取数据保存路径
    :param pollutants: 污染物类型，用列表存储
    """
    df = pandas.read_csv(csv_pth, encoding='utf-8')
    df_pollutants = df[['编号', '站点', 'date'] + pollutants]

    df_pollutants.to_csv(save_pth, encoding='utf-8')
    print(f'污染物数据保存至：{save_pth}')


def clean_specific_columns(csv_pth, pollutants):
    """
    :param csv_pth: csv文件路径
    :param pollutants: 污染物类型，用列表存储
    """
    # 读取CSV文件
    df = pd.read_csv(csv_pth)
    # 记录原始行数
    original_rows = len(df)
    # 只检查指定列是否包含0或-9999
    mask = ~((df[pollutants] == 0) | (df[pollutants] == -9999)).any(axis=1)
    cleaned_df = df[mask]
    # 记录删除的行数
    removed_rows = original_rows - len(cleaned_df)
    # 确定保存路径
    output_path = csv_pth
    # 保存文件
    cleaned_df.to_csv(output_path, index=False)

    print(f"处理完成: 原始行数 {original_rows}, 删除 {removed_rows} 行, 剩余 {len(cleaned_df)} 行")
    print(f"文件已保存到: {output_path}")


def test_connection():
    try:
        connection = pymysql.connect(**config)
        print("数据库连接成功！")
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"MySQL版本: {version['VERSION()']}")
    except Exception as e:
        print(f"连接失败: {e}")
    finally:
        if 'connection' in locals() and connection.open:
            connection.close()


def initialize_database():
    """只创建需要的字段（忽略CSV中的无关列）"""
    connection = pymysql.connect(**config)
    try:
        with connection.cursor() as cursor:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS imported_data (
                编号 INT AUTO_INCREMENT PRIMARY KEY,
                站点 VARCHAR(100) NOT NULL,   
                date DATETIME,                     
                SO2 FLOAT,
                NO2 FLOAT,
                O3 FLOAT
            )
            """
            cursor.execute(create_table_sql)
        connection.commit()
    finally:
        connection.close()


# 将csv文件中的特定列导入数据库中
def import_selected_columns(vars_dict):
    try:
        # 1. 使用pandas读取CSV
        df = pd.read_csv(vars_dict['save_pth'], encoding='utf-8')
        # 2. 验证必要列是否存在
        missing_columns = [csv_col for csv_col in FIELD_MAPPING
                           if csv_col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV缺少必要列: {missing_columns}")
        # 3. 选择需要的列并重命名
        df_selected = df[list(FIELD_MAPPING.keys())].rename(columns=FIELD_MAPPING)
        # 4. 数据清洗
        df_selected = df_selected.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
        # 5. 创建SQLAlchemy引擎
        engine = create_engine(
            f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}/{config['database']}",
            connect_args={'charset': 'utf8mb4'}
        )
        # 6. 导入数据库（自动创建表或追加数据）
        df_selected.to_sql(
            name=vars_dict['table_name'],
            con=engine,
            if_exists='fail',  # 如果表存在则追加，第一次导入后可以改为fail，实际上重复导入并不影响
            index=False,  # 不导入pandas的索引列
            chunksize=1000  # 分批提交
        )
        print(f"成功导入 {len(df_selected)} 条数据")

    except Exception as e:
        print(f"导入失败: {str(e)}")

# pymysql数据处理操作
# 1. 查询表中元素个数
def get_total_rows(vars_dict):
    # 获取连接
    try:
        connection = pymysql.connect(**config)

        with connection.cursor() as cursor:
            # 执行COUNT查询
            sql = f"SELECT COUNT(*) AS total FROM `{vars_dict['table_name']}`"
            cursor.execute(sql)
            result = cursor.fetchone()  # 获取结果
            print('表中元素个数为:', result['total'])

    except pymysql.Error as e:
        print(f"数据库错误: {e}")
        return None
    finally:
        if connection:
            connection.close()  # 确保连接关闭


# 2.查询站点
def get_site_information(vars_dict):
    # 获取连接
    try:
        connection = pymysql.connect(**config)
        with connection.cursor() as cursor:
            # 查询数量
            sql = f"SELECT COUNT(DISTINCT 站点) AS num_sites FROM {vars_dict['table_name']}"
            cursor.execute(sql)
            count_result = cursor.fetchone()
            print(f"不重复的站点数量: {count_result['num_sites']}")

            # 查询所有名称
            sql = f"SELECT DISTINCT 站点 FROM {vars_dict['table_name']}"
            cursor.execute(sql)
            names_result = cursor.fetchall()

            site_names = [row['站点'] for row in names_result]
            print("不重复的站点名称列表:", site_names)


    except pymysql.Error as e:
        print(f"数据库错误: {e}")
        return None
    finally:
        if connection:
            connection.close()  # 确保连接关闭


# 3、查询每个站点不同污染物的最大最小值
def get_max_min_group_by_site(vars_dict, pollutants='NO2'):
    try:
        connection = pymysql.connect(**config)
        with connection.cursor() as cursor:
            sql = f"""
                    SELECT 站点, MAX({pollutants}) AS max_{pollutants}, MIN({pollutants}) AS min_{pollutants}
                    FROM {vars_dict['table_name']}
                    GROUP BY 站点
                    """
            cursor.execute(sql)
            results = cursor.fetchall()

            for row in results:
                print(f"站点: {row['站点']}, {pollutants}最大值: {row[f'max_{pollutants}']}, {pollutants}最小值: {row[f'min_{pollutants}']}")

    except pymysql.Error as e:
        print(f"数据库错误: {e}")
        return None
    finally:
        if connection:
            connection.close()  # 确保连接关闭


# 4、给定污染物以及对应的浓度阈值，查询所有节点超出这一阈值的次数
def get_Exceedance_Count(vars_dict, pollutants, threshold):
    try:
        connection = pymysql.connect(**config)
        with connection.cursor() as cursor:
            sql = f"""
                    SELECT 站点, COUNT(*) AS 超标次数
                    FROM {vars_dict['table_name']}
                    WHERE {pollutants} > %s
                    GROUP BY 站点
            """
            cursor.execute(sql, (threshold,))
            results = cursor.fetchall()
            for row in results:
                print(f"站点: {row['站点']}, 超标次数: {row['超标次数']}")

    except pymysql.Error as e:
        print(f"数据库错误: {e}")
        return None
    finally:
        if connection:
            connection.close()  # 确保连接关闭


# 联合Matplotlib可视化处理
# 1、统计某月的污染物浓度平均值，并绘制直方图
def get_monthly_avg(vars_dict, pollutant, target_month):
    try:
        connection = pymysql.connect(**config)
        with connection.cursor() as cursor:
            sql =  f"""
                SELECT 站点, DATE_FORMAT(date, '%%Y-%%m') AS 月份, AVG({pollutant}) AS 月平均浓度
                FROM {vars_dict['table_name']}
                WHERE DATE_FORMAT(date, '%%Y-%%m') = %s
                GROUP BY 站点, DATE_FORMAT(date, '%%Y-%%m')
            """
            cursor.execute(sql, (target_month,))
            results = cursor.fetchall()

    except pymysql.Error as e:
        print(f"数据库错误: {e}")
        return None
    finally:
        if connection:
            connection.close()  # 确保连接关闭

    df = pd.DataFrame(results)

    # 画柱状图：展示所有站点在该月的平均浓度分布
    plt.figure(figsize=(10, 6))
    plt.bar(df['站点'], df['月平均浓度'], color='skyblue', edgecolor='black')

    plt.title(f'{target_month} 各站点 {pollutant} 平均浓度')
    plt.xlabel('站点')
    plt.ylabel(f'{pollutant} 平均浓度 (μg/m³)')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# 2、对某个站点和某种污染物，提取其全年的月度平均浓度并可视化
def get_monthly_avg_by_site(vars_dict, site_name, pollutant):
    try:
        connection = pymysql.connect(**config)
        with connection.cursor() as cursor:
            sql = f"""
                    SELECT 
                        DATE_FORMAT(date, '%%Y-%%m') AS 月份,
                        AVG(`{pollutant}`) AS 月平均浓度
                    FROM {vars_dict['table_name']}
                    WHERE 站点 = %s
                    GROUP BY 月份
                    ORDER BY 月份
                """

            with connection.cursor() as cursor:
                cursor.execute(sql, (site_name,))
                results = cursor.fetchall()

            # 转换为DataFrame方便处理
            df = pd.DataFrame(results, columns=['月份', '月平均浓度'])
            df['月份'] = pd.to_datetime(df['月份'])

    except pymysql.Error as e:
        print(f"数据库错误: {e}")
        return None
    finally:
        if connection:
            connection.close()  # 确保连接关闭

    if df is None or df.empty:
        print("无有效数据可绘制")
        return

    plt.figure(figsize=(12, 6))

    # 折线图（显示趋势）
    plt.plot(
        df['月份'],
        df['月平均浓度'],
        marker='o',
        color='#1f77b4',
        linewidth=2,
        label='月均浓度'
    )

    # 设置图表属性
    plt.title(f"{site_name}站点 {pollutant} 月度平均浓度趋势", fontsize=14)
    plt.xlabel("月份", fontsize=12)
    plt.ylabel(f"{pollutant}浓度 (μg/m³)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 格式化x轴为月份
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # 添加数据标签
    for x, y in zip(df['月份'], df['月平均浓度']):
        plt.text(x, y + 1, f"{y:.1f}", ha='center', va='bottom', fontsize=9)

    plt.legend()
    plt.tight_layout()
    plt.show()


def get_hourly_exceedances(vars_dict, site_name, pollutant, threshold):
    try:
        connection = pymysql.connect(**config)
        with connection.cursor() as cursor:
            sql = f"""
                    SELECT 
                        HOUR(date) AS hour,
                        COUNT(*) AS exceed_count
                    FROM {vars_dict['table_name']}
                    WHERE 
                        站点 = %s AND 
                        `{pollutant}` > %s
                    GROUP BY hour
                    ORDER BY hour
                """

            cursor.execute(sql, (site_name, threshold))
            results = cursor.fetchall()

            # 转换为DataFrame并补全24小时
            df = pd.DataFrame(results, columns=['hour', 'exceed_count'])
            df_full = pd.DataFrame({'hour': range(24)})
            df = df_full.merge(df, on='hour', how='left').fillna(0)

    except pymysql.Error as e:
        print(f"数据库错误: {e}")
        return None
    finally:
        if connection:
            connection.close()  # 确保连接关闭

    if df is None or df.empty:
        print("无超标数据")
        return

    plt.figure(figsize=(14, 6))

    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("custom", colors)

    norm = df['exceed_count'].max() if df['exceed_count'].max() != 0 else 1
    color_values = df['exceed_count'] / norm

    plt.bar(
        df['hour'],
        df['exceed_count'],
        color=cmap(color_values),
        edgecolor='white'
    )

    # 标记最大值
    max_hour = df.loc[df['exceed_count'].idxmax(), 'hour']
    plt.axvline(x=max_hour, color='red', linestyle='--', alpha=0.5)
    plt.text(
        max_hour,
        df['exceed_count'].max() * 0.9,
        f"最易超标时段: {max_hour}:00",
        ha='center',
        color='red'
    )

    plt.title(f"{site_name}站点 {pollutant} 每小时超标次数（阈值={threshold}μg/m³）", fontsize=14)
    plt.xlabel("小时", fontsize=12)
    plt.ylabel("超标次数", fontsize=12)
    plt.xticks(range(24))
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{site_name}_{pollutant}_小时超标统计.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    vars_dict = _init_vars()
    # 1、读取污染物数据并单独保存到一个文件
    select_pollutants(vars_dict['csv_pth'], vars_dict['save_pth'], vars_dict['pollutants'])

    # 2、读取新csv文件并清除其中包含0或-9999的行
    clean_specific_columns(vars_dict['save_pth'], vars_dict['pollutants'])

    # 3、将csv文件移动到MySQL上
    # 3.1、测试连接
    test_connection()
    # 3.2、初始化数据库
    initialize_database()
    # 3.3、将csv文件中的数据导入至数据库
    import_selected_columns(vars_dict)

    # 4、利用MySQL完成各种操作
    # 4.1、获取表中所有行数
    get_total_rows(vars_dict)
    # 4.2、获取表中“站点”字段的信息
    get_site_information(vars_dict)
    # 4.3 查询每个站点不同污染物的最大最小值
    get_max_min_group_by_site(vars_dict, 'NO2')
    # 4.4 给定污染物以及对应的浓度阈值，查询所有节点超出这一阈值的次数
    get_Exceedance_Count(vars_dict, 'NO2', 80)

    # 5、Matplotlib可视化处理
    # 5.1 给定某一污染物以及某一月份，获取所有节点在该月的该污染物浓度平均值并绘制直方图
    get_monthly_avg(vars_dict, pollutant='NO2', target_month='2023-01')
    # 5.2 对某个站点和某种污染物，提取其全年的月度平均浓度并可视化
    get_monthly_avg_by_site(vars_dict, site_name='西乡', pollutant='NO2')
    # 5.3 针对某站点某污染物，每小时超标次数统计
    get_hourly_exceedances(vars_dict, site_name='西乡', pollutant='O3', threshold=75)
