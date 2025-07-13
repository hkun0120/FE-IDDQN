import pandas as pd
import numpy as np
from datetime import datetime
import json
import networkx as nx
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

dtype = {str(i): 'string' for i in range(100)}   # 先全部当字符串读

def load_data(path):
    return pd.read_csv(path,engine='python',
                 dtype=dtype,
                 on_bad_lines='skip',   # 忽略异常行
                 nrows=1000)
# 加载五个CSV文件
process_definition = load_data('~/Downloads/oceanbase_t_ds_process_definition.csv')
process_instance = load_data('~/Downloads/gaussdb_t_ds_process_instance_a.csv')
task_definition = load_data('~/Downloads/oceanbase_t_ds_task_definition.csv')
task_instance = load_data('~/Downloads/gaussdb_t_ds_task_instance_a.csv')
process_task_relation = load_data('~/Downloads/oceanbase_t_ds_process_task_relation.csv')

# 查看数据基本信息
for df, name in zip(
    [process_definition, process_instance, task_definition, task_instance, process_task_relation],
    ['process_definition', 'process_instance', 'task_definition', 'task_instance', 'process_task_relation']
):
    print(f"===== {name} =====")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("\n")

# 时间字段转换函数
def convert_datetime(df, datetime_cols):
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# 转换各表中的时间字段
datetime_cols = ['create_time', 'update_time', 'start_time', 'end_time', 'schedule_time']
process_definition = convert_datetime(process_definition, datetime_cols)
process_instance = convert_datetime(process_instance, datetime_cols)
task_definition = convert_datetime(task_definition, datetime_cols)
task_instance = convert_datetime(task_instance, datetime_cols)

# 检查关键字段的缺失情况
def check_missing_key_fields(df, key_fields, df_name):
    missing = df[key_fields].isnull().sum()
    print(f"Missing values in key fields of {df_name}:")
    print(missing)
    return missing

# 处理缺失值
def handle_missing_values(df, strategy='drop'):
    """
    处理缺失值
    strategy: 'drop' - 删除含有缺失值的行
              'fill_mean' - 用均值填充数值型缺失值
              'fill_mode' - 用众数填充分类型缺失值
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill_mean':
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'fill_mode':
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# 对各表应用缺失值处理
process_instance = handle_missing_values(process_instance, 'fill_mean')
task_instance = handle_missing_values(task_instance, 'fill_mean')


# 检测执行时间的异常值
def detect_execution_time_anomalies(df):
    """检测执行时间异常值"""
    df['execution_time'] = (df['end_time'] - df['start_time']).dt.total_seconds()

    # 负值检测
    negative_time = df[df['execution_time'] < 0]
    print(f"Negative execution time count: {len(negative_time)}")

    # 极端值检测 (使用IQR方法)
    Q1 = df['execution_time'].quantile(0.25)
    Q3 = df['execution_time'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['execution_time'] < (Q1 - 1.5 * IQR)) |
                  (df['execution_time'] > (Q3 + 1.5 * IQR))]
    print(f"Execution time outliers count: {len(outliers)}")

    return negative_time, outliers


# 处理执行时间异常值
def handle_execution_time_anomalies(df, negative_time, outliers, strategy='clip'):
    """
    处理执行时间异常值
    strategy: 'clip' - 将异常值限制在合理范围内
              'remove' - 删除异常值
              'replace' - 用中位数替换异常值
    """
    if strategy == 'clip':
        Q1 = df['execution_time'].quantile(0.25)
        Q3 = df['execution_time'].quantile(0.75)
        IQR = Q3 - Q1
        df['execution_time'] = df['execution_time'].clip(lower=0, upper=Q3 + 1.5 * IQR)
    elif strategy == 'remove':
        df = df[~df.index.isin(negative_time.index)]
        df = df[~df.index.isin(outliers.index)]
    elif strategy == 'replace':
        median = df['execution_time'].median()
        df.loc[df.index.isin(negative_time.index), 'execution_time'] = median
        df.loc[df.index.isin(outliers.index), 'execution_time'] = median

    return df


# 应用到任务实例表
negative_time, outliers = detect_execution_time_anomalies(task_instance)
task_instance = handle_execution_time_anomalies(task_instance, negative_time, outliers, 'clip')


def extract_task_features(task_instance, task_definition):
    """提取任务级别特征"""
    # 合并任务实例和任务定义表，使用suffixes参数避免列名冲突
    task_features = task_instance.merge(
        task_definition[['code', 'task_type', 'task_params', 'task_priority']],
        left_on='task_code',
        right_on='code',
        how='left',
        suffixes=('', '_def')  # 任务定义表的列添加_def后缀
    )
    
    # 调试信息：检查合并后的列
    print("合并后的列名:", task_features.columns.tolist())
    print("task_params列是否存在:", 'task_params' in task_features.columns)
    if 'task_params' in task_features.columns:
        print("task_params列的前几个值:", task_features['task_params'].head())

    # 计算任务执行时间
    task_features['execution_time'] = (task_features['end_time'] - task_features['start_time']).dt.total_seconds()

    # 提取任务参数复杂度
    def get_params_complexity(params):
        try:
            if pd.isna(params):
                return 0
            params_dict = json.loads(params)
            return len(json.dumps(params_dict))
        except:
            return 0

    # 检查task_params列是否存在，如果不存在则创建一个默认值
    if 'task_params' in task_features.columns:
        task_features['params_complexity'] = task_features['task_params'].apply(get_params_complexity)
    else:
        print("警告：task_params列不存在，使用默认值0")
        task_features['params_complexity'] = 0

    # 任务重试次数
    task_features['retry_times'] = task_features['retry_times'].fillna(0)

    # 任务优先级 - 使用任务定义表中的优先级，如果不存在则使用实例表中的
    if 'task_priority_def' in task_features.columns:
        task_features['task_priority'] = task_features['task_priority_def'].fillna(0)
    else:
        print("警告：task_priority列不存在，使用默认值0")
        task_features['task_priority'] = 0

    # 任务状态编码
    task_features['state_code'] = task_features['state'].astype('category').cat.codes

    # 任务类型独热编码
    task_type_dummies = pd.get_dummies(task_features['task_type'], prefix='task_type')
    task_features = pd.concat([task_features, task_type_dummies], axis=1)

    return task_features


# 应用任务特征提取
task_features = extract_task_features(task_instance, task_definition)


def extract_workflow_features(process_instance, task_instance, process_task_relation):
    """提取工作流级别特征"""
    # 按工作流实例分组统计任务数量
    workflow_task_count = task_instance.groupby('process_instance_id').size().reset_index(name='task_count')

    # 合并到工作流实例表
    workflow_features = process_instance.merge(
        workflow_task_count,
        left_on='id',
        right_on='process_instance_id',
        how='left'
    )

    # 计算工作流执行时间
    workflow_features['execution_time'] = (
                workflow_features['end_time'] - workflow_features['start_time']).dt.total_seconds()

    # 提取时间特征
    workflow_features['hour_of_day'] = workflow_features['start_time'].dt.hour
    workflow_features['day_of_week'] = workflow_features['start_time'].dt.dayofweek
    workflow_features['is_weekend'] = workflow_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 计算工作流的并行度
    def calculate_parallelism(process_def_code):
        # 获取该工作流定义的任务关系
        relations = process_task_relation[process_task_relation['process_definition_code'] == process_def_code]

        if len(relations) == 0:
            return 1

        # 构建有向图
        G = nx.DiGraph()
        for _, row in relations.iterrows():
            G.add_edge(row['pre_task_code'], row['post_task_code'])

        # 计算关键路径长度
        try:
            critical_path_length = nx.dag_longest_path_length(G)
            return critical_path_length
        except:
            return 1

    # 应用并行度计算（注意：这可能比较耗时，可以只对部分数据计算）
    sample_workflows = workflow_features.head(100)
    sample_workflows['critical_path_length'] = sample_workflows['process_definition_code'].apply(calculate_parallelism)

    # 估计并行度
    sample_workflows['estimated_parallelism'] = sample_workflows['task_count'] / sample_workflows[
        'critical_path_length']

    return workflow_features


# 应用工作流特征提取
workflow_features = extract_workflow_features(process_instance, task_instance, process_task_relation)


def extract_dependency_features(task_instance, process_task_relation, process_instance):
    """提取任务依赖关系特征"""
    # 首先从process_instance表获取process_definition_code
    process_mapping = process_instance[['id', 'process_definition_code']].rename(
        columns={'id': 'process_instance_id'}
    )
    
    # 将process_definition_code添加到task_instance
    task_instance_with_def = task_instance.merge(
        process_mapping,
        on='process_instance_id',
        how='left'
    )

    # 合并任务实例和任务关系表
    dependency_features = task_instance_with_def.merge(
        process_task_relation,
        left_on=['task_code', 'process_definition_code'],
        right_on=['post_task_code', 'process_definition_code'],
        how='left'
    )

    # 计算每个任务的前置任务数量（入度）
    task_in_degree = process_task_relation.groupby('post_task_code').size().reset_index(name='in_degree')

    # 计算每个任务的后置任务数量（出度）
    task_out_degree = process_task_relation.groupby('pre_task_code').size().reset_index(name='out_degree')

    # 合并入度和出度到任务特征
    dependency_features = dependency_features.merge(
        task_in_degree,
        left_on='task_code',
        right_on='post_task_code',
        how='left'
    )

    dependency_features = dependency_features.merge(
        task_out_degree,
        left_on='task_code',
        right_on='pre_task_code',
        how='left'
    )

    # 填充缺失值
    dependency_features['in_degree'] = dependency_features['in_degree'].fillna(0)
    dependency_features['out_degree'] = dependency_features['out_degree'].fillna(0)

    # 计算任务的中心性（入度+出度）
    dependency_features['centrality'] = dependency_features['in_degree'] + dependency_features['out_degree']

    # 调试信息：检查列名
    print("dependency_features列名:", dependency_features.columns.tolist())

    return dependency_features


# 应用依赖关系特征提取
dependency_features = extract_dependency_features(task_instance, process_task_relation, process_instance)


def extract_temporal_features(task_instance, process_instance):
    """提取时序特征"""
    # 合并任务实例和工作流实例
    temporal_features = task_instance.merge(
        process_instance[['id', 'start_time']],
        left_on='process_instance_id',
        right_on='id',
        how='left',
        suffixes=('', '_workflow')
    )

    # 计算任务等待时间（任务开始时间 - 工作流开始时间）
    temporal_features['wait_time'] = (
                temporal_features['start_time'] - temporal_features['start_time_workflow']).dt.total_seconds()

    # 提取时间特征
    temporal_features['hour_of_day'] = temporal_features['start_time'].dt.hour
    temporal_features['day_of_week'] = temporal_features['start_time'].dt.dayofweek
    temporal_features['is_weekend'] = temporal_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 计算任务执行时间在一天中的百分比位置
    temporal_features['time_of_day_pct'] = (temporal_features['hour_of_day'] * 3600 +
                                            temporal_features['start_time'].dt.minute * 60 +
                                            temporal_features['start_time'].dt.second) / 86400

    return temporal_features


# 应用时序特征提取
temporal_features = extract_temporal_features(task_instance, process_instance)


def simulate_resource_features(task_features):
    """模拟资源使用特征（由于缺少实际资源数据）"""
    # 基于任务类型和执行时间模拟CPU使用率
    task_type_cpu_intensity = {
        'SHELL': 0.7,
        'SQL': 0.5,
        'PROCEDURE': 0.6,
        'PYTHON': 0.8,
        'SPARK': 0.9,
        'MR': 0.9,
        'FLINK': 0.85,
        'HTTP': 0.3,
        'DATAX': 0.7,
        'DEPENDENT': 0.1,
        'CONDITIONS': 0.2,
        'SUB_PROCESS': 0.4
    }

    # 默认CPU强度为中等
    default_cpu_intensity = 0.5

    # 应用CPU强度映射
    task_features['cpu_intensity'] = task_features['task_type'].map(
        lambda x: task_type_cpu_intensity.get(x, default_cpu_intensity)
    )

    # 模拟CPU使用率 = CPU强度 * (1 + 随机波动)
    np.random.seed(42)  # 设置随机种子以确保可重复性
    task_features['cpu_usage'] = task_features['cpu_intensity'] * (1 + np.random.normal(0, 0.2, len(task_features)))
    task_features['cpu_usage'] = task_features['cpu_usage'].clip(0, 1)  # 限制在0-1范围内

    # 模拟内存使用
    task_features['memory_usage'] = 0.4 + 0.5 * task_features['params_complexity'] / task_features[
        'params_complexity'].max()
    task_features['memory_usage'] = task_features['memory_usage'] * (1 + np.random.normal(0, 0.15, len(task_features)))
    task_features['memory_usage'] = task_features['memory_usage'].clip(0, 1)

    # 模拟I/O操作频率
    task_features['io_intensity'] = 0.3 + 0.6 * task_features['execution_time'] / task_features['execution_time'].max()
    task_features['io_intensity'] = task_features['io_intensity'] * (1 + np.random.normal(0, 0.1, len(task_features)))
    task_features['io_intensity'] = task_features['io_intensity'].clip(0, 1)

    return task_features


# 应用资源使用特征模拟
resource_features = simulate_resource_features(task_features)


def merge_all_features(task_features, workflow_features, dependency_features, temporal_features, resource_features):
    """融合所有特征"""
    # 合并任务特征和依赖特征
    merged_features = task_features.merge(
        dependency_features[['id_x', 'in_degree', 'out_degree', 'centrality']],
        left_on='id',
        right_on='id_x',
        how='left'
    )
    # 删除重复的id_x列
    merged_features = merged_features.drop(columns=['id_x'])

    # 合并时序特征
    merged_features = merged_features.merge(
        temporal_features[['id', 'wait_time', 'time_of_day_pct', 'is_weekend']],
        on='id',
        how='left'
    )

    # 合并资源特征
    merged_features = merged_features.merge(
        resource_features[['id', 'cpu_usage', 'memory_usage', 'io_intensity']],
        on='id',
        how='left'
    )

    # 合并工作流特征
    merged_features = merged_features.merge(
        workflow_features[['id', 'task_count', 'execution_time']],
        left_on='process_instance_id',
        right_on='id',
        how='left',
        suffixes=('', '_workflow')
    )

    # 填充可能的缺失值
    merged_features = merged_features.fillna(0)

    return merged_features


# 应用特征融合
all_features = merge_all_features(
    task_features,
    workflow_features,
    dependency_features,
    temporal_features,
    resource_features
)


def select_features(all_features, method='correlation', threshold=0.8):
    """
    特征选择
    method: 'correlation' - 基于相关性
            'importance' - 基于特征重要性
    """
    # 选择数值型特征
    numeric_features = all_features.select_dtypes(include=['float64', 'int64'])

    if method == 'correlation':
        # 计算相关性矩阵
        corr_matrix = numeric_features.corr().abs()

        # 获取上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # 找出相关性高的特征
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print(f"Features to drop based on correlation: {to_drop}")

        # 删除高相关性特征
        selected_features = all_features.drop(columns=to_drop)

    elif method == 'importance':
        from sklearn.ensemble import RandomForestRegressor

        # 选择目标变量（这里假设是任务执行时间）
        X = numeric_features.drop(columns=['execution_time'])
        y = numeric_features['execution_time']

        # 训练随机森林模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # 获取特征重要性
        importances = rf.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("Feature importance:")
        print(feature_importance)

        # 选择重要性大于阈值的特征
        important_features = feature_importance[feature_importance['importance'] > threshold]['feature'].tolist()

        # 确保保留目标变量和分类特征
        important_features.append('execution_time')
        categorical_features = all_features.select_dtypes(include=['object', 'category']).columns.tolist()
        all_selected_features = important_features + categorical_features

        selected_features = all_features[all_selected_features]

    return selected_features


# 应用特征选择
selected_features = select_features(all_features, method='correlation', threshold=0.8)


def standardize_features(selected_features):
    """标准化数值特征"""
    # 选择数值型特征
    numeric_features = selected_features.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # 排除ID列和目标变量
    exclude_cols = ['id', 'process_instance_id', 'task_code', 'execution_time']
    numeric_features = [col for col in numeric_features if col not in exclude_cols]

    # 标准化
    scaler = StandardScaler()
    selected_features[numeric_features] = scaler.fit_transform(selected_features[numeric_features])

    return selected_features, scaler


# 应用特征标准化
standardized_features, scaler = standardize_features(selected_features)


def visualize_features(features):
    """特征可视化与分析"""
    # 选择数值型特征
    numeric_features = features.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # 排除ID列
    exclude_cols = ['id', 'process_instance_id', 'task_code']
    numeric_features = [col for col in numeric_features if col not in exclude_cols]

    # 相关性热图
    plt.figure(figsize=(12, 10))
    corr = features[numeric_features].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    plt.close()

    # 执行时间分布
    plt.figure(figsize=(10, 6))
    sns.histplot(features['execution_time'], kde=True)
    plt.title('Task Execution Time Distribution')
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('execution_time_distribution.png')
    plt.close()

    # 任务类型与执行时间关系
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='task_type', y='execution_time', data=features)
    plt.title('Execution Time by Task Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('execution_time_by_task_type.png')
    plt.close()

    # 特征重要性（使用随机森林）
    from sklearn.ensemble import RandomForestRegressor

    X = features[numeric_features].drop(columns=['execution_time'])
    y = features['execution_time']

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    return feature_importance


# 应用特征可视化
feature_importance = visualize_features(standardized_features)

def save_processed_data(features, output_path='processed_features.csv'):
    """保存处理后的特征数据"""
    features.to_csv(output_path, index=False)
    print(f"Processed features saved to {output_path}")

# 保存处理后的特征
save_processed_data(standardized_features, 'fe_iddqn_processed_features.csv')