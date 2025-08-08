import os
import pandas as pd
import numpy as np
import shutil
import re
import random
import json

class EVDataTransformer:
    """
    电动汽车数据转换器
    将CSV格式的原始数据转换为时间序列分类格式(.ts文件)
    """
    
    def __init__(self, 
                 channels=None,
                 sample_size=25000,
                 pool_size=50,
                 max_samples_per_file=6,
                 max_samples_per_abnormal_file=6,
                 abnormal_prefixes_file=None):
        """
        初始化数据转换器
        
        Args:
            channels: 通道列表，默认使用标准配置
            sample_size: 每个样本使用的时间点数
            pool_size: 池化窗口大小
            max_samples_per_file: 每个正常文件最多提取的样本数
            max_samples_per_abnormal_file: 每个异常文件最多提取的样本数
            abnormal_prefixes_file: 异常文件前缀txt文件路径（从文件读取）
        """
        # 默认通道配置
        if channels is None:
            self.channels = [
                'SUM_VOLTAGE',    # 总电压
                'SUM_CURRENT',    # 总电流
                'SOC',            # 电池荷电状态
                'U_SD',           # 单体电压标准差（自动计算）
                'T_SD'            # 温度标准差（自动计算）
            ]
        else:
            self.channels = channels
        
        # 数据处理参数
        self.sample_size = sample_size
        self.pool_size = pool_size
        self.expected_output_size = sample_size // pool_size
        self.max_samples_per_file = max_samples_per_file
        self.max_samples_per_abnormal_file = max_samples_per_abnormal_file
        
        # 异常文件前缀
        self.abnormal_prefixes = self.load_abnormal_prefixes_from_file(abnormal_prefixes_file)

        # 标签映射
        self.label_map = {'正常': '0', '异常': '1'}
        
        # 计算实际通道数
        self.actual_channel_count = len(self.channels)
        
        # 样本-文件映射关系
        self.sample_to_file_mapping = {}
    
    def _print_config(self):
        """打印配置信息"""
        print(f"数据处理参数:")
        print(f"- 每个样本时间点数: {self.sample_size}")
        print(f"- 池化窗口大小: {self.pool_size}")
        print(f"- 池化后输出大小: {self.expected_output_size}")
        print(f"- 每个正常文件最多样本数: {self.max_samples_per_file}")
        print(f"- 每个异常文件最多样本数: {self.max_samples_per_abnormal_file}")
        print(f"- 实际通道数: {self.actual_channel_count}")
        print(f"- 通道列表: {self.channels}")
        print(f"- 异常文件前缀数量: {len(self.abnormal_prefixes)}")
        print(f"- 异常文件前缀: {self.abnormal_prefixes}")
        print("="*50)
    
    def get_label_from_filename(self, filename, default_label='正常'):
        """
        从文件名获取标签
        
        Args:
            filename: 文件名
            default_label: 默认标签，用于预测模式
        
        Returns:
            标签值, 是否为异常文件
        """
        # 检查文件名是否以任何一个异常前缀开头
        for prefix in self.abnormal_prefixes:
            if filename.startswith(prefix):
                return self.label_map['异常'], True
        
        return self.label_map[default_label], False

    def find_channels_by_pattern(self, df, pattern):
        """
        在DataFrame中找到匹配模式的列名
        
        Args:
            df: DataFrame
            pattern: 模式，如'U_'或'T_'
        
        Returns:
            匹配的列名列表
        """
        matching_columns = []
        pattern_regex = f"^{pattern}\\d+$"  # 匹配如U_01, U_02, T_01, T_02等
        
        for col in df.columns:
            if re.match(pattern_regex, col):
                matching_columns.append(col)
        
        return sorted(matching_columns)  # 排序确保一致性

    def calculate_standard_deviation(self, df, columns):
        """
        计算指定列在每个时刻的标准差
        
        Args:
            df: DataFrame
            columns: 要计算标准差的列名列表
        
        Returns:
            每个时刻的标准差序列
        """
        if len(columns) == 0:
            return pd.Series([0.0] * len(df))  # 如果没有找到列，返回全0序列
        
        # 选择指定列并计算每行的标准差
        subset_df = df[columns]
        std_values = subset_df.std(axis=1, skipna=True)  # axis=1表示按行计算
        
        # 处理NaN值，用0填充
        std_values = std_values.fillna(0.0)
        
        return std_values

    def add_calculated_channels(self, df):
        """
        为DataFrame添加计算的U_SD和T_SD列
        
        Args:
            df: 输入DataFrame
        
        Returns:
            添加了U_SD和T_SD列的DataFrame
        """
        df_copy = df.copy()
        
        # 找到所有U_xx列
        u_columns = self.find_channels_by_pattern(df_copy, 'U_')
        
        # 找到所有T_xx列  
        t_columns = self.find_channels_by_pattern(df_copy, 'T_')
        
        # 计算U_SD
        df_copy['U_SD'] = self.calculate_standard_deviation(df_copy, u_columns)
        
        # 计算T_SD
        df_copy['T_SD'] = self.calculate_standard_deviation(df_copy, t_columns)
        
        return df_copy

    def validate_channels(self, df, required_channels):
        """
        验证DataFrame中是否包含所需的通道
        
        Args:
            df: DataFrame
            required_channels: 需要的通道列表
        
        Returns:
            可用通道列表, 缺失通道列表
        """
        available_channels = []
        missing_channels = []
        
        for ch in required_channels:
            if ch in df.columns:
                available_channels.append(ch)
            else:
                missing_channels.append(ch)
        
        return available_channels, missing_channels

    def average_pooling(self, data, pool_size=None):
        """
        对数据进行平均池化
        
        Args:
            data: 输入数据列表
            pool_size: 池化窗口大小，默认使用实例配置
        
        Returns:
            池化后的数据
        """
        if pool_size is None:
            pool_size = self.pool_size
            
        if len(data) < pool_size:
            return []
        
        # 计算可以进行池化的完整窗口数量
        num_windows = len(data) // pool_size
        pooled_data = []
        
        for i in range(num_windows):
            start_idx = i * pool_size
            end_idx = start_idx + pool_size
            window_data = data[start_idx:end_idx]
            # 计算平均值
            avg_value = np.mean(window_data)
            pooled_data.append(avg_value)
        
        return pooled_data

    def load_abnormal_prefixes_from_file(self, file_path):
        """
        从txt文件中读取异常文件前缀
        
        Args:
            file_path: txt文件路径，每行一个前缀，如果为None则返回空列表
        
        Returns:
            异常文件前缀列表
        """
        if file_path is None:
            return []
            
        prefixes = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 忽略空行
                    prefixes.append(line)
        print(f"从文件 {file_path} 读取到 {len(prefixes)} 个异常文件前缀")
        return prefixes

    def save_sample_csv(self, sample_folder, file, sample_idx, channel_data, channel_label, label, is_abnormal):
        """
        保存单个样本为CSV文件
        
        Args:
            sample_folder: 样本文件夹路径
            file: 原始文件名
            sample_idx: 样本索引
            channel_data: 通道数据列表（字符串格式）
            channel_label: 通道标签列表
            label: 样本标签
            is_abnormal: 是否为异常样本
        """
        sample_df = pd.DataFrame()
        for i, ch in enumerate(channel_label):
            # 从字符串恢复数据
            values = [float(v) for v in channel_data[i].split(',')]
            sample_df[ch] = values
        
        # 添加标签列
        sample_df['Label'] = int(label)
        
        # 生成文件名
        base_filename = os.path.splitext(file)[0]
        file_type = "abnormal" if is_abnormal else "normal"
        sample_filename = f"{base_filename}_sample_{sample_idx + 1}_label_{label}_{file_type}.csv"
        sample_filepath = os.path.join(sample_folder, sample_filename)
        
        # 保存CSV文件
        sample_df.to_csv(sample_filepath, index=False)

    def process_csv_files(self, csv_folder, default_label='正常', save_samples=False):
        """
        处理CSV文件并转换为时间序列样本
        
        Args:
            csv_folder: CSV文件夹路径
            default_label: 默认标签，用于预测模式
            save_samples: 是否保存单个样本CSV文件
        
        Returns:
            处理后的样本行列表
        """
        all_samples = []
        sample_index = 0  # 全局样本索引
        
        # 创建sample文件夹
        if save_samples:
            sample_folder = './sample'
            if os.path.exists(sample_folder):
                shutil.rmtree(sample_folder)
            os.makedirs(sample_folder)
        
        for file in os.listdir(csv_folder):
            if not file.endswith('.csv'):
                continue
                
            csv_path = os.path.join(csv_folder, file)
            df = pd.read_csv(csv_path)
            df = self.add_calculated_channels(df)
            
            # 处理单个文件的样本
            file_samples, sample_count = self._process_single_file(df, file, default_label, sample_index)
            all_samples.extend(file_samples)
            sample_index += sample_count
        
        return all_samples

    def _process_single_file(self, df, file, default_label='正常', start_sample_index=0):
        """
        处理单个CSV文件，提取样本
        
        Args:
            df: DataFrame
            file: 文件名
            default_label: 默认标签
            start_sample_index: 起始样本索引
        
        Returns:
            样本行列表, 样本数量
        """
        samples = []
        
        # 基本信息
        total_rows = len(df)
        label, is_abnormal = self.get_label_from_filename(file, default_label)
        max_samples = self.max_samples_per_abnormal_file if is_abnormal else self.max_samples_per_file
        samples_to_extract = min(max_samples, total_rows // self.sample_size)
        
        # 如果数据不足一个完整样本，返回空但不跳过文件
        if samples_to_extract == 0:
            print(f"文件 {file} 数据不足({total_rows}行)，无法提取完整样本(需要{self.sample_size}行)")
            return samples, 0
        
        sample_count = 0
        for sample_idx in range(samples_to_extract):
            # 计算起始位置
            start_idx = total_rows - (sample_idx + 1) * self.sample_size
            end_idx = total_rows - sample_idx * self.sample_size
            
            if start_idx < 0:
                continue
                
            split_df = df.iloc[start_idx:end_idx]
            channel_data = []
            
            for ch in self.channels:
                values = split_df[ch].dropna().tolist()[:self.sample_size]
                pooled_values = self.average_pooling(values)
                values_str = ','.join(map(str, pooled_values))
                channel_data.append(values_str)
                    
            if len(channel_data) == self.actual_channel_count:
                line = ':'.join(channel_data) + f":{label}\n"
                samples.append(line)
                
                # 记录样本-文件映射关系
                global_sample_index = start_sample_index + sample_count
                self.sample_to_file_mapping[global_sample_index] = {
                    'file': file,
                    'sample_in_file': sample_idx,
                    'label': label,
                    'is_abnormal': is_abnormal
                }
                
                sample_count += 1
        
        return samples, sample_count

    def save_sample_mapping(self, output_path):
        """
        保存样本-文件映射关系到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.sample_to_file_mapping, f, ensure_ascii=False, indent=2)
        print(f"样本映射关系已保存到: {output_path}")
    
    def load_sample_mapping(self, mapping_path):
        """
        从JSON文件加载样本-文件映射关系
        
        Args:
            mapping_path: 映射文件路径
        
        Returns:
            映射关系字典
        """
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        # 将键转换回整数
        return {int(k): v for k, v in mapping.items()}
    
    def generate_file_predictions(self, sample_predictions, mapping_path, output_path, threshold=0.5):
        """
        根据样本预测结果生成文件级别的预测结果
        
        Args:
            sample_predictions: 样本预测结果列表 (0或1)
            mapping_path: 样本-文件映射关系文件路径
            output_path: 输出文件路径
            threshold: 判断异常的阈值，默认0.5（多数投票）
        """
        # 加载映射关系
        mapping = self.load_sample_mapping(mapping_path)
        
        # 按文件聚合预测结果
        file_results = {}
        
        for sample_idx, prediction in enumerate(sample_predictions):
            if sample_idx in mapping:
                file_info = mapping[sample_idx]
                file_name = file_info['file']
                true_label = int(file_info['label']) if 'label' in file_info else None
                
                if file_name not in file_results:
                    file_results[file_name] = {
                        'true_label': true_label,
                        'predictions': [],
                        'is_abnormal': file_info.get('is_abnormal', False)
                    }
                
                file_results[file_name]['predictions'].append(prediction)
        
        # 计算每个文件的最终预测结果
        results = []
        for file_name, info in file_results.items():
            predictions = info['predictions']
            # 使用指定阈值决定文件标签
            abnormal_ratio = sum(predictions) / len(predictions)
            predicted_label = 1 if abnormal_ratio > threshold else 0
            
            result = {
                'file': file_name,
                'sample_count': len(predictions),
                'abnormal_sample_count': sum(predictions),
                'abnormal_sample_ratio': abnormal_ratio,
                'predicted_label': predicted_label,
                'is_abnormal_file': 'Yes' if predicted_label == 1 else 'No'
            }
            
            results.append(result)
        
        # 保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        
        # 统计信息
        total_files = len(results)
        abnormal_files = sum(1 for r in results if r['predicted_label'] == 1)
        
        print(f"文件总数: {total_files}, 预测异常文件: {abnormal_files}, 异常文件比例: {abnormal_files/total_files:.4f}")
        
        # 导出预测为故障的文件名到txt
        abnormal_file_names = [r['file'] for r in results if r['predicted_label'] == 1]
        abnormal_txt_path = output_path.replace('.csv', '_abnormal_files.txt')
        with open(abnormal_txt_path, 'w', encoding='utf-8') as f:
            for filename in abnormal_file_names:
                f.write(f"{filename}\n")
        print(f"预测为故障的文件共 {len(abnormal_file_names)} 个，已保存到: {abnormal_txt_path}")
        
        return results_df

    def create_header(self):
        """创建.ts文件头部"""
        return [
            "@relation 'EV'\n", 
            "@problemname 'EV'\n", 
            "@timestamps false\n", 
            "@missing false\n", 
            "@univariate false\n", 
            f"@dimension {self.actual_channel_count}\n", 
            "@equallength true\n", 
            f"@serieslength {self.expected_output_size}\n", 
            "@targetlabel true\n", 
            "@classlabel true 0 1\n", 
            "@data\n"
        ]
    
    def save_ts_file(self, samples, output_path):
        """
        保存样本到.ts文件
        
        Args:
            samples: 样本数据列表
            output_path: 输出文件路径
        """
        header = self.create_header()
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(header)
            f.writelines(samples)
    
    def count_labels(self, file_path):
        """
        统计.ts文件中的标签数量
        
        Args:
            file_path: .ts文件路径
        
        Returns:
            标签0数量, 标签1数量
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data_lines = [line for line in lines if not line.startswith('@')]
        label_0_count = sum(1 for line in data_lines if line.strip().endswith(':0'))
        label_1_count = sum(1 for line in data_lines if line.strip().endswith(':1'))
        return label_0_count, label_1_count
    
    def transform_train_test(self, data_folder, output_dir='all_datasets/EV', train_ratio=0.5):
        """
        从同一文件夹读取CSV文件，正常文件和故障文件分别随机分配给训练集和测试集
        
        Args:
            data_folder: 数据文件夹路径
            output_dir: 输出目录
            train_ratio: 训练集比例，默认0.5（一半一半）
        """
        # 清空映射关系
        self.sample_to_file_mapping = {}
        
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        
        if len(csv_files) == 0:
            return
        
        # 分离正常文件和故障文件
        normal_files = []
        abnormal_files = []
        
        for file in csv_files:
            _, is_abnormal = self.get_label_from_filename(file)
            if is_abnormal:
                abnormal_files.append(file)
            else:
                normal_files.append(file)
        
        print(f"正常文件数: {len(normal_files)}, 故障文件数: {len(abnormal_files)}")
        
        # 分别对正常文件和故障文件进行随机划分
        random.shuffle(normal_files)
        random.shuffle(abnormal_files)
        
        normal_train_count = int(len(normal_files) * train_ratio)
        abnormal_train_count = int(len(abnormal_files) * train_ratio)
        
        train_files = normal_files[:normal_train_count] + abnormal_files[:abnormal_train_count]
        test_files = normal_files[normal_train_count:] + abnormal_files[abnormal_train_count:]
        
        print(f"训练集: 正常{normal_train_count}个, 故障{abnormal_train_count}个")
        print(f"测试集: 正常{len(normal_files)-normal_train_count}个, 故障{len(abnormal_files)-abnormal_train_count}个")
        
        # 处理训练集
        train_lines = []
        train_sample_index = 0
        train_mapping = {}
        
        for file in train_files:
            csv_path = os.path.join(data_folder, file)
            df = pd.read_csv(csv_path)
            df = self.add_calculated_channels(df)
            file_samples, sample_count = self._process_single_file(df, file, start_sample_index=train_sample_index)
            train_lines.extend(file_samples)
            
            # 保存训练集的映射关系
            for i in range(sample_count):
                train_mapping[train_sample_index + i] = self.sample_to_file_mapping[train_sample_index + i]
            
            train_sample_index += sample_count
        
        # 处理测试集
        test_lines = []
        test_sample_index = 0
        test_mapping = {}
        self.sample_to_file_mapping = {}  # 重置映射
        
        for file in test_files:
            csv_path = os.path.join(data_folder, file)
            df = pd.read_csv(csv_path)
            df = self.add_calculated_channels(df)
            file_samples, sample_count = self._process_single_file(df, file, start_sample_index=test_sample_index)
            test_lines.extend(file_samples)
            
            # 保存测试集的映射关系
            for i in range(sample_count):
                test_mapping[test_sample_index + i] = self.sample_to_file_mapping[test_sample_index + i]
            
            test_sample_index += sample_count
        
        # 保存文件
        train_path = os.path.join(output_dir, 'EV_TRAIN.ts')
        test_path = os.path.join(output_dir, 'EV_TEST.ts')
        
        self.save_ts_file(train_lines, train_path)
        self.save_ts_file(test_lines, test_path)
        
        # 保存映射关系
        train_mapping_path = os.path.join(output_dir, 'train_sample_mapping.json')
        test_mapping_path = os.path.join(output_dir, 'test_sample_mapping.json')
        
        with open(train_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(train_mapping, f, ensure_ascii=False, indent=2)
        
        with open(test_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(test_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"训练集映射关系已保存到: {train_mapping_path}")
        print(f"测试集映射关系已保存到: {test_mapping_path}")
        
        # 统计结果
        train_label_0, train_label_1 = self.count_labels(train_path)
        test_label_0, test_label_1 = self.count_labels(test_path)
        
        print(f"TRAIN: {train_label_0 + train_label_1}样本 (0:{train_label_0}, 1:{train_label_1})")
        print(f"TEST: {test_label_0 + test_label_1}样本 (0:{test_label_0}, 1:{test_label_1})")
    
    def transform_predict(self, predict_folder, output_dir='all_datasets/EV', output_filename='EV_PRE.ts'):
        """
        转换预测数据集
        
        Args:
            predict_folder: 预测数据文件夹
            output_dir: 输出目录
            output_filename: 输出文件名
        """
        # 清空映射关系
        self.sample_to_file_mapping = {}
        
        predict_lines = self.process_csv_files(predict_folder, default_label='正常', save_samples=False)
        
        output_path = os.path.join(output_dir, output_filename)
        self.save_ts_file(predict_lines, output_path)
        
        # 保存预测集的映射关系
        predict_mapping_path = os.path.join(output_dir, 'predict_sample_mapping.json')
        self.save_sample_mapping(predict_mapping_path)
        
        predict_label_0, predict_label_1 = self.count_labels(output_path)
        print(f"{output_filename}: {predict_label_0 + predict_label_1}样本 (0:{predict_label_0}, 1:{predict_label_1})")