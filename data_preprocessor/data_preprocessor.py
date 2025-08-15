import os
import pandas as pd
import numpy as np
import shutil
from scipy import signal
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    数据预处理器：集成FFT扰动和数据预处理功能
    """
    
    def __init__(self, 
                 target_channels=None,
                 noise_level=0.08,
                 frequency_ratio=0.5,
                 wave_length_factor=1.2):
        """
        初始化数据预处理器
        
        Args:
            target_channels: 目标通道列表，默认使用标准配置
            noise_level: FFT噪声强度
            frequency_ratio: FFT扰动频率范围比例
            wave_length_factor: 波长因子
        """
        if target_channels is None:
            self.target_channels = [
                'SUM_VOLTAGE',    # 总电压
                'SUM_CURRENT',    # 总电流
                'SOC',            # 电池荷电状态
                'MAX_CELL_VOLT',  # 最大单体电压
                'MIN_CELL_VOLT',  # 最小单体电压
                'MAX_TEMP',       # 最高温度
                'MIN_TEMP',       # 最低温度
            ]
        else:
            self.target_channels = target_channels
        
        self.noise_level = noise_level
        self.frequency_ratio = frequency_ratio
        self.wave_length_factor = wave_length_factor
    
    def load_file_list_from_txt(self, txt_path):
        """
        从txt文件读取文件名列表
        
        Args:
            txt_path: txt文件路径，每行一个文件名
        
        Returns:
            文件名列表
        """
        file_list = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    file_list.append(line)
        return file_list
    
    def apply_fft_perturbation(self, signal):
        """
        对信号应用FFT频域扰动
        
        Args:
            signal: 输入信号（1D数组）
        
        Returns:
            perturbed_signal: 扰动后的信号
        """
        fft_signal = np.fft.fft(signal)
        n_samples = len(signal)
        
        # 保护基频
        protected_range = 2
        
        # 计算扰动范围
        low_freq_start = protected_range
        low_freq_range = max(protected_range + 1, int(n_samples * self.frequency_ratio / self.wave_length_factor))
        mid_freq_range = int(n_samples * self.frequency_ratio)
        
        # 生成扰动
        magnitude_perturbation = np.ones(n_samples)
        phase_perturbation = np.zeros(n_samples)
        
        # 低频段扰动
        low_freq_noise = self.noise_level * self.wave_length_factor
        if low_freq_range > low_freq_start:
            magnitude_perturbation[low_freq_start:low_freq_range] = np.random.normal(1.0, low_freq_noise, low_freq_range - low_freq_start)
            magnitude_perturbation[-(low_freq_range-low_freq_start):] = np.random.normal(1.0, low_freq_noise, low_freq_range - low_freq_start)
            phase_perturbation[low_freq_start:low_freq_range] = np.random.normal(0.0, low_freq_noise * np.pi, low_freq_range - low_freq_start)
            phase_perturbation[-(low_freq_range-low_freq_start):] = np.random.normal(0.0, low_freq_noise * np.pi, low_freq_range - low_freq_start)
        
        # 中频段扰动
        if mid_freq_range > low_freq_range:
            mid_freq_noise = self.noise_level
            magnitude_perturbation[low_freq_range:mid_freq_range] = np.random.normal(1.0, mid_freq_noise, mid_freq_range - low_freq_range)
            magnitude_perturbation[-(mid_freq_range-low_freq_range):] = np.random.normal(1.0, mid_freq_noise, mid_freq_range - low_freq_range)
            phase_perturbation[low_freq_range:mid_freq_range] = np.random.normal(0.0, mid_freq_noise * np.pi, mid_freq_range - low_freq_range)
            phase_perturbation[-(mid_freq_range-low_freq_range):] = np.random.normal(0.0, mid_freq_noise * np.pi, mid_freq_range - low_freq_range)
        
        # 应用扰动
        perturbed_fft = fft_signal * magnitude_perturbation * np.exp(1j * phase_perturbation)
        
        # 保护基频
        perturbed_fft[:protected_range] = fft_signal[:protected_range]
        if protected_range > 0:
            perturbed_fft[-protected_range:] = fft_signal[-protected_range:]
        
        # 逆FFT
        perturbed_signal = np.fft.ifft(perturbed_fft).real
        return perturbed_signal
    
    def generate_fft_variants(self, input_folder, file_list, n_variants=3, output_suffix="_fft"):
        """
        为指定文件生成FFT扰动变体
        
        Args:
            input_folder: 输入文件夹路径
            file_list: 要处理的文件名列表
            n_variants: 每个文件生成的变体数量
            output_suffix: 输出文件后缀
        
        Returns:
            生成成功的文件数量
        """
        success_count = 0
        
        # 使用进度条显示处理进度
        with tqdm(file_list, desc="生成FFT扰动变体", unit="文件") as pbar:
            for filename in pbar:
                input_path = os.path.join(input_folder, filename)
                
                # 更新进度条描述，显示当前处理的文件
                pbar.set_postfix_str(f"正在处理: {filename}")
                
                df = pd.read_csv(input_path)
                    
                # 生成n个变体
                for i in range(n_variants):
                    df_variant = df.copy()
                        
                    # 对每个目标通道应用FFT扰动
                    for channel in self.target_channels:
                        if channel in df_variant.columns:
                            original_signal = df_variant[channel].values
                            perturbed_signal = self.apply_fft_perturbation(original_signal)
                            df_variant[channel] = perturbed_signal
                        
                    # 保存变体文件到同一文件夹
                    name, ext = os.path.splitext(filename)
                    variant_filename = f"{name}{output_suffix}_{i+1}{ext}"
                    output_path = os.path.join(input_folder, variant_filename)
                    df_variant.to_csv(output_path, index=False)
                    
                success_count += 1
        
        print(f"FFT扰动完成，共处理 {success_count} 个文件，生成 {success_count * n_variants} 个变体")
        return success_count
    
    def remove_outliers_vectorized(self, data, window_size=5, threshold=3.0):
        """
        使用向量化操作高效去除数据中的毛刺（异常值）
        
        Args:
            data: 输入数据序列
            window_size: 滑动窗口大小
            threshold: 异常值阈值（标准差倍数）
        
        Returns:
            处理后的数据序列
        """
        data = np.array(data, dtype=float)
        n = len(data)
        cleaned_data = data.copy()
        
        # 如果数据太短，直接返回
        if n < window_size:
            return cleaned_data
        
        # 使用pandas rolling操作进行向量化计算
        df_temp = pd.DataFrame({'data': data})
        
        # 计算滚动平均和标准差
        rolling_mean = df_temp['data'].rolling(window=window_size, center=True, min_periods=1).mean()
        rolling_std = df_temp['data'].rolling(window=window_size, center=True, min_periods=1).std()
        
        # 找到异常值的位置
        outlier_mask = (rolling_std > 0) & (np.abs(data - rolling_mean) > threshold * rolling_std)
        
        if np.any(outlier_mask):
            # 计算滚动中位数来替换异常值
            rolling_median = df_temp['data'].rolling(window=window_size, center=True, min_periods=1).median()
            cleaned_data[outlier_mask] = rolling_median[outlier_mask]
        
        return cleaned_data

    def remove_outliers_chunked(self, data, window_size=5, threshold=3.0, chunk_size=50000):
        """
        分块处理超长序列的异常值去除，适用于超长样本
        
        Args:
            data: 输入数据序列
            window_size: 滑动窗口大小
            threshold: 异常值阈值（标准差倍数）
            chunk_size: 分块大小
        
        Returns:
            处理后的数据序列
        """
        data = np.array(data, dtype=float)
        n = len(data)
        
        # 如果数据长度小于chunk_size，直接使用向量化方法
        if n <= chunk_size:
            return self.remove_outliers_vectorized(data, window_size, threshold)
        
        cleaned_data = np.empty_like(data)
        overlap = window_size  # 重叠区域大小
        
        # 分块处理
        for start in range(0, n, chunk_size - overlap):
            end = min(start + chunk_size, n)
            
            # 提取当前块（包含重叠区域）
            chunk = data[start:end]
            
            # 处理当前块
            cleaned_chunk = self.remove_outliers_vectorized(chunk, window_size, threshold)
            
            # 将结果写入输出数组
            if start == 0:
                # 第一个块，完整写入
                write_end = min(chunk_size - overlap//2, len(cleaned_chunk))
                cleaned_data[start:start + write_end] = cleaned_chunk[:write_end]
            elif end == n:
                # 最后一个块，从重叠区域中间开始写入
                write_start = overlap // 2
                cleaned_data[start + write_start:end] = cleaned_chunk[write_start:]
            else:
                # 中间块，只写入中间部分避免重叠
                write_start = overlap // 2
                write_end = len(cleaned_chunk) - overlap // 2
                cleaned_data[start + write_start:start + write_end] = cleaned_chunk[write_start:write_end]
        
        return cleaned_data

    def remove_outliers(self, data, window_size=5, threshold=3.0):
        """
        智能选择最优的异常值去除方法
        
        Args:
            data: 输入数据序列
            window_size: 滑动窗口大小
            threshold: 异常值阈值（标准差倍数）
        
        Returns:
            处理后的数据序列
        """
        data = np.array(data, dtype=float)
        n = len(data)
        
        # 根据数据长度选择处理策略
        if n < 1000:
            # 短序列使用原始方法
            return self._remove_outliers_original(data, window_size, threshold)
        elif n < 100000:
            # 中等长度使用向量化方法
            return self.remove_outliers_vectorized(data, window_size, threshold)
        else:
            # 超长序列使用分块方法
            return self.remove_outliers_chunked(data, window_size, threshold)

    def _remove_outliers_original(self, data, window_size=5, threshold=3.0):
        """
        原始的循环方法（保留用于短序列）
        """
        data = np.array(data, dtype=float)
        cleaned_data = data.copy()
        
        # 使用滑动窗口方法
        for i in range(len(data)):
            # 定义窗口范围
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            window = data[start:end]
            
            # 计算窗口统计量
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            # 判断是否为异常值
            if window_std > 0 and abs(data[i] - window_mean) > threshold * window_std:
                # 用窗口中位数替换异常值
                cleaned_data[i] = np.median(window)
        
        return cleaned_data
    
    def _process_single_file(self, args):
        """
        处理单个文件的工作函数（用于并行处理）
        
        Args:
            args: (input_path, output_path, target_channels, window_size, threshold)
        
        Returns:
            (filename, success, error_msg)
        """
        input_path, output_path, target_channels, window_size, threshold = args
        filename = os.path.basename(input_path)
        
        try:
            df = pd.read_csv(input_path)
            
            # 对每个目标通道进行去毛刺处理
            for channel in target_channels:
                if channel in df.columns:
                    cleaned_data = self.remove_outliers(
                        df[channel].values, 
                        window_size=window_size, 
                        threshold=threshold
                    )
                    df[channel] = cleaned_data
            
            # 保存处理后的文件
            df.to_csv(output_path, index=False)
            return (filename, True, None)
            
        except Exception as e:
            return (filename, False, str(e))

    def denoise_folder_parallel(self, folder_path, window_size=6, threshold=1.5, output_suffix="_clean", n_processes=None):
        """
        使用多进程并行处理文件夹中所有CSV文件的去毛刺
        
        Args:
            folder_path: 文件夹路径
            window_size: 滑动窗口大小
            threshold: 异常值阈值
            output_suffix: 输出文件夹后缀
            n_processes: 进程数，None表示使用所有CPU核心
        
        Returns:
            处理成功的文件数量
        """
        if n_processes is None:
            n_processes = cpu_count()
        
        # 创建输出文件夹
        parent_dir = os.path.dirname(folder_path)
        folder_name = os.path.basename(folder_path)
        output_folder = os.path.join(parent_dir, f"{folder_name}{output_suffix}")
        os.makedirs(output_folder, exist_ok=True)
        
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"文件夹 {folder_path} 中没有找到CSV文件")
            return 0
        
        # 准备并行处理参数
        process_args = []
        for filename in csv_files:
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            process_args.append((input_path, output_path, self.target_channels, window_size, threshold))
        
        success_count = 0
        failed_files = []
        
        print(f"使用 {n_processes} 个进程并行处理 {len(csv_files)} 个文件...")
        
        # 使用进程池进行并行处理
        with Pool(processes=n_processes) as pool:
            # 使用imap获取结果并显示进度
            results = list(tqdm(
                pool.imap(self._process_single_file, process_args),
                total=len(process_args),
                desc=f"并行处理 {os.path.basename(folder_path)}",
                unit="文件"
            ))
        
        # 统计结果
        for filename, success, error_msg in results:
            if success:
                success_count += 1
            else:
                failed_files.append((filename, error_msg))
        
        # 输出结果
        print(f"并行去毛刺处理完成:")
        print(f"  - 成功处理: {success_count} 个文件")
        print(f"  - 失败: {len(failed_files)} 个文件")
        print(f"  - 输出文件夹: {output_folder}")
        
        if failed_files:
            print("失败的文件:")
            for filename, error in failed_files[:5]:  # 只显示前5个错误
                print(f"  - {filename}: {error}")
            if len(failed_files) > 5:
                print(f"  ... 还有 {len(failed_files) - 5} 个文件处理失败")
        
        return success_count

    def denoise_folder(self, folder_path, window_size=6, threshold=1.5, output_suffix="_clean", use_parallel=True, n_processes=None):
        """
        对文件夹中所有CSV文件进行去毛刺处理
        
        Args:
            folder_path: 文件夹路径
            window_size: 滑动窗口大小
            threshold: 异常值阈值
            output_suffix: 输出文件夹后缀
            use_parallel: 是否使用并行处理
            n_processes: 进程数（仅在use_parallel=True时有效）
        
        Returns:
            处理成功的文件数量
        """
        if use_parallel:
            return self.denoise_folder_parallel(folder_path, window_size, threshold, output_suffix, n_processes)
        else:
            return self._denoise_folder_sequential(folder_path, window_size, threshold, output_suffix)

    def _denoise_folder_sequential(self, folder_path, window_size=6, threshold=1.5, output_suffix="_clean"):
        """
        顺序处理文件夹中所有CSV文件进行去毛刺处理
        
        Args:
            folder_path: 文件夹路径
            window_size: 滑动窗口大小
            threshold: 异常值阈值
            output_suffix: 输出文件夹后缀
        
        Returns:
            处理成功的文件数量
        """
        # 创建输出文件夹：原文件夹名 + 后缀
        parent_dir = os.path.dirname(folder_path)
        folder_name = os.path.basename(folder_path)
        output_folder = os.path.join(parent_dir, f"{folder_name}{output_suffix}")
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        success_count = 0
        
        # 使用进度条显示处理进度
        with tqdm(csv_files, desc=f"顺序处理 {os.path.basename(folder_path)}", unit="文件") as pbar:
            for filename in pbar:
                input_path = os.path.join(folder_path, filename)
                
                # 更新进度条描述，显示当前处理的文件
                pbar.set_postfix_str(f"正在处理: {filename}")
                
                df = pd.read_csv(input_path)
                
                # 对每个目标通道进行去毛刺处理
                for channel in self.target_channels:
                    if channel in df.columns:
                        cleaned_data = self.remove_outliers(
                            df[channel].values, 
                            window_size=window_size, 
                            threshold=threshold
                        )
                        df[channel] = cleaned_data
                
                # 保存处理后的文件到输出文件夹
                output_path = os.path.join(output_folder, filename)
                df.to_csv(output_path, index=False)
                
                success_count += 1
        
        print(f"顺序去毛刺处理完成，共处理 {success_count} 个文件，保存到: {output_folder}")
        return success_count
    
    def denoise_multiple_folders(self, folder_paths, window_size=6, threshold=1.5, output_suffix="_clean"):
        """
        对多个文件夹中的所有CSV文件进行去毛刺处理
        
        Args:
            folder_paths: 文件夹路径列表
            window_size: 滑动窗口大小
            threshold: 异常值阈值
            output_suffix: 输出文件后缀
        
        Returns:
            每个文件夹处理成功的文件数量字典
        """
        if isinstance(folder_paths, str):
            folder_paths = [folder_paths]
            
        results = {}
        total_files = 0
        
        # 使用进度条显示文件夹处理进度
        with tqdm(folder_paths, desc="处理多个文件夹", unit="文件夹") as folder_pbar:
            for folder_path in folder_pbar:
                folder_pbar.set_postfix_str(f"正在处理: {os.path.basename(folder_path)}")
                
                success_count = self.denoise_folder(folder_path, window_size, threshold, output_suffix)
                results[folder_path] = success_count
                total_files += success_count
                
                # 更新总体进度
                folder_pbar.set_postfix_str(f"已完成: {os.path.basename(folder_path)} ({success_count}个文件)")
        
        print(f"\n所有文件夹去毛刺处理完成，共处理 {total_files} 个文件")
        print("创建的输出文件夹:")
        for folder_path in folder_paths:
            parent_dir = os.path.dirname(folder_path)
            folder_name = os.path.basename(folder_path)
            output_folder = os.path.join(parent_dir, f"{folder_name}{output_suffix}")
            print(f"  - {output_folder}")
        return results
