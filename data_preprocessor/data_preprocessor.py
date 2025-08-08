import os
import pandas as pd
import numpy as np
import shutil
from scipy import signal
import random
from tqdm import tqdm


class DataPreprocessor:
    """
    数据预处理器：集成FFT扰动和数据预处理功能
    """
    
    def __init__(self, 
                 target_channels=None,
                 noise_level=0.2,
                 frequency_ratio=0.8,
                 wave_length_factor=1.5):
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
    
    def remove_outliers(self, data, window_size=5, threshold=3.0):
        """
        去除数据中的毛刺（异常值）
        
        Args:
            data: 输入数据序列
            window_size: 滑动窗口大小
            threshold: 异常值阈值（标准差倍数）
        
        Returns:
            处理后的数据序列
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
    
    def denoise_folder(self, folder_path, window_size=6, threshold=1.5, output_suffix="_clean"):
        """
        对文件夹中所有CSV文件进行去毛刺处理
        
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
        with tqdm(csv_files, desc=f"处理文件夹 {os.path.basename(folder_path)}", unit="文件") as pbar:
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
        
        print(f"去毛刺处理完成，共处理 {success_count} 个文件，保存到: {output_folder}")
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
