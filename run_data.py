#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from data_preprocessor.data_preprocessor import DataPreprocessor
from data_transform.data_transform import EVDataTransformer

def main():
    parser = argparse.ArgumentParser(description='数据预处理脚本：FFT扰动和去毛刺处理')
    
    # 通用参数
    parser.add_argument('--target_channels', nargs='+', 
                        default=['SUM_VOLTAGE', 'SUM_CURRENT', 'SOC', 'MAX_CELL_VOLT', 'MIN_CELL_VOLT', 'MAX_TEMP', 'MIN_TEMP'],
                        help='目标处理通道列表')
    
    # FFT扰动相关参数
    fft_group = parser.add_argument_group('FFT扰动参数')
    fft_group.add_argument('--fft', action='store_true', help='启用FFT扰动功能')
    fft_group.add_argument('--input_folder', type=str, default= 'all_datasets/train_datasets', help='FFT扰动输入文件夹路径')
    fft_group.add_argument('--target_files_txt', type=str, default= 'data_preprocessor/fft_list.txt', help='目标文件列表txt文件路径')
    fft_group.add_argument('--n_variants', type=int, default=3, help='每个文件生成的FFT变体数量')
    fft_group.add_argument('--noise_level', type=float, default=0.1, help='FFT噪声强度')
    fft_group.add_argument('--frequency_ratio', type=float, default=0.6, help='FFT扰动频率范围比例')
    fft_group.add_argument('--wave_length_factor', type=float, default=1.2, help='波长因子')
    fft_group.add_argument('--fft_output_suffix', type=str, default='_fft', help='FFT输出文件后缀')
    
    # 去毛刺相关参数
    denoise_group = parser.add_argument_group('去毛刺参数')
    denoise_group.add_argument('--denoise', action='store_true', help='启用去毛刺功能')
    denoise_group.add_argument('--denoise_folder', nargs='+', default=['all_datasets/train_datasets','all_datasets/predict_datasets'], help='去毛刺处理文件夹路径（可以是多个文件夹）')
    denoise_group.add_argument('--window_size', type=int, default=6, help='去毛刺滑动窗口大小')
    denoise_group.add_argument('--threshold', type=float, default=1.5, help='去毛刺异常值阈值')
    denoise_group.add_argument('--denoise_output_suffix', type=str, default='_clean', help='去毛刺输出文件夹后缀')
    
    # 数据转换相关参数
    transform_group = parser.add_argument_group('数据转换参数')
    transform_group.add_argument('--transform', action='store_true', help='启用数据转换功能')
    transform_group.add_argument('--data_folder', type=str, default='all_datasets/train_datasets_clean', help='原始CSV数据文件夹路径')
    transform_group.add_argument('--output_dir', type=str, default='all_datasets/EV', help='转换后数据输出目录')
    transform_group.add_argument('--train_ratio', type=float, default=0.5, help='训练集比例')
    transform_group.add_argument('--abnormal_prefixes_file', type=str, default='all_datasets/abnormal_list.txt', help='异常文件前缀txt文件路径')
    transform_group.add_argument('--predict_folder', type=str, default='all_datasets/predict_datasets_clean', help='预测数据文件夹路径')
    transform_group.add_argument('--predict_filename', type=str, default='EV_PRE.ts', help='预测数据输出文件名')
    transform_group.add_argument('--sample_size', type=int, default=20000, help='每个样本使用的时间点数')
    transform_group.add_argument('--pool_size', type=int, default=10, help='池化窗口大小')
    transform_group.add_argument('--max_samples_per_file', type=int, default=10, help='每个正常文件最多提取的样本数')
    transform_group.add_argument('--max_samples_per_abnormal_file', type=int, default=10, help='每个异常文件最多提取的样本数')
    
    args = parser.parse_args()
    
    # 创建数据预处理器实例
    preprocessor = DataPreprocessor(
        target_channels=args.target_channels,
        noise_level=args.noise_level,
        frequency_ratio=args.frequency_ratio,
        wave_length_factor=args.wave_length_factor
    )
    
    # 创建数据转换器实例
    if args.transform:
        transformer = EVDataTransformer(
            sample_size=args.sample_size,
            pool_size=args.pool_size,
            max_samples_per_file=args.max_samples_per_file,
            max_samples_per_abnormal_file=args.max_samples_per_abnormal_file,
            abnormal_prefixes_file=args.abnormal_prefixes_file
        )
    
    print("数据预处理脚本启动")
    print(f"目标通道: {args.target_channels}")
    print("="*60)
    
    if args.fft:
        print("执行FFT扰动处理...")
        file_list = preprocessor.load_file_list_from_txt(args.target_files_txt)
        print(f"从 {args.target_files_txt} 读取到 {len(file_list)} 个文件")
        
        success_count = preprocessor.generate_fft_variants(
            input_folder=args.input_folder,
            file_list=file_list,
            n_variants=args.n_variants,
            output_suffix=args.fft_output_suffix
        )
        
        print(f"FFT扰动完成，成功处理 {success_count} 个文件")
    
    if args.denoise:
        print("执行去毛刺处理...")
        
        if len(args.denoise_folder) == 1:
            # 单个文件夹
            success_count = preprocessor.denoise_folder(
                folder_path=args.denoise_folder[0],
                window_size=args.window_size,
                threshold=args.threshold,
                output_suffix=args.denoise_output_suffix
            )
            print(f"去毛刺处理完成，成功处理 {success_count} 个文件")
        else:
            # 多个文件夹
            results = preprocessor.denoise_multiple_folders(
                folder_paths=args.denoise_folder,
                window_size=args.window_size,
                threshold=args.threshold,
                output_suffix=args.denoise_output_suffix
            )
            total_count = sum(results.values())
            print(f"多文件夹去毛刺处理完成，共成功处理 {total_count} 个文件")
    
    if args.transform:
        print("执行数据转换...")
        transformer._print_config()

        if args.data_folder:
            print("转换训练和测试数据...")
            transformer.transform_train_test(
                data_folder=args.data_folder,
                output_dir=args.output_dir,
                train_ratio=args.train_ratio
            )
            print("训练和测试数据转换完成")
        
        if args.predict_folder:
            print("转换预测数据...")
            transformer.transform_predict(
                predict_folder=args.predict_folder,
                output_dir=args.output_dir,
                output_filename=args.predict_filename
            )
            print("预测数据转换完成")

    
    print("="*60)
    print("数据预处理完成")

if __name__ == "__main__":
    main()
