from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import numpy as np
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import csv

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _calculate_f1_score(self, y_true, y_pred):
        """
        简单的F1 score计算函数（当sklearn不可用时）
        """
        # 获取唯一类别
        classes = np.unique(np.concatenate([y_true, y_pred]))
        f1_scores = []
        
        for cls in classes:
            # 计算每个类别的精确度和召回率
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
                
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)
                
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
                
            f1_scores.append(f1)
        
        # 返回宏平均F1 score
        return np.mean(f1_scores)

    def _calculate_class_f1_score(self, y_true, y_pred, target_class=1):
        """
        计算特定类别的F1 score
        """
        return f1_score(y_true, y_pred, labels=[target_class], average= 'binary')

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        
        # 计算类别1的F1分数
        f1_class1 = f1_score(trues, predictions, labels=[1], average='binary')

        self.model.train()
        return total_loss, accuracy, f1_class1

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        # 创建plot文件夹
        plot_path = os.path.join('./plots', setting)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        # 用于记录训练过程
        train_losses = []
        vali_losses = []
        vali_f1_class1_list = []
        epochs_list = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy, val_f1_class1 = self.vali(vali_data, vali_loader, criterion)

            # 记录训练数据
            epochs_list.append(epoch + 1)
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            vali_f1_class1_list.append(val_f1_class1)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Vali F1: {5:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, val_f1_class1))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 10 == 0:
                adjust_learning_rate(model_optim,  scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 绘制训练曲线
        self._plot_training_curves(epochs_list, train_losses, vali_losses, vali_f1_class1_list, plot_path)

        return self.model

    def test(self, setting, test=0, threshold=0.5):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        
        accuracy = cal_accuracy(predictions, trues)
        
        # 计算F1 score
        f1_class1 = f1_score(trues, predictions, labels=[1], average='binary')
            
        # 创建plot文件夹
        plot_path = os.path.join('./plots', setting)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        # 详细分类报告
        class_report = classification_report(trues, predictions)
        print("Classification Report:")
        print(class_report)
        
        print('accuracy: {:.4f}'.format(accuracy))
        print('F1 Score: {:.4f}'.format(f1_class1))

        # 绘制混淆矩阵
        class_names = ['Normal', 'Abnormal']  # 根据你的实际类别名称调整
        self._plot_confusion_matrix(trues, predictions, class_names, plot_path, setting)
        
        # 文件级别验证
        self._analyze_file_level_predictions(predictions, setting, plot_path, 'test', threshold)

        
        return accuracy, f1_class1

    def predict(self, setting, data_path='EV_PRE.ts', threshold=0.3):
        """
        使用训练好的模型对新数据进行预测
        Args:
            setting: 模型设置名称
            data_path: 要预测的数据文件路径，默认为 'EV_PRE.ts'
            threshold: 判断异常的阈值，默认0.3
        """
        print(f'开始预测数据: {data_path}')
        
        # 加载训练好的模型
        model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        
        print('加载模型...')
        self.model.load_state_dict(torch.load(model_path))
        
        # 构建完整的数据文件路径
        predict_root_path = self.args.root_path
        full_data_path = os.path.join(predict_root_path, data_path)
        
        print(f'加载预测数据: {full_data_path}')
        
        # 创建预测数据集 - 使用UEAloader加载指定的.ts文件
        from data_provider.data_loader import UEAloader
        predict_dataset = UEAloader(
            root_path=predict_root_path,
            file_list=[os.path.basename(data_path)],  # 只加载指定的文件
            limit_size=None,
            flag=None
        )
        
        print(f'预测数据集大小: {len(predict_dataset)}')
        print(f'序列长度: {predict_dataset.max_seq_len}')
            
        # 创建数据加载器 (不打乱数据顺序)
        from torch.utils.data import DataLoader
        from data_provider.uea import collate_fn
            
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
            collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len)
        )
            
        # 进行预测
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(predict_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                    
                outputs = self.model(batch_x, padding_mask, None, None)
                preds.append(outputs.detach())
            
        # 整合预测结果
        preds = torch.cat(preds, 0)
            
        # 计算概率和预测标签
        probabilities = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
        predictions = torch.argmax(preds, dim=1).cpu().numpy()
            
        # 创建结果保存文件夹
        predict_folder = './predict_results/' + setting + '/'
        if not os.path.exists(predict_folder):
            os.makedirs(predict_folder)
            
            
        # 输出统计信息
        normal_count = np.sum(predictions == 0)
        abnormal_count = np.sum(predictions == 1)
        total_count = len(predictions)
            
        print("\n预测结果统计:")
        print("=" * 50)
        print(f"总样本数: {total_count}")
        print(f"正常样本数: {normal_count} ({normal_count/total_count*100:.2f}%)")
        print(f"异常样本数: {abnormal_count} ({abnormal_count/total_count*100:.2f}%)")
        
        # 添加文件级别预测分析

        self._analyze_file_level_predictions(predictions, setting, predict_folder, 'predict', threshold)

        return predictions, probabilities

    def _plot_training_curves(self, epochs, train_losses, vali_losses, vali_f1_class1, plot_path):
        """
        绘制训练曲线
        """            
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
            
        # 创建子图
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
        # 绘制Loss曲线
        axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, vali_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
            
        # 绘制类别1的F1分数曲线
        axes[1].plot(epochs, vali_f1_class1, 'g-', label='Validation F1 (Class 1)', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('F1 Score for Class 1 (Abnormal)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存到: {os.path.join(plot_path, 'training_curves.png')}")

    def _plot_confusion_matrix(self, y_true, y_pred, class_names, plot_path, setting):
        """
        绘制混淆矩阵
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
            
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算比例矩阵（个数/总数）
        total = np.sum(cm)
        cm_proportion = cm / total
            
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
            
        # 创建图形
        plt.figure(figsize=(8, 6))
            
        # 绘制热力图，使用比例数据，但标注显示原始数量和比例
        annot_labels = np.array([[f'{cm[i,j]}\n({cm_proportion[i,j]:.3f})' 
                                 for j in range(cm.shape[1])] 
                                 for i in range(cm.shape[0])])
        
        sns.heatmap(cm_proportion, annot=annot_labels, fmt='', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Proportion'})
            
        plt.title(f'Confusion Matrix - {setting}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
            
        # 添加统计信息
        f1_class1 = self._calculate_class_f1_score(y_true, y_pred, target_class=1)
        plt.figtext(0.02, 0.02, f'F1 score: {f1_class1:.4f}', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵已保存到: {os.path.join(plot_path, 'confusion_matrix.png')}")

    def _analyze_file_level_predictions(self, predictions, setting, output_folder, data_type='test', threshold=0.5):
        """
        分析文件级别的预测结果
        
        Args:
            predictions: 样本预测结果数组
            setting: 模型设置名称
            output_folder: 输出文件夹路径
            data_type: 数据类型 ('test' 或 'predict')
            threshold: 判断异常的阈值，默认0.5
        """
        from data_transform.data_transform import EVDataTransformer
        
        # 构建映射文件路径 - 使用root_path而不是data_path
        mapping_filename = f'{data_type}_sample_mapping.json'
        if hasattr(self.args, 'root_path'):
            mapping_path = os.path.join(self.args.root_path, mapping_filename)
        else:
            # 如果没有root_path，尝试从data_path获取目录部分
            data_dir = os.path.dirname(self.args.data_path) if self.args.data_path else '.'
            mapping_path = os.path.join(data_dir, mapping_filename)
        
        # 创建数据转换器实例
        transformer = EVDataTransformer()
        
        # 生成文件级别预测结果
        file_results_path = os.path.join(output_folder, f'file_level_predictions_{data_type}.csv')
        results_df = transformer.generate_file_predictions(
            sample_predictions=predictions.tolist(),
            mapping_path=mapping_path,
            output_path=file_results_path,
            threshold=threshold
        )
        
        print(f"文件级别预测结果已保存到: {file_results_path}")
