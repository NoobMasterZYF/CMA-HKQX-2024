# ============================================================================
# TV-L1全样本_UNet.py
# 基于原 TV-L1全样本.py，将 Farneback 光流法替换为 U-Net 光流法。
# U-Net 需要预训练：从数据集中选取连续帧对，按 9:1 划分训练集/验证集，
# 使用 Farneback 光流作为伪标签进行监督预训练。
# ============================================================================

import os
import re
import glob
import random
import psutil
import logging
import multiprocessing
import copy
import itertools
import sys
import ast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, accuracy_score, roc_auc_score, average_precision_score
)
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification
from PIL import Image
import cv2
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ============================================================================
# 配置
# ============================================================================
ROOT_DIR = 'D:\\CMA-HKQX-2024'
info_excel = os.path.join(ROOT_DIR, 'dataset-for-training', 'infomation.xlsx')
label_dir = os.path.join(ROOT_DIR, 'GHA-SCW-Datasets', 'Label')
base_dir = os.path.join(ROOT_DIR, 'dataset-for-training')
save_path = os.path.join(base_dir, 'saved_dataset.npz')
output_dir = os.path.join(base_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

time_steps = 6
forecast_steps = 3
img_size_radar = (400, 400)
img_size_satellite = (200, 200)
num_angles = 15
channels_satellite = 3
awos_features_dim = 9
batch_size = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# U-Net 预训练配置
UNET_PRETRAIN_MODEL_PATH = os.path.join(base_dir, 'unet_optical_flow.pth')
UNET_PRETRAIN_DIR = os.path.join(base_dir, 'unet_pretrain')
os.makedirs(UNET_PRETRAIN_DIR, exist_ok=True)
UNET_PRETRAIN_EPOCHS = 50
UNET_PRETRAIN_LR = 1e-4
UNET_PRETRAIN_BATCH_SIZE = 4
UNET_PRETRAIN_PATIENCE = 10


# ============================================================================
# U-Net 光流模型
# ============================================================================
class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FlowUNet(nn.Module):
    """
    U-Net 光流估计模型。
    输入：两张连续雷达帧（stacked, 2 通道）
    输出：光流场 (u, v) — (batch, 2, H, W)
    """
    def __init__(self, in_channels=2, base_ch=32):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 2, kernel_size=1)

    def forward(self, x):
        # x: (B, 2, H, W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self._pad_and_concat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._pad_and_concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_and_concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_and_concat(d1, e1)
        d1 = self.dec1(d1)

        flow = self.out_conv(d1)
        return flow

    @staticmethod
    def _pad_and_concat(up, skip):
        """Pad up tensor to match skip tensor spatial dims, then concat."""
        diff_h = skip.size(2) - up.size(2)
        diff_w = skip.size(3) - up.size(3)
        up = F.pad(up, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])
        return torch.cat([up, skip], dim=1)


# ============================================================================
# U-Net 光流预训练
# ============================================================================
def extract_flow_training_pairs(base_dir, info_excel_path, max_pairs_per_date=50):
    """
    从雷达数据中提取连续帧对，用于 U-Net 预训练。
    返回：pairs = [(frame_t, frame_t+1), ...] 每个 shape (400, 400)
    """
    pairs = []
    info_df = pd.read_excel(info_excel_path)
    dates = info_df['filename'].astype(str).tolist()

    for date_str in dates:
        radar_dir = os.path.join(base_dir, date_str, 'radar_img')
        if not os.path.exists(radar_dir):
            continue

        # 收集该日期所有时间戳和对应雷达文件
        ts_files = {}
        pattern = re.compile(r'(\d{8}_\d{6})_(\d+(?:\.\d+)?)_50kM\.jpg', re.IGNORECASE)
        for fname in sorted(os.listdir(radar_dir)):
            match = pattern.match(fname)
            if match:
                ts = match.group(1)
                angle = float(match.group(2))
                int_angle = min(max(int(round(angle)), 0), num_angles - 1)
                if int_angle == 2:  # 第3仰角
                    ts_files.setdefault(ts, []).append(os.path.join(radar_dir, fname))

        # 排序时间戳并提取连续帧对
        sorted_ts = sorted(ts_files.keys())
        count = 0
        for i in range(len(sorted_ts) - 1):
            if count >= max_pairs_per_date:
                break
            t1, t2 = sorted_ts[i], sorted_ts[i + 1]
            # 只取时间间隔为10分钟的连续帧对
            dt1 = datetime.strptime(t1, '%Y%m%d_%H%M%S')
            dt2 = datetime.strptime(t2, '%Y%m%d_%H%M%S')
            if abs((dt2 - dt1).total_seconds() - 600) <= 60:  # 10min ± 1min
                if ts_files[t1] and ts_files[t2]:
                    try:
                        img1 = np.array(Image.open(ts_files[t1][0]).convert('L')) / 255.0
                        img2 = np.array(Image.open(ts_files[t2][0]).convert('L')) / 255.0
                        if img1.shape == (400, 400) and img2.shape == (400, 400):
                            pairs.append((img1.astype(np.float32), img2.astype(np.float32)))
                            count += 1
                    except Exception:
                        continue

    print(f"提取 {len(pairs)} 对连续帧用于 U-Net 预训练")
    return pairs


def compute_farneback_pseudo_label(img1, img2):
    """
    使用 Farneback 计算伪标签光流。
    输入: (H, W) float32 [0, 1]
    输出: (2, H, W) float32
    """
    flow = cv2.calcOpticalFlowFarneback(
        (img1 * 255).astype(np.uint8),
        (img2 * 255).astype(np.uint8),
        None,
        pyr_scale=0.5, levels=4, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return np.transpose(flow, (2, 0, 1)).astype(np.float32)


def warp_image(img, flow):
    """
    使用光流对图像进行 warp。
    img: (B, 1, H, W) 或 (B, H, W)
    flow: (B, 2, H, W)
    返回 warped image，与 img 同 shape
    """
    if img.dim() == 3:
        img = img.unsqueeze(1)
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=torch.float32),
        torch.arange(W, device=img.device, dtype=torch.float32),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    # Normalize flow to [-1, 1] range for grid_sample
    flow_norm = flow.clone()
    flow_norm[:, 0] = flow_norm[:, 0] * 2.0 / (W - 1)
    flow_norm[:, 1] = flow_norm[:, 1] * 2.0 / (H - 1)
    sample_grid = grid + flow_norm.permute(0, 2, 3, 1)
    warped = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped


class PhotometricLoss(nn.Module):
    """光度损失：warp 后图像与目标图像的 L1 差"""
    def __init__(self):
        super().__init__()

    def forward(self, img1, img2, flow):
        warped = warp_image(img1, flow)
        return F.l1_loss(warped, img2)


class SmoothnessLoss(nn.Module):
    """光流平滑正则化"""
    def __init__(self):
        super().__init__()

    def forward(self, flow):
        # flow: (B, 2, H, W)
        du_dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
        du_dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
        return du_dx + du_dy


def pretrain_flow_unet():
    """
    预训练 U-Net 光流模型。
    使用 Farneback 光流作为伪标签进行监督训练，
    同时加入光度损失和平滑正则化。
    训练集:验证集 = 9:1
    """
    print("=" * 60)
    print("开始 U-Net 光流预训练...")
    print("=" * 60)

    # 如果已有预训练模型，直接加载
    if os.path.exists(UNET_PRETRAIN_MODEL_PATH):
        print(f"加载已有预训练模型: {UNET_PRETRAIN_MODEL_PATH}")
        model = FlowUNet(in_channels=2, base_ch=32).to(device)
        model.load_state_dict(torch.load(UNET_PRETRAIN_MODEL_PATH))
        model.eval()
        return model

    # 提取训练数据对
    pairs = extract_flow_training_pairs(base_dir, info_excel, max_pairs_per_date=50)
    if len(pairs) < 20:
        print(f"警告: 训练对不足 ({len(pairs)}), 使用 Farneback 作为 fallback")
        return None

    # 计算 Farneback 伪标签
    print("计算 Farneback 伪标签...")
    all_inputs = []
    all_targets = []

    for i, (img1, img2) in enumerate(pairs):
        flow_pseudo = compute_farneback_pseudo_label(img1, img2)
        stack = np.stack([img1, img2], axis=0)  # (2, 400, 400)
        all_inputs.append(stack)
        all_targets.append(flow_pseudo)

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(pairs)} 对")

    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    print(f"训练数据: 输入 {all_inputs.shape}, 目标 {all_targets.shape}")

    # 9:1 划分
    n_total = len(all_inputs)
    n_train = int(n_total * 0.9)
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    print(f"训练集: {len(train_idx)} 对, 验证集: {len(val_idx)} 对")

    # 创建 DataLoader
    train_dataset = TensorDataset(
        torch.from_numpy(all_inputs[train_idx]),
        torch.from_numpy(all_targets[train_idx])
    )
    val_dataset = TensorDataset(
        torch.from_numpy(all_inputs[val_idx]),
        torch.from_numpy(all_targets[val_idx])
    )
    train_loader = DataLoader(train_dataset, batch_size=UNET_PRETRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=UNET_PRETRAIN_BATCH_SIZE, shuffle=False)

    # 创建模型
    model = FlowUNet(in_channels=2, base_ch=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=UNET_PRETRAIN_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    l1_loss = nn.L1Loss()
    photo_loss = PhotometricLoss()
    smooth_loss = SmoothnessLoss()

    # 损失权重
    w_supervised = 0.5
    w_photometric = 0.3
    w_smooth = 0.2

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(UNET_PRETRAIN_EPOCHS):
        # ---------- Training ----------
        model.train()
        train_loss_sum = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # 分离两张图像
            img1 = batch_inputs[:, 0:1, :, :]  # (B, 1, H, W)
            img2 = batch_inputs[:, 1:2, :, :]

            optimizer.zero_grad()
            flow_pred = model(batch_inputs)

            loss_sup = l1_loss(flow_pred, batch_targets)
            loss_photo = photo_loss(img1, img2, flow_pred)
            loss_smooth = smooth_loss(flow_pred)
            loss = w_supervised * loss_sup + w_photometric * loss_photo + w_smooth * loss_smooth

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---------- Validation ----------
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                img1 = batch_inputs[:, 0:1, :, :]
                img2 = batch_inputs[:, 1:2, :, :]

                flow_pred = model(batch_inputs)

                loss_sup = l1_loss(flow_pred, batch_targets)
                loss_photo = photo_loss(img1, img2, flow_pred)
                loss_smooth = smooth_loss(flow_pred)
                loss = w_supervised * loss_sup + w_photometric * loss_photo + w_smooth * loss_smooth

                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{UNET_PRETRAIN_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), UNET_PRETRAIN_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= UNET_PRETRAIN_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # ---------- 保存预训练曲线 ----------
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('U-Net Flow Pretraining Loss Curve')
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(UNET_PRETRAIN_DIR, 'pretrain_loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"预训练损失曲线已保存到: {loss_curve_path}")

    # 保存 loss 数据
    loss_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_df.to_csv(os.path.join(UNET_PRETRAIN_DIR, 'pretrain_loss.csv'), index=False)

    # ---------- 可视化示例 ----------
    model.eval()
    with torch.no_grad():
        sample_input = torch.from_numpy(all_inputs[val_idx[0:1]]).to(device)
        sample_target = all_targets[val_idx[0]]
        sample_pred = model(sample_input).cpu().numpy()[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # 输入帧
    axes[0, 0].imshow(all_inputs[val_idx[0], 0], cmap='gray')
    axes[0, 0].set_title('Frame t')
    axes[0, 1].imshow(all_inputs[val_idx[0], 1], cmap='gray')
    axes[0, 1].set_title('Frame t+1')
    # 伪标签光流
    mag_true = np.sqrt(sample_target[0] ** 2 + sample_target[1] ** 2)
    axes[0, 2].imshow(mag_true, cmap='hot')
    axes[0, 2].set_title('Farneback (pseudo-label)')
    # 预测光流
    mag_pred = np.sqrt(sample_pred[0] ** 2 + sample_pred[1] ** 2)
    axes[1, 0].imshow(mag_pred, cmap='hot')
    axes[1, 0].set_title('U-Net Prediction')
    # 误差
    diff = np.abs(mag_true - mag_pred)
    axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title('|Diff|')
    # 误差直方图
    axes[1, 2].hist(diff.flatten(), bins=50)
    axes[1, 2].set_title('Error Distribution')

    for ax in axes.flat:
        ax.axis('off')
    axes[1, 2].axis('on')

    sample_path = os.path.join(UNET_PRETRAIN_DIR, 'flow_comparison.png')
    plt.tight_layout()
    plt.savefig(sample_path)
    plt.close()
    print(f"光流对比图已保存到: {sample_path}")

    # 加载最佳模型
    model.load_state_dict(torch.load(UNET_PRETRAIN_MODEL_PATH))
    model.eval()
    print("U-Net 光流预训练完成！")
    print("=" * 60)

    return model


# ============================================================================
# 原始函数 (保留不变的部分)
# ============================================================================

def sample_selection(info_excel_path, label_base_dir, time_steps=6, forecast_steps=3):
    def check_data_completeness(date_str, input_ts_list, label_ts_str, label_path):
        missing_reasons = []
        awos_path = os.path.join(base_dir, date_str, 'AWS', f"{date_str}_AWS.csv")
        if not os.path.exists(awos_path):
            missing_reasons.append(f"AWOS: 缺失 {awos_path}")
        else:
            print(f"  AWOS: 存在 {awos_path}")
        if not os.path.exists(label_path):
            missing_reasons.append(f"标签: 缺失 {label_path}")
        else:
            print(f"  标签: 存在 {label_path}")
        radar_dir = os.path.join(base_dir, date_str, 'radar_img')
        for ts_str in input_ts_list:
            files = [f for f in os.listdir(radar_dir) if f.startswith(ts_str + '_') and f.endswith('.jpg')]
            if len(files) != 15:
                missing_reasons.append(f"雷达: 未找到15文件 for {ts_str} (found {len(files)})")
            else:
                print(f"  雷达: 找到 {len(files)} 文件 (e.g., {os.path.join(radar_dir, files[0])})")
        for ts_str in input_ts_list:
            parts = ts_str.split('_')
            if len(parts) != 2:
                missing_reasons.append(f"无效 ts 格式: {ts_str}")
                continue
            date_part = parts[0]
            time_part = parts[1]
            time_no_ss = time_part[:4]
            timestamp_no_ss = f"{date_part}{time_no_ss}"
            for prefix, sub in zip(['IEC', 'UEC', 'WEC'], ['I', 'U', 'W']):
                sat_path = os.path.join(base_dir, date_str, 'cloud_img', sub, f"{prefix}{timestamp_no_ss}_GH4.jpg")
                if not os.path.exists(sat_path):
                    missing_reasons.append(f"卫星 {prefix} 缺失 for {ts_str}: {sat_path}")
                else:
                    print(f"  卫星 {prefix}: 存在 {sat_path}")
        if missing_reasons:
            print(f"丢弃样本: {date_str}, {label_ts_str} - 缺失数据. 原因: {'; '.join(missing_reasons)}")
            return False
        return True

    info_df = pd.read_excel(info_excel_path)
    dates = info_df['filename'].astype(str).tolist()
    print(f"步骤1: 从 infomation.xlsx 读取到 {len(dates)} 个日期: {dates[:5]}...")

    all_positive = []
    all_negative = []

    for date_str in dates:
        try:
            formatted_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            print(f"  转换: {date_str} -> {formatted_date}")
        except ValueError:
            print(f"警告: 无效日期格式: {date_str} (跳过)")
            continue

        label_path = os.path.join(label_base_dir, f"{formatted_date}.xlsx")
        if not os.path.exists(label_path):
            print(f"警告: 步骤2: 标签文件不存在: {label_path}")
            continue
        df = pd.read_excel(label_path)
        df['timestamp'] = pd.to_datetime(df['时间(LT)'])
        timestamps = df['timestamp'].tolist()
        labels = df[['雷暴', '短时强降水', '大风']].values.tolist()
        print(f"步骤2: 对于日期 {formatted_date}, 加载标签Excel, 总时间点: {len(timestamps)}")

        start_skip = 10
        end_skip = 10
        if len(timestamps) < start_skip + end_skip + 1:
            print(f"警告: 步骤3: {formatted_date} 时间点不足, 跳过")
            continue

        for i in range(start_skip, len(timestamps) - end_skip):
            label_idx = i + forecast_steps
            if label_idx >= len(timestamps):
                continue
            target_ts = timestamps[i]
            label_ts = timestamps[label_idx]
            target_labels = labels[label_idx]

            if not (7 <= label_ts.hour <= 22):
                continue

            input_timestamps = timestamps[i - time_steps + 1: i + 1]
            if len(input_timestamps) < time_steps:
                continue

            input_ts_list = [dt.strftime('%Y%m%d_%H%M00') for dt in input_timestamps]
            label_ts_str = label_ts.strftime('%Y%m%d_%H%M00')

            if check_data_completeness(date_str, input_ts_list, label_ts_str, label_path):
                if any(target_labels):
                    all_positive.append((formatted_date, target_ts, tuple(target_labels)))
                else:
                    all_negative.append((formatted_date, target_ts))

    print(f"步骤4: 潜在正样本数: {len(all_positive)}, 潜在负样本数: {len(all_negative)}")

    # Save potential samples
    potential_data = []
    classes_list = ['雷暴', '短时强降水', '大风']
    for date, ts, lbls in all_positive:
        sample_classes = ','.join([classes_list[j] for j in range(3) if lbls[j] == 1])
        potential_data.append({
            'Type': 'Positive', 'Date': date,
            'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'Classes': sample_classes, 'Labels': str(list(lbls))
        })
    for date, ts in all_negative:
        potential_data.append({
            'Type': 'Negative', 'Date': date,
            'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'Classes': '', 'Labels': '[0, 0, 0]'
        })
    if potential_data:
        df_potential = pd.DataFrame(potential_data)
        output_path = os.path.join(output_dir, 'potential_samples.xlsx')
        df_potential.to_excel(output_path, index=False)
        print(f"潜在样本池已保存到: {output_path} (总行: {len(df_potential)})")

    return all_positive, all_negative


def find_nearest_ts(base_dir, yyyymmdd, target_ts_str, max_delta=10):
    target_dt = datetime.strptime(target_ts_str, '%Y%m%d_%H%M%S')
    available_files = glob.glob(os.path.join(base_dir, yyyymmdd, 'radar_img', '*_50kM.jpg'))
    available_ts = []
    for f in available_files:
        parts = os.path.basename(f).split('_')
        if len(parts) >= 2:
            ts = parts[0] + '_' + parts[1] if '_' not in parts[1] else parts[0] + '_' + parts[1].split('_')[0]
            try:
                available_ts.append(ts)
            except:
                continue

    all_ts_str = []
    for f in available_files:
        basename = os.path.basename(f)
        match = re.match(r'(\d{8}_\d{6})_', basename)
        if match:
            all_ts_str.append(match.group(1))

    min_delta = None
    nearest_ts = None
    for ts in all_ts_str:
        try:
            dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
            delta = abs((dt - target_dt).total_seconds()) / 60.0
            if delta <= max_delta and (min_delta is None or delta < min_delta):
                min_delta = delta
                nearest_ts = ts
        except ValueError:
            continue
    return nearest_ts


def load_radar_images(base_dir, dates_timestamps, img_size=(400, 400), num_angles=15):
    radar_data = {}
    all_timestamps = {}
    pattern = re.compile(r'(\d{8}_\d{6})_(\d+(\.\d+)?)_50kM\.jpg', re.IGNORECASE)

    for date_str in dates_timestamps:
        yyyymmdd = date_str.replace('-', '')
        date_dir = os.path.join(base_dir, yyyymmdd, 'radar_img')
        if not os.path.exists(date_dir):
            print(f"警告: 雷达目录不存在: {date_dir}")
            continue

        date_radar = {}
        required_ts = set(dates_timestamps[date_str])
        loaded_count = 0

        for filename in sorted(os.listdir(date_dir)):
            match = pattern.match(filename)
            if match and match.group(1) in required_ts:
                timestamp = match.group(1)
                angle = float(match.group(2))
                int_angle = min(max(int(round(angle)), 0), num_angles - 1)
                if timestamp not in date_radar:
                    date_radar[timestamp] = np.zeros((num_angles, img_size[0], img_size[1], 1), dtype=np.float32)
                img_path = os.path.join(date_dir, filename)
                img = np.array(Image.open(img_path).convert('L')) / 255.0
                date_radar[timestamp][int_angle] = img[:, :, np.newaxis]
                loaded_count += 1

        sorted_ts = sorted(date_radar.keys())
        radar_array = np.array([date_radar[ts] for ts in sorted_ts])
        radar_data[date_str] = radar_array
        all_timestamps[date_str] = sorted_ts
        print(f"雷达 {date_str}: 加载 {loaded_count} 个图像, 唯一时间戳 {len(sorted_ts)}")

    return radar_data, all_timestamps


def load_satellite_images(base_dir, dates_timestamps, img_size=(200, 200)):
    satellite_data = {}
    all_timestamps = {}
    pattern = re.compile(r'(IEC|UEC|WEC)(\d{8}\d{4})\_GH4\.jpg', re.IGNORECASE)
    channel_map = {'IEC': 0, 'UEC': 1, 'WEC': 2}
    subdirs = {'IEC': 'I', 'UEC': 'U', 'WEC': 'W'}

    for date_str in dates_timestamps:
        date_dir_str = date_str.replace('-', '') if '-' in date_str else date_str
        date_dir = os.path.join(base_dir, date_dir_str, 'cloud_img')
        if not os.path.exists(date_dir):
            print(f"警告: 卫星目录不存在: {date_dir}")
            continue

        date_satellite_dict = {}
        date_ts_list = []

        for channel_prefix, subdir in subdirs.items():
            sub_dir = os.path.join(date_dir, subdir)
            if not os.path.exists(sub_dir):
                print(f"警告: 子文件夹不存在: {sub_dir}")
                continue

            for filename in sorted(os.listdir(sub_dir)):
                match = pattern.match(filename)
                if match:
                    file_channel = match.group(1)
                    if file_channel != channel_prefix:
                        continue
                    timestamp_raw = match.group(2)
                    try:
                        if '_' in timestamp_raw:
                            dt = datetime.strptime(timestamp_raw, '%Y%m%d_%H%M')
                        else:
                            dt = datetime.strptime(timestamp_raw, '%Y%m%d%H%M')
                    except ValueError:
                        try:
                            dt = datetime.strptime(timestamp_raw, '%Y-%m-%d_%H%M')
                        except ValueError:
                            continue
                    full_ts = dt.strftime('%Y%m%d_%H%M%S')
                    if full_ts not in dates_timestamps.get(date_str, []):
                        continue

                    img_path = os.path.join(sub_dir, filename)
                    try:
                        img = Image.open(img_path).convert('L')
                        img = img.resize((img_size[1], img_size[0]), Image.Resampling.LANCZOS)
                        img_array = np.array(img) / 255.0
                    except Exception as e:
                        print(f"警告: 无法读取/处理卫星图像: {img_path} - {e}")
                        continue

                    channel_idx = channel_map[file_channel]
                    if full_ts not in date_satellite_dict:
                        date_satellite_dict[full_ts] = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
                    date_satellite_dict[full_ts][:, :, channel_idx] = img_array
                    if full_ts not in date_ts_list:
                        date_ts_list.append(full_ts)

        sorted_ts = sorted(date_ts_list)
        if sorted_ts:
            satellite_array = np.array([date_satellite_dict[ts] for ts in sorted_ts])
            satellite_data[date_str] = satellite_array
            all_timestamps[date_str] = sorted_ts
            print(f"卫星 {date_str}: 总时间戳数: {len(sorted_ts)}")
        else:
            print(f"警告: {date_str} 没有加载任何卫星数据")

    return satellite_data, all_timestamps


def load_awos_data(base_dir, dates_timestamps):
    awos_data = {}
    all_timestamps = {}
    features = ['10分风向', '10分风速', '气压', '温度', '湿度',
                '24h_变温度', '3h_变温度', '24h_变气压', '3h_变气压']
    scaler = StandardScaler()

    for date_str in dates_timestamps:
        yyyymmdd = date_str.replace('-', '')
        awos_path = os.path.join(base_dir, yyyymmdd, 'AWS', f"{yyyymmdd}_AWS.csv")
        if not os.path.exists(awos_path):
            print(f"警告: AWS文件不存在: {awos_path}")
            continue

        try:
            df = pd.read_csv(awos_path, encoding='utf-8', sep=',')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(awos_path, encoding='gb18030', sep=',')
            except Exception as e:
                print(f"错误: 读取失败 {awos_path}: {e}")
                continue
        except Exception as e:
            print(f"错误: 读取失败 {awos_path}: {e}")
            continue

        required_cols = features + ['datetime']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"错误: 缺失列 in {awos_path}: {missing_cols}")
            continue

        df = df.dropna(subset=required_cols)
        parsed_ts = []
        for t in df['datetime']:
            dt = None
            formats = ['%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M']
            for fmt in formats:
                try:
                    dt = datetime.strptime(t.strip(), fmt)
                    break
                except ValueError:
                    pass
            if dt:
                parsed_ts.append(dt.strftime('%Y%m%d_%H%M%S'))
            else:
                parsed_ts.append(None)

        df['timestamp'] = parsed_ts
        required_ts = set(dates_timestamps[date_str])
        mask = df['timestamp'].isin(required_ts)
        df_filtered = df[mask]

        if df_filtered.empty:
            print(f"警告: {date_str} 没有匹配的时间戳")
            continue

        awos_values = df_filtered[features].values.astype(np.float32)
        awos_values = scaler.fit_transform(awos_values)
        awos_data[date_str] = awos_values
        all_timestamps[date_str] = df_filtered['timestamp'].tolist()
        print(f"AWOS {date_str}: 加载 {len(df_filtered)} 行")

    return awos_data, all_timestamps


def load_labels(label_base_dir, dates_timestamps):
    labels_data = {}
    all_timestamps = {}

    for date_str in dates_timestamps:
        label_path = os.path.join(label_base_dir, f"{date_str}.xlsx")
        if not os.path.exists(label_path):
            print(f"警告: 标签文件不存在: {label_path}")
            continue

        df = pd.read_excel(label_path)
        df['timestamp'] = pd.to_datetime(df['时间(LT)']).apply(lambda dt: dt.strftime('%Y%m%d_%H%M%S'))
        required_ts = set(dates_timestamps[date_str])
        mask = df['timestamp'].isin(required_ts)
        df_filtered = df[mask]

        if df_filtered.empty:
            print(f"警告: {date_str} 没有匹配的时间戳")
            continue

        labels_values = df_filtered[['雷暴', '短时强降水', '大风']].values.astype(np.float32)
        labels_data[date_str] = labels_values
        all_timestamps[date_str] = df_filtered['timestamp'].tolist()
        print(f"标签 {date_str}: 加载 {len(df_filtered)} 行")

    return labels_data, all_timestamps


# ============================================================================
# 核心修改: create_samples — 使用 U-Net 替代 Farneback
# ============================================================================

def create_samples(radar_data, satellite_data, awos_data, labels_data, samples_list,
                   all_timestamps, time_steps=6, forecast_steps=3, flow_unet=None):
    """
    创建样本。光流部分使用预训练的 U-Net 模型。
    """
    samples = {'radar': [], 'satellite': [], 'awos': [], 'optical_flow': [],
               'labels': [], 'indices': [], 'timestamps': []}
    skipped_count = 0

    for idx, (date, label_ts, substituted) in enumerate(samples_list):
        try:
            input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(time_steps)]
            input_str_list = [ts.strftime('%Y%m%d_%H%M%S') for ts in input_ts_list]
            label_str = label_ts.strftime('%Y%m%d_%H%M%S')

            for i, ts_str in enumerate(input_str_list):
                if ts_str in substituted:
                    input_str_list[i] = substituted[ts_str]
            if label_str in substituted:
                label_str = substituted[label_str]

            date_radar = radar_data.get(date, np.array([]))
            date_satellite = satellite_data.get(date, np.array([]))
            date_awos = awos_data.get(date, np.array([]))
            date_labels = labels_data.get(date, np.array([]))
            date_ts = all_timestamps.get(date, [])

            if not date_ts or date_radar.size == 0:
                raise ValueError("无可用时间戳或数据为空")

            input_indices = [date_ts.index(ts_str) for ts_str in input_str_list if ts_str in date_ts]
            label_idx = date_ts.index(label_str) if label_str in date_ts else -1

            if len(input_indices) != time_steps or label_idx == -1:
                raise ValueError(f"时间戳不完整: 输入 {len(input_indices)}/{time_steps}")

            if max(input_indices) >= len(date_radar) or label_idx >= len(date_labels):
                raise IndexError(f"索引超出范围")

            # ==================================================================
            # 使用 U-Net 计算光流（替代 Farneback）
            # ==================================================================
            radar_3rd = date_radar[input_indices, 2, :, :, 0]  # (T, 400, 400)
            optical_flow = []

            for t in range(len(radar_3rd) - 1):
                prev = radar_3rd[t]   # (400, 400) float [0, 1]
                curr = radar_3rd[t + 1]

                if flow_unet is not None:
                    # U-Net 推理
                    stack = np.stack([prev, curr], axis=0)  # (2, 400, 400)
                    stack_tensor = torch.from_numpy(stack).unsqueeze(0).to(device)  # (1, 2, 400, 400)
                    with torch.no_grad():
                        flow = flow_unet(stack_tensor).cpu().numpy()[0]  # (2, 400, 400)
                else:
                    # Fallback: Farneback
                    prev_u8 = (prev * 255.0).astype(np.uint8)
                    curr_u8 = (curr * 255.0).astype(np.uint8)
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_u8, curr_u8, None,
                        pyr_scale=0.5, levels=4, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )
                    flow = np.transpose(flow, (2, 0, 1)).astype(np.float32)

                optical_flow.append(flow)

            optical_flow = np.array(optical_flow, dtype=np.float32)
            if len(optical_flow) < time_steps - 1:
                optical_flow = np.pad(optical_flow,
                                      ((0, time_steps - 1 - len(optical_flow)), (0, 0), (0, 0), (0, 0)),
                                      mode='edge')

            radar_data_sample = date_radar[input_indices]
            satellite_data_sample = date_satellite[input_indices]
            awos_data_sample = date_awos[input_indices]
            optical_flow_sample = optical_flow
            labels_sample = date_labels[label_idx]
            timestamps_sample = label_str
            indices_sample = idx

            samples['radar'].append(radar_data_sample)
            samples['satellite'].append(satellite_data_sample)
            samples['awos'].append(awos_data_sample)
            samples['optical_flow'].append(optical_flow_sample)
            samples['labels'].append(labels_sample)
            samples['timestamps'].append(timestamps_sample)
            samples['indices'].append(indices_sample)

        except Exception as e:
            print(f"警告: 样本 {idx} on {date} 被跳过: {e}")
            skipped_count += 1
            continue

    for key in samples:
        samples[key] = np.array(samples[key])

    print(f"创建样本: {len(samples['indices'])}, 跳过 {skipped_count} 个")
    return samples


# ============================================================================
# 数据增强
# ============================================================================

def augment_rare(samples, rare_multiplier=3):
    augmented = {k: list(samples[k]) for k in samples}
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ])

    for i in range(len(samples['labels'])):
        lbl = samples['labels'][i]
        if lbl[1] or lbl[2]:
            for _ in range(rare_multiplier - 1):
                radar = samples['radar'][i]
                aug_radar = np.zeros_like(radar)
                for t in range(radar.shape[0]):
                    for a in range(radar.shape[1]):
                        img = radar[t, a]
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                        if img_tensor.shape[0] == 1:
                            img_tensor = img_tensor.repeat(3, 1, 1)
                        aug_tensor = transform(img_tensor)
                        if aug_tensor.shape[0] == 3:
                            aug_tensor = aug_tensor.mean(dim=0, keepdim=True)
                        aug_img = aug_tensor.permute(1, 2, 0).numpy()
                        aug_radar[t, a] = aug_img

                satellite = samples['satellite'][i]
                aug_satellite = np.zeros_like(satellite)
                for t in range(satellite.shape[0]):
                    img = satellite[t]
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                    aug_tensor = transform(img_tensor)
                    aug_img = aug_tensor.permute(1, 2, 0).numpy()
                    aug_satellite[t] = aug_img

                aug_optical_flow = samples['optical_flow'][i]
                aug_awos = samples['awos'][i] + np.random.normal(0, 0.01, samples['awos'][i].shape)

                augmented['radar'].append(aug_radar)
                augmented['satellite'].append(aug_satellite)
                augmented['awos'].append(aug_awos)
                augmented['optical_flow'].append(aug_optical_flow)
                augmented['labels'].append(samples['labels'][i])
                augmented['indices'].append(samples['indices'][i])
                augmented['timestamps'].append(samples['timestamps'][i])

    for k in augmented:
        augmented[k] = np.array(augmented[k])

    print(f"增强后样本数: {len(augmented['labels'])}, 原: {len(samples['labels'])}")
    return augmented


# ============================================================================
# 数据加载（主函数）
# ============================================================================

def load_data(flow_unet=None):
    if os.path.exists(save_path):
        print(f"步骤: 加载已保存数据集从 {save_path}")
        loaded = np.load(save_path, allow_pickle=True)
        samples = {key: loaded[key] for key in loaded.files}
        print(f"步骤: 已加载样本数: {len(samples['indices'])}")

        negative_samples = []
        for i, lbl in enumerate(samples['labels']):
            if np.all(lbl == [0, 0, 0]):
                ts = samples['timestamps'][i]
                negative_samples.append({
                    'Index': samples['indices'][i],
                    'Timestamp': ts,
                    'Labels': str(list(lbl)),
                    'Type': 'Negative'
                })

        df_negative = pd.DataFrame(negative_samples)
        print("负样本总数:", len(df_negative))
        print(df_negative.head(10))

        output_path = os.path.join(base_dir, 'negative_samples.xlsx')
        df_negative.to_excel(output_path, index=False)
        print(f"负样本信息已保存到: {output_path}")
        return samples

    # 获取潜在样本池
    all_positive, all_negative = sample_selection(info_excel, label_dir)
    print(f"潜在正样本池: {len(all_positive)}, 负: {len(all_negative)}")

    all_samples_list = [(date, target_ts, {}) for date, target_ts, _ in all_positive] + \
                       [(date, target_ts, {}) for date, target_ts in all_negative]
    dates_timestamps = {}
    for date, label_ts, _ in all_samples_list:
        input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(time_steps)]
        required_ts = input_ts_list + [label_ts]
        required_str = [ts.strftime('%Y%m%d_%H%M%S') for ts in required_ts]
        dates_timestamps.setdefault(date, []).extend(required_str)
        dates_timestamps[date] = sorted(set(dates_timestamps[date]))

    radar_data, radar_ts = load_radar_images(base_dir, dates_timestamps)
    print(f"加载雷达数据用于负样本优化: {sum(len(ts) for ts in radar_ts.values())} timestamps")

    def extract_radar_features(date, label_ts, radar_data, radar_ts, time_steps=6):
        try:
            input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(time_steps)]
            input_str_list = [ts.strftime('%Y%m%d_%H%M%S') for ts in input_ts_list]
            date_ts_list = radar_ts.get(date, [])
            if not date_ts_list:
                return None
            input_indices = [date_ts_list.index(ts_str) for ts_str in input_str_list if ts_str in date_ts_list]
            if len(input_indices) != time_steps:
                return None
            radar_seq = radar_data[date][input_indices]
            features = []
            for t in range(time_steps):
                slice_t = radar_seq[t].flatten()
                mean_t = np.mean(slice_t)
                var_t = np.var(slice_t)
                features.extend([mean_t, var_t])
            return np.array(features)
        except Exception as e:
            print(f"特征提取失败 for {date}, {label_ts}: {e}")
            return None

    pos_features = []
    pos_keys = []
    for date, label_ts, _ in all_positive:
        feat = extract_radar_features(date, label_ts, radar_data, radar_ts)
        if feat is not None:
            pos_features.append(feat)
            pos_keys.append((date, label_ts))
    pos_features = np.array(pos_features)
    print(f"提取正样本特征: {len(pos_features)}")

    neg_features = []
    neg_keys = []
    for date, target_ts in all_negative:
        feat = extract_radar_features(date, target_ts, radar_data, radar_ts)
        if feat is not None:
            neg_features.append(feat)
            neg_keys.append((date, target_ts))
    neg_features = np.array(neg_features)
    print(f"提取负样本特征: {len(neg_features)}")

    if len(pos_features) > 0 and len(neg_features) > 0:
        similarities = cosine_similarity(neg_features, pos_features).mean(axis=1)
        sorted_indices = np.argsort(similarities)[::-1]
        delete_ratio = 0.3
        num_delete = int(len(neg_features) * delete_ratio)
        to_keep_indices = sorted_indices[num_delete:]
        refined_neg_keys = [neg_keys[i] for i in to_keep_indices]
        print(f"负样本优化: 原 {len(all_negative)}, 删除 {num_delete} 硬负样本, 保留 {len(refined_neg_keys)}")
        all_negative = refined_neg_keys
    else:
        print("警告: 无法优化负样本 (特征提取为空)，使用原列表")

    manual_neg_path = os.path.join(base_dir, 'negative_samples0.xlsx')
    if os.path.exists(manual_neg_path):
        try:
            df_manual = pd.read_excel(manual_neg_path)
            if 'Timestamp' not in df_manual.columns:
                raise ValueError("negative_samples0.xlsx 缺少 'Timestamp' 列")
            all_negative = []
            for _, row in df_manual.iterrows():
                timestamp_str = row['Timestamp']
                dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                date = dt.strftime('%Y-%m-%d')
                all_negative.append((date, dt))
            print(f"使用人工调整负样本: {len(all_negative)} 个 from {manual_neg_path}")
        except Exception as e:
            print(f"警告: 读取 negative_samples0.xlsx 失败: {e}")

    del radar_data, radar_ts

    # 分离稀有正样本
    rare_positive = [c for c in all_positive if c[2][1] == 1 or c[2][2] == 1]
    other_positive = [c for c in all_positive if c not in rare_positive]
    print(f"调试: 稀有正样本数: {len(rare_positive)}, 其他正样本数: {len(other_positive)}")

    target_positive = 10
    negative_ratio = 1

    # 验证稀有正样本
    valid_rare = []
    substituted_timestamps = {}
    for candidate in rare_positive:
        date, label_ts, lbls = candidate
        input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(6)]
        required_ts = input_ts_list + [label_ts]
        required_str = [ts.strftime('%Y%m%d_%H%M%S') for ts in required_ts]

        all_exist = True
        sample_substituted = {}
        for ts_str in required_str:
            yyyymmdd = ts_str[:8]
            radar_pattern = os.path.join(base_dir, yyyymmdd, 'radar_img', f'{yyyymmdd}_{ts_str[9:15]}_*_50kM.jpg')
            files = glob.glob(radar_pattern)
            if len(files) == 0:
                substitute_ts = find_nearest_ts(base_dir, yyyymmdd, ts_str)
                if substitute_ts:
                    sample_substituted[ts_str] = substitute_ts
                else:
                    all_exist = False
                    break
        if all_exist:
            valid_rare.append((date, label_ts, lbls))
            if sample_substituted:
                substituted_timestamps[(date, label_ts.strftime('%Y%m%d_%H%M%S'))] = sample_substituted

    num_other_needed = max(0, target_positive - len(valid_rare))
    valid_other = []
    random.shuffle(other_positive)
    for candidate in other_positive:
        if len(valid_other) >= num_other_needed:
            break
        date, label_ts, lbls = candidate
        input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(6)]
        required_str = [ts.strftime('%Y%m%d_%H%M%S') for ts in (input_ts_list + [label_ts])]

        all_exist = True
        sample_substituted = {}
        for ts_str in required_str:
            yyyymmdd = ts_str[:8]
            radar_pattern = os.path.join(base_dir, yyyymmdd, 'radar_img', f'{yyyymmdd}_{ts_str[9:15]}_*_50kM.jpg')
            files = glob.glob(radar_pattern)
            if len(files) == 0:
                substitute_ts = find_nearest_ts(base_dir, yyyymmdd, ts_str)
                if substitute_ts:
                    sample_substituted[ts_str] = substitute_ts
                else:
                    all_exist = False
                    break
        if all_exist:
            valid_other.append((date, label_ts, lbls))
            if sample_substituted:
                substituted_timestamps[(date, label_ts.strftime('%Y%m%d_%H%M%S'))] = sample_substituted

    valid_positive = valid_rare + valid_other
    print(f"有效正样本: 稀有 {len(valid_rare)}, 其他 {len(valid_other)}, 总 {len(valid_positive)}")

    target_negative = len(valid_positive) * negative_ratio
    valid_negative = []
    candidates_neg = list(all_negative)
    random.shuffle(candidates_neg)
    for candidate in candidates_neg:
        if len(valid_negative) >= target_negative:
            break
        valid_negative.append(candidate)

    samples_list = [(date, label_ts, substituted_timestamps.get((date, label_ts.strftime('%Y%m%d_%H%M%S')), {}))
                    for date, label_ts, _ in valid_positive] + \
                   [(date, label_ts, {}) for date, label_ts in valid_negative]
    print(f"有效样本: 正 {len(valid_positive)}, 负 {len(valid_negative)} (总 {len(samples_list)})")

    dates_timestamps = {}
    for date, label_ts, substituted in samples_list:
        input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(6)]
        required_ts = input_ts_list + [label_ts]
        required_str = [ts.strftime('%Y%m%d_%H%M%S') for ts in required_ts]
        for i, ts_str in enumerate(required_str):
            if ts_str in substituted:
                required_str[i] = substituted[ts_str]
        dates_timestamps.setdefault(date, []).extend(required_str)
        dates_timestamps[date] = sorted(set(dates_timestamps[date]))

    radar_data, radar_ts = load_radar_images(base_dir, dates_timestamps)
    satellite_data, satellite_ts = load_satellite_images(base_dir, dates_timestamps)
    awos_data, awos_ts = load_awos_data(base_dir, dates_timestamps)
    labels_data, labels_ts = load_labels(label_dir, dates_timestamps)

    samples = create_samples(radar_data, satellite_data, awos_data, labels_data,
                             samples_list, radar_ts, flow_unet=flow_unet)
    samples = augment_rare(samples, rare_multiplier=3)

    rare_count = sum(1 for lbl in samples['labels'] if any(lbl[1:]))
    wind_count = sum(1 for lbl in samples['labels'] if lbl[2])
    rain_count = sum(1 for lbl in samples['labels'] if lbl[1])
    print(f"最终样本统计: 稀有样本总数: {rare_count}, 大风: {wind_count}, 短时强降水: {rain_count}")

    if len(samples['indices']) > 0:
        print(f"步骤: 保存数据集到 {save_path}")
        np.savez(save_path, **samples)

    return samples


# ============================================================================
# 模型架构
# ============================================================================

class ConvLSTM2d(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, num_layers=1, batch_first=False):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.conv = nn.Conv2d(
            in_channels=input_size + hidden_size,
            out_channels=4 * hidden_size,
            kernel_size=self.kernel_size,
            padding=self.padding, bias=True
        )

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        batch_size, seq_len, _, height, width = input_tensor.size()
        if hidden_state is None:
            h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, height, width, device=input_tensor.device)
            c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, height, width, device=input_tensor.device)
        else:
            h_t, c_t = hidden_state

        h_t_new = torch.zeros_like(h_t)
        c_t_new = torch.zeros_like(c_t)
        for layer in range(self.num_layers):
            h_t_layer = h_t[layer].clone()
            c_t_layer = c_t[layer].clone()
            output_inner = []
            for t in range(seq_len):
                combined = torch.cat((input_tensor[:, t], h_t_layer), dim=1)
                gates = self.conv(combined)
                i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                o_gate = torch.sigmoid(o_gate)
                c_tilde = torch.tanh(c_gate)
                c_t_layer_new = f_gate * c_t_layer + i_gate * c_tilde
                h_t_layer_new = o_gate * torch.tanh(c_t_layer_new)
                output_inner.append(h_t_layer_new.clone())
                h_t_layer = h_t_layer_new
                c_t_layer = c_t_layer_new
            h_t_new[layer] = h_t_layer.clone()
            c_t_new[layer] = c_t_layer.clone()
        last_state = (h_t_new, c_t_new)
        return torch.stack(output_inner, dim=1), last_state


class TwoDCNN(nn.Module):
    def __init__(self, input_channels, feature_dim=64):
        super(TwoDCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        h, w = (400, 400) if input_channels in [15, 2] else (200, 200)
        h, w = h // 4, w // 4
        self.fc = nn.Linear(64 * h * w, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class WeatherModel(nn.Module):
    def __init__(self, time_steps=6, use_optical_flow=False):
        super(WeatherModel, self).__init__()
        self.time_steps = time_steps
        self.use_optical_flow = use_optical_flow
        self.radar_cnn = TwoDCNN(input_channels=15, feature_dim=64)
        self.satellite_cnn = TwoDCNN(input_channels=3, feature_dim=64)
        self.conv_lstm = ConvLSTM2d(input_size=64, hidden_size=64, kernel_size=3, num_layers=1, batch_first=True)
        self.fc_awos = nn.Linear(9, 16)
        input_dim = 64 + 64 + 16
        if use_optical_flow:
            self.flow_cnn = TwoDCNN(input_channels=2, feature_dim=64)
            input_dim += 64
        self.fc = nn.Linear(input_dim, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, radar_data, satellite_data, awos_data, optical_flow=None):
        batch_size = radar_data.size(0)
        radar_features = []
        for t in range(self.time_steps):
            radar_t = radar_data[:, t].squeeze(-1)
            radar_features.append(self.radar_cnn(radar_t))
        radar_features = torch.stack(radar_features, dim=1)

        satellite_features = []
        for t in range(self.time_steps):
            satellite_t = satellite_data[:, t].permute(0, 3, 1, 2)
            satellite_features.append(self.satellite_cnn(satellite_t))
        satellite_features = torch.stack(satellite_features, dim=1)

        flow_features = None
        if self.use_optical_flow and optical_flow is not None:
            flow_features = []
            flow_seq_len = optical_flow.size(1)
            for t in range(min(flow_seq_len, self.time_steps)):
                flow_t = optical_flow[:, t]
                if flow_t.size(1) != 2:
                    flow_t = flow_t.permute(0, 3, 1, 2)
                flow_features.append(self.flow_cnn(flow_t))
            if flow_features:
                flow_features = torch.stack(flow_features, dim=1)
                if flow_features.size(1) < self.time_steps:
                    pad = torch.zeros(batch_size, self.time_steps - flow_features.size(1), 64, device=radar_data.device)
                    flow_features = torch.cat([flow_features, pad], dim=1)
            else:
                flow_features = torch.zeros(batch_size, self.time_steps, 64, device=radar_data.device)

        radar_features = radar_features.unsqueeze(-1).unsqueeze(-1)
        _, (h_n, _) = self.conv_lstm(radar_features)
        radar_out = h_n.squeeze(0).view(batch_size, -1)

        satellite_features = satellite_features.unsqueeze(-1).unsqueeze(-1)
        _, (h_n, _) = self.conv_lstm(satellite_features)
        satellite_out = h_n.squeeze(0).view(batch_size, -1)

        if self.use_optical_flow and flow_features is not None:
            flow_features = flow_features.unsqueeze(-1).unsqueeze(-1)
            _, (h_n, _) = self.conv_lstm(flow_features)
            flow_out = h_n.squeeze(0).view(batch_size, -1)
        else:
            flow_out = torch.zeros(batch_size, 64, device=radar_data.device)

        awos_features = self.relu(self.fc_awos(awos_data[:, -1]))
        if self.use_optical_flow and flow_features is not None:
            features = torch.cat([radar_out, satellite_out, awos_features, flow_out], dim=1)
        else:
            features = torch.cat([radar_out, satellite_out, awos_features], dim=1)
        return self.sigmoid(self.fc(features))


# ============================================================================
# 损失函数
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, device=None):
        super(FocalLoss, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
        self.gamma = gamma
        self.device = device

    def forward(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha[None, :] * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha, gamma, focal_weight=0.7, dice_weight=0.3, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def dice_loss(self, inputs, targets):
        intersection = (inputs * targets).sum(dim=0)
        sum_pred = inputs.sum(dim=0)
        sum_true = targets.sum(dim=0)
        dice = (2. * intersection + self.smooth) / (sum_pred + sum_true + self.smooth)
        return 1 - dice.mean()

    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + self.dice_weight * self.dice_loss(inputs, targets)


# ============================================================================
# 评估指标
# ============================================================================

def compute_ts_score(y_true, y_pred, class_idx=None):
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("Inputs must be 2D arrays for multi-label.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    n_classes = y_true.shape[1]
    ts_scores = []
    for i in range(n_classes):
        if class_idx is not None and i != class_idx:
            continue
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        tp = cm[1, 1] if cm.shape[0] > 1 else 0
        fp = cm[0, 1] if cm.shape[0] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        ts = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        ts_scores.append(ts)
    return np.mean(ts_scores) if ts_scores else 0.0


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    per_class_acc = [accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    metrics['accuracy'] = np.mean(per_class_acc)
    metrics['f1_macro'] = sk_f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc_scores = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0 and y_prob[:, i].sum() > 0:
            auc_scores.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    metrics['roc_auc'] = np.mean(auc_scores) if auc_scores else 0.0
    metrics['ts'] = compute_ts_score(y_true, y_pred)
    pod_scores, far_scores, csi_scores = [], [], []
    for i in range(y_true.shape[1]):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        tp = cm[1, 1] if cm.shape[0] > 1 else 0
        fp = cm[0, 1] if cm.shape[0] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        pod_scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        far_scores.append(fp / (tp + fp) if (tp + fp) > 0 else 0.0)
        csi_scores.append(tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0)
    metrics['pod'] = np.mean(pod_scores) if pod_scores else 0.0
    metrics['far'] = np.mean(far_scores) if far_scores else 0.0
    metrics['csi'] = np.mean(csi_scores) if csi_scores else 0.0
    return metrics


# ============================================================================
# 训练函数
# ============================================================================

def train_model(model, train_loader, val_loader, timestamp_list, model_name='model',
                lr=0.002, alpha=None, gamma=5, num_epochs=100, patience=20,
                trial_id=None, is_final=False, focal_weight=0.7):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, 'output.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_labels = np.concatenate([batch[4].numpy() for batch in train_loader])
    label_dist = np.mean(train_labels, axis=0)
    if alpha is None:
        alpha = [1 / max(d, 1e-6) for d in label_dist]
        alpha_sum = sum(alpha)
        alpha = [a / alpha_sum for a in alpha]
    logging.info(f"Dynamic alpha for {model_name}: {alpha}")
    print(f"Dynamic alpha: {alpha}")
    criterion = CombinedLoss(alpha=alpha, gamma=gamma, focal_weight=focal_weight, dice_weight=0.3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    best_ts = 0
    best_metrics = {}
    patience_counter = 0
    epoch_records = []
    train_loss_list, val_loss_list, lr_history = [], [], []

    if is_final:
        suffix = '_final'
    elif trial_id is not None:
        suffix = f'_trial_{trial_id}'
    else:
        suffix = ''
    model_file = os.path.join(output_dir, f'model_{model_name}{suffix}.pth')
    metrics_file = os.path.join(output_dir, f'metrics_{model_name}{suffix}.csv')
    error_file = os.path.join(output_dir, f'errors_{model_name}{suffix}.csv')

    if os.path.exists(metrics_file):
        logging.info(f"Skipping training: {metrics_file} exists")
        print(f"Skipping training: {metrics_file} exists")
        metrics = pd.read_csv(metrics_file).to_dict('records')[0]
        try:
            errors = pd.read_csv(error_file).to_dict('records')
        except Exception:
            errors = []
        return metrics, errors

    val_labels_all = np.concatenate([batch[4].numpy() for batch in val_loader])
    logging.info(f"Train label dist: {label_dist}")
    logging.info(f"Val label dist: {np.mean(val_labels_all, axis=0)}")

    try:
        if os.path.exists(model_file):
            logging.info(f"Loading existing model from {model_file}")
            print(f"Loading existing model from {model_file}")
            model.load_state_dict(torch.load(model_file))
            model.eval()
            y_true, y_pred, y_prob, y_errors = [], [], [], []
            val_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    r, s, a, o, lbl, idx, bi = [x.to(device) for x in batch]
                    outputs = model(r, s, a, o)
                    loss = criterion(outputs, lbl)
                    val_loss += loss.item()
                    preds = (outputs > 0.5).float()
                    y_errors.extend([(idx.item(), pred.tolist(), label.tolist(), timestamp_list[bi.item()])
                                     for idx, pred, label, bi in zip(idx, preds, lbl, bi)])
                    y_true.extend(lbl.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_prob.extend(outputs.cpu().numpy())
            val_loss /= len(val_loader)
            y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
            metrics = compute_metrics(y_true, y_pred, y_prob)
            metrics['val_loss'] = val_loss
            pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
            pd.DataFrame(y_errors, columns=['index', 'prediction', 'label', 'timestamp']).to_csv(error_file, index=False)
            return metrics, y_errors

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                r, s, a, o, lbl, idx, bi = [x.to(device) for x in batch]
                optimizer.zero_grad()
                outputs = model(r, s, a, o)
                loss = criterion(outputs, lbl)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss /= len(train_loader)
            scheduler.step()
            train_loss_list.append(train_loss)
            lr_history.append(optimizer.param_groups[0]['lr'])

            model.eval()
            y_true, y_pred, y_prob, y_errors = [], [], [], []
            val_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    r, s, a, o, lbl, idx, bi = [x.to(device) for x in batch]
                    outputs = model(r, s, a, o)
                    loss = criterion(outputs, lbl)
                    val_loss += loss.item()
                    preds = (outputs > 0.5).float()
                    y_errors.extend([(idx.item(), pred.tolist(), label.tolist(), timestamp_list[bi.item()])
                                     for idx, pred, label, bi in zip(idx, preds, lbl, bi)])
                    y_true.extend(lbl.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_prob.extend(outputs.cpu().numpy())

            val_loss /= len(val_loader)
            val_loss_list.append(val_loss)
            y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
            metrics = compute_metrics(y_true, y_pred, y_prob)
            metrics['val_loss'] = val_loss
            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"TS={metrics['ts']:.4f}, POD={metrics['pod']:.4f}")

            epoch_records.append({
                'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                'accuracy': metrics['accuracy'], 'f1_macro': metrics['f1_macro'],
                'roc_auc': metrics['roc_auc'], 'ts': metrics['ts'],
                'pod': metrics['pod'], 'far': metrics['far'], 'csi': metrics['csi']
            })

            if metrics['ts'] > best_ts:
                best_ts = metrics['ts']
                best_metrics = metrics.copy()
                errors = y_errors
                checkpoint = {'state_dict': model.state_dict(), 'train_loss': train_loss_list,
                              'val_loss': val_loss_list, 'lr_history': lr_history}
                torch.save(checkpoint, model_file)
                print(f"Saved best checkpoint at epoch {epoch + 1}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        pd.DataFrame(epoch_records).to_csv(metrics_file, index=False)
        if os.path.exists(error_file):
            os.remove(error_file)
        pd.DataFrame(errors, columns=['index', 'prediction', 'label', 'timestamp']).to_csv(error_file, index=False)
        print(f"Best Metrics for {model_name}: {best_metrics}")
        return best_metrics, errors

    except Exception as e:
        logging.error(f"Trial {trial_id} failed: {str(e)}")
        print(f"Trial {trial_id} failed: {str(e)}")
        raise


# ============================================================================
# 超参数调优
# ============================================================================

def hyperparameter_tune(train_loader, val_loader, timestamp_list, model_name='model', n_trials=20):
    output_dir_hp = os.path.join(base_dir, 'output')
    os.makedirs(output_dir_hp, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir_hp, 'output.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    param_grid = {
        'lr': [0.0001, 0.0005, 0.001],
        'alpha': [[0.8, 0.9, 0.95], [0.85, 0.95, 0.98], [0.85, 0.95, 1.0]],
        'gamma': [3, 4, 5],
        'num_epochs': [50],
        'focal_weight': [0.6, 0.7, 0.8],
        'patience': [50]
    }
    default_params = {'lr': 0.0005, 'alpha': [0.8, 0.95, 1.0], 'gamma': 3,
                      'focal_weight': 0.7, 'num_epochs': 50, 'patience': 50}
    results = []
    params_file = os.path.join(output_dir_hp, f'trial_params_{model_name}.csv')
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))
    print(f"Starting hyperparameter tuning for {model_name}: {len(all_combinations)} combinations")

    for trial_idx, combo in enumerate(all_combinations):
        trial_id = trial_idx + 1
        model_file = os.path.join(output_dir_hp, f'model_{model_name}_trial_{trial_id}.pth')
        metrics_file = os.path.join(output_dir_hp, f'metrics_{model_name}_trial_{trial_id}.csv')

        if os.path.exists(metrics_file):
            print(f"Skipping trial {trial_id}: already exists")
            metrics_df = pd.read_csv(metrics_file)
            results.append({
                'trial': trial_id,
                'lr': None, 'alpha': '[]', 'gamma': None, 'num_epochs': None,
                'patience': None, 'focal_weight': None,
                'accuracy': metrics_df['accuracy'].max(),
                'f1_macro': metrics_df['f1_macro'].max(),
                'roc_auc': metrics_df['roc_auc'].max(),
                'ts': metrics_df['ts'].max(),
                'pod': metrics_df['pod'].max() if 'pod' in metrics_df.columns else None,
                'far': metrics_df['far'].min() if 'far' in metrics_df.columns else None,
                'csi': metrics_df['csi'].max() if 'csi' in metrics_df.columns else None
            })
            continue

        params = dict(zip(keys, combo))
        print(f"\nTrial {trial_id}/{len(all_combinations)}: {params}")
        trial_params = {'trial': trial_id, 'lr': params['lr'], 'alpha': str(params['alpha']),
                        'gamma': params['gamma'], 'num_epochs': params['num_epochs'],
                        'patience': params['patience'], 'focal_weight': params['focal_weight']}
        pd.DataFrame([trial_params]).to_csv(params_file, mode='a' if os.path.exists(params_file) else 'w',
                                            header=not os.path.exists(params_file), index=False)

        model = WeatherModel(use_optical_flow=(model_name == 'flow')).to(device)
        model.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, (nn.Conv2d, nn.Linear)) else None)

        try:
            best_metrics, errors = train_model(
                model, train_loader, val_loader, timestamp_list, model_name,
                lr=params['lr'], alpha=params['alpha'], gamma=params['gamma'],
                num_epochs=params['num_epochs'], patience=params['patience'],
                trial_id=trial_id, focal_weight=params['focal_weight']
            )
            results.append({
                'trial': trial_id, 'lr': params['lr'], 'alpha': str(params['alpha']),
                'gamma': params['gamma'], 'num_epochs': params['num_epochs'],
                'patience': params['patience'], 'focal_weight': params['focal_weight'],
                'accuracy': best_metrics['accuracy'], 'f1_macro': best_metrics['f1_macro'],
                'roc_auc': best_metrics['roc_auc'], 'ts': best_metrics['ts'],
                'pod': best_metrics.get('pod'), 'far': best_metrics.get('far'),
                'csi': best_metrics.get('csi')
            })
        except Exception as e:
            print(f"Trial {trial_id} failed: {str(e)}")
            continue

    results_file = os.path.join(output_dir_hp, f'hyperparam_results_{model_name}.csv')
    pd.DataFrame(results).to_csv(results_file, index=False)
    print(f"Saved hyperparameter results to {results_file}")

    if results:
        best_trial = max(results, key=lambda x: x['ts'])
        print(f"Best Trial for {model_name}: {best_trial}")
        return best_trial
    print(f"No successful trials for {model_name}, returning defaults")
    return {'trial': 0, 'lr': default_params['lr'], 'alpha': str(default_params['alpha']),
            'gamma': default_params['gamma'], 'num_epochs': default_params['num_epochs'],
            'patience': default_params['patience'], 'focal_weight': default_params['focal_weight'],
            'accuracy': 0, 'f1_macro': 0, 'roc_auc': 0, 'ts': 0}


# ============================================================================
# 保存样本名称
# ============================================================================

def save_sample_names(all_positive, all_negative, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    positive_file = os.path.join(out_dir, "positive_samples.txt")
    with open(positive_file, 'w', encoding='utf-8') as f:
        f.write("正样本列表\n" + "=" * 60 + "\n")
        for i, (date, target_ts, labels) in enumerate(all_positive, 1):
            ts_str = target_ts.strftime('%Y-%m-%d %H:%M:%S')
            label_names = []
            if labels[0] == 1: label_names.append("雷暴")
            if labels[1] == 1: label_names.append("短时强降水")
            if labels[2] == 1: label_names.append("大风")
            label_str = "、".join(label_names) if label_names else "无"
            f.write(f"{i:3d}. {date} {ts_str} [{label_str}]\n")
    print(f"正样本: {positive_file} ({len(all_positive)}个)")

    negative_file = os.path.join(out_dir, "negative_samples.txt")
    with open(negative_file, 'w', encoding='utf-8') as f:
        f.write("负样本列表\n" + "=" * 60 + "\n")
        for i, (date, target_ts) in enumerate(all_negative, 1):
            ts_str = target_ts.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{i:3d}. {date} {ts_str}\n")
    print(f"负样本: {negative_file} ({len(all_negative)}个)")


# ============================================================================
# 分析函数
# ============================================================================

def predict_fn(model, data):
    if len(data) == 4:
        radar, satellite, awos, flow = data
    elif len(data) == 3:
        radar, satellite, awos = data
        flow = None
    else:
        raise ValueError(f"Unexpected number of inputs: {len(data)}")
    model.eval()
    with torch.no_grad():
        radar = radar.to(device); satellite = satellite.to(device); awos = awos.to(device)
        if model.use_optical_flow and flow is not None:
            flow = flow.to(device)
            outputs = model(radar, satellite, awos, flow)
        else:
            outputs = model(radar, satellite, awos)
        return outputs.cpu().numpy()


def analyze_features_and_errors(model_flow, model_no_flow, val_data, val_labels,
                                val_timestamps, checkpoint_flow, checkpoint_no_flow):
    def plot_loss_curve(train_loss, val_loss, title, file_prefix):
        if len(train_loss) == 0 or len(val_loss) == 0:
            return
        epochs = np.arange(1, len(train_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(title)
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{file_prefix}_loss_curve.png')); plt.close()
        pd.DataFrame({'epoch': epochs, 'train_loss': train_loss, 'val_loss': val_loss}).to_csv(
            os.path.join(output_dir, f'{file_prefix}_loss.csv'), index=False)

    def plot_lr_curve(lr_history, title, file_prefix):
        if len(lr_history) == 0: return
        epochs = np.arange(1, len(lr_history) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, lr_history, label='LR'); plt.xlabel('Epoch')
        plt.ylabel('LR'); plt.title(title); plt.yscale('log')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{file_prefix}_lr_curve.png')); plt.close()

    plot_loss_curve(checkpoint_flow.get('train_loss', []), checkpoint_flow.get('val_loss', []),
                    'Loss Curve (With U-Net Flow)', 'flow_unet')
    plot_lr_curve(checkpoint_flow.get('lr_history', []), 'LR Curve (With U-Net Flow)', 'flow_unet')
    plot_loss_curve(checkpoint_no_flow.get('train_loss', []), checkpoint_no_flow.get('val_loss', []),
                    'Loss Curve (Without Flow)', 'no_flow')
    plot_lr_curve(checkpoint_no_flow.get('lr_history', []), 'LR Curve (Without Flow)', 'no_flow')

    val_labels_np = np.array(val_labels)
    classes = ['thunderstorm', 'heavy_rain', 'strong_wind']

    val_data_flow = [v.clone().detach().cpu() if isinstance(v, torch.Tensor) else v for v in val_data]
    val_data_no_flow = val_data_flow[:3]

    preds_flow_prob = predict_fn(model_flow, val_data_flow)
    preds_no_flow_prob = predict_fn(model_no_flow, val_data_no_flow)
    preds_flow_bin = (preds_flow_prob > 0.5).astype(int)
    preds_no_flow_bin = (preds_no_flow_prob > 0.5).astype(int)

    metrics_flow = compute_metrics(val_labels_np, preds_flow_bin, preds_flow_prob)
    metrics_no_flow = compute_metrics(val_labels_np, preds_no_flow_bin, preds_no_flow_prob)
    print(f"\nWith U-Net Flow: Acc={metrics_flow['accuracy']:.4f}, F1={metrics_flow['f1_macro']:.4f}, "
          f"AUC={metrics_flow['roc_auc']:.4f}, TS={metrics_flow['ts']:.4f}")
    print(f"Without Flow:    Acc={metrics_no_flow['accuracy']:.4f}, F1={metrics_no_flow['f1_macro']:.4f}, "
          f"AUC={metrics_no_flow['roc_auc']:.4f}, TS={metrics_no_flow['ts']:.4f}")

    pd.DataFrame([metrics_flow]).to_csv(os.path.join(output_dir, 'performance_flow_unet.csv'), index=False)
    pd.DataFrame([metrics_no_flow]).to_csv(os.path.join(output_dir, 'performance_no_flow.csv'), index=False)

    # Feature importance
    def custom_permutation_importance(model, data_list, y_true, features, n_repeats=5):
        base_prob = predict_fn(model, data_list)
        base_bin = (base_prob > 0.5).astype(int)
        base_score = compute_ts_score(y_true, base_bin)
        importances = np.zeros((len(features), n_repeats))
        for i in range(len(features)):
            for r in range(n_repeats):
                shuf_data = [d.clone() for d in data_list]
                shuf_data[i] = shuf_data[i][torch.randperm(shuf_data[i].shape[0])]
                shuf_prob = predict_fn(model, shuf_data)
                shuf_bin = (shuf_prob > 0.5).astype(int)
                importances[i, r] = base_score - compute_ts_score(y_true, shuf_bin)
        return {'importances_mean': np.mean(importances, axis=1),
                'importances_std': np.std(importances, axis=1)}

    try:
        features_flow = ['Radar', 'Satellite', 'AWOS', 'U-Net Flow']
        imp_flow = custom_permutation_importance(model_flow, val_data_flow, val_labels_np, features_flow)
        print("\nFeature Importance (With U-Net Flow):")
        for i, (m, s) in enumerate(zip(imp_flow['importances_mean'], imp_flow['importances_std'])):
            print(f"  {features_flow[i]}: {m:.4f} +/- {s:.4f}")
        pd.DataFrame({'feature': features_flow, 'mean': imp_flow['importances_mean'],
                      'std': imp_flow['importances_std']}).to_csv(
            os.path.join(output_dir, 'feature_importance_flow_unet.csv'), index=False)
    except Exception as e:
        print(f"Feature importance failed: {e}")

    # Confusion matrices
    for i, cls in enumerate(classes):
        for mode, preds in [('With_UNet_Flow', preds_flow_bin), ('Without_Flow', preds_no_flow_bin)]:
            cm = confusion_matrix(val_labels_np[:, i], preds[:, i])
            plt.figure(figsize=(6, 4))
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title(f'CM ({mode} - {cls})'); plt.colorbar()
            for x, y in np.ndindex(cm.shape):
                plt.text(y, x, str(cm[x, y]), ha="center",
                         color="white" if cm[x, y] > cm.max() / 2 else "black")
            plt.ylabel('True'); plt.xlabel('Pred')
            plt.savefig(os.path.join(output_dir, f'cm_{mode}_{cls}.png')); plt.close()

    # ROC curves
    for mode, preds_prob in [('With UNet Flow', preds_flow_prob), ('Without Flow', preds_no_flow_prob)]:
        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(val_labels_np[:, i], preds_prob[:, i])
            plt.plot(fpr, tpr, label=f'{cls} (AUC={auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC ({mode})')
        plt.legend(loc="lower right"); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'roc_{mode.lower().replace(" ", "_")}.png')); plt.close()

    # Model comparison bar chart
    metrics_names = ['POD', 'FAR', 'CSI']
    values_f = [metrics_flow['pod'], metrics_flow['far'], metrics_flow['csi']]
    values_nf = [metrics_no_flow['pod'], metrics_no_flow['far'], metrics_no_flow['csi']]
    plt.figure(figsize=(8, 6))
    x = np.arange(len(metrics_names)); w = 0.35
    plt.bar(x - w/2, values_f, w, label='With U-Net Flow')
    plt.bar(x + w/2, values_nf, w, label='Without Flow')
    plt.xlabel('Metrics'); plt.ylabel('Value')
    plt.title('Model Comparison (U-Net Flow vs No Flow)')
    plt.xticks(x, metrics_names); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'model_comparison_unet.png')); plt.close()
    print("Analysis completed")


def run_analysis():
    samples = load_data()
    val_data = [
        torch.FloatTensor(samples['radar'][int(0.8 * len(samples['radar'])):]),
        torch.FloatTensor(samples['satellite'][int(0.8 * len(samples['satellite'])):]),
        torch.FloatTensor(samples['awos'][int(0.8 * len(samples['awos'])):]),
        torch.FloatTensor(samples['optical_flow'][int(0.8 * len(samples['optical_flow'])):])
    ]
    val_labels = samples['labels'][int(0.8 * len(samples['labels'])):]
    val_timestamps = samples['timestamps'][int(0.8 * len(samples['timestamps'])):]

    checkpoint_flow = torch.load(os.path.join(output_dir, 'model_flow_final.pth'))
    model_flow = WeatherModel(use_optical_flow=True).to(device)
    if isinstance(checkpoint_flow, dict) and 'state_dict' in checkpoint_flow:
        model_flow.load_state_dict(checkpoint_flow['state_dict'])
    else:
        model_flow.load_state_dict(checkpoint_flow)

    checkpoint_no_flow = torch.load(os.path.join(output_dir, 'model_no_flow_final.pth'))
    model_no_flow = WeatherModel(use_optical_flow=False).to(device)
    if isinstance(checkpoint_no_flow, dict) and 'state_dict' in checkpoint_no_flow:
        model_no_flow.load_state_dict(checkpoint_no_flow['state_dict'])
    else:
        model_no_flow.load_state_dict(checkpoint_no_flow)

    analyze_features_and_errors(model_flow, model_no_flow, val_data, val_labels,
                                val_timestamps, checkpoint_flow, checkpoint_no_flow)


# ============================================================================
# 主函数
# ============================================================================

def main():
    output_dir_main = os.path.join(base_dir, 'output_best')
    os.makedirs(output_dir_main, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir_main, 'output.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Step 1: U-Net 光流预训练
        print("\n" + "=" * 60)
        print("Step 1: U-Net 光流预训练 (训练集:验证集 = 9:1)")
        print("=" * 60)
        flow_unet = pretrain_flow_unet()

        # Step 2: 获取样本池
        all_positive, all_negative = sample_selection(info_excel, label_dir)
        print(f"潜在正样本池: {len(all_positive)}, 负: {len(all_negative)}")
        save_sample_names(all_positive, all_negative, output_dir_main)

        # Step 3: 加载数据（使用 U-Net 光流）
        samples = load_data(flow_unet=flow_unet)
        logging.info(f"Loaded samples: {len(samples['radar'])}")
        print(f"Loaded samples: {len(samples['radar'])}")

        # Step 4: 准备张量
        radar_tensor = torch.FloatTensor(samples['radar'])
        satellite_tensor = torch.FloatTensor(samples['satellite'])
        awos_tensor = torch.FloatTensor(samples['awos'])
        flow_tensor = torch.FloatTensor(samples['optical_flow'])
        label_tensor = torch.FloatTensor(samples['labels'])
        indices = torch.LongTensor(samples['indices'])
        timestamp_list = samples['timestamps']

        print(f"Tensor shapes - radar: {radar_tensor.shape}, satellite: {satellite_tensor.shape}, "
              f"awos: {awos_tensor.shape}, flow: {flow_tensor.shape}, labels: {label_tensor.shape}")

        label_array = label_tensor.numpy()
        if label_array.ndim == 2 and label_array.shape[1] == 3:
            print(f"Label counts: {np.sum(label_array, axis=0)}")

        # Step 5: 划分数据集
        train_idx, val_idx = train_test_split(range(len(label_tensor)), test_size=0.2, random_state=42)
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        dataset = TensorDataset(radar_tensor, satellite_tensor, awos_tensor, flow_tensor,
                                label_tensor, indices, torch.arange(len(radar_tensor)))
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        num_workers = 2
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                                  num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                                num_workers=num_workers, pin_memory=False)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Step 6: 超参数调优
        print("\nStarting hyperparameter tuning...")
        best_trial_flow = hyperparameter_tune(train_loader, val_loader, timestamp_list, model_name='flow')
        best_trial_no_flow = hyperparameter_tune(train_loader, val_loader, timestamp_list, model_name='no_flow')

        # Step 7: 训练最终模型
        print("\nTraining final models...")
        model_flow = WeatherModel(use_optical_flow=True).to(device)
        model_flow.apply(lambda m: nn.init.xavier_uniform_(m.weight)
                         if isinstance(m, (nn.Conv2d, nn.Linear)) else None)
        alpha_flow = ast.literal_eval(best_trial_flow['alpha'])
        best_metrics_flow, _ = train_model(
            model_flow, train_loader, val_loader, timestamp_list, model_name='flow',
            lr=best_trial_flow['lr'], alpha=alpha_flow, gamma=best_trial_flow['gamma'],
            num_epochs=best_trial_flow['num_epochs'], patience=best_trial_flow['patience'],
            trial_id=0, is_final=True, focal_weight=best_trial_flow['focal_weight']
        )

        model_no_flow = WeatherModel(use_optical_flow=False).to(device)
        model_no_flow.apply(lambda m: nn.init.xavier_uniform_(m.weight)
                            if isinstance(m, (nn.Conv2d, nn.Linear)) else None)
        alpha_no_flow = ast.literal_eval(best_trial_no_flow['alpha'])
        best_metrics_no_flow, _ = train_model(
            model_no_flow, train_loader, val_loader, timestamp_list, model_name='no_flow',
            lr=best_trial_no_flow['lr'], alpha=alpha_no_flow, gamma=best_trial_no_flow['gamma'],
            num_epochs=best_trial_no_flow['num_epochs'], patience=best_trial_no_flow['patience'],
            trial_id=0, is_final=True, focal_weight=best_trial_no_flow['focal_weight']
        )

        # Step 8: 模型对比
        comparison = pd.DataFrame({
            'Model': ['NoFlow', 'U-Net Flow'],
            'Accuracy': [best_metrics_no_flow['accuracy'], best_metrics_flow['accuracy']],
            'F1_macro': [best_metrics_no_flow['f1_macro'], best_metrics_flow['f1_macro']],
            'ROC_AUC': [best_metrics_no_flow['roc_auc'], best_metrics_flow['roc_auc']],
            'TS': [best_metrics_no_flow['ts'], best_metrics_flow['ts']],
            'POD': [best_metrics_no_flow['pod'], best_metrics_flow['pod']],
            'FAR': [best_metrics_no_flow['far'], best_metrics_flow['far']],
            'CSI': [best_metrics_no_flow['csi'], best_metrics_flow['csi']]
        })
        comparison.to_csv(os.path.join(output_dir_main, 'model_comparison_unet.csv'), index=False)
        print("\nModel Comparison:")
        print(comparison)

        # Step 9: 分析
        print("\nRunning analysis...")
        run_analysis()

    except Exception as e:
        logging.error(f"Main function failed: {str(e)}")
        print(f"Main function failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
