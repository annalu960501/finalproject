import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
class CircleSegmentationDataset(Dataset):
    def __init__(self, num_samples=800):
        super(CircleSegmentationDataset, self).__init__()
        self.num_samples = num_samples # 把變數存進自己裡面(?)
        # 1. 依簡報規定每張圖必須隨機產⽣三個｢純紅､純綠､純藍｣且｢互不重疊｣的實心圓
        # 2. 必須使⽤線代兩點距離公式進⾏碰撞偵測d = sqrt(Δx^2 + Δy^2) > r1 + r2 ｡
        # 3. 原始影像維度(num_samples, 128, 128, 3)
        # 4. 原始遮罩維度(num_samples, 128, 128)前景為255，背景為 0
        self.X, self.Y = self._generate_raw_data() # images存進X，mask存進Y
        
  
    def _generate_raw_data(self):
        # TODO: 實作不重疊圓形生成演算法
        
        # 初始空白圖片、遮罩(800張，128*128)(用浮點數存起來)
        images = np.zeros((self.num_samples, 128, 128, 3), dtype=np.float32)
        masks = np.zeros((self.num_samples, 128, 128, 1), dtype=np.float32)
      
        # 請在此處初始化畫布利用循環與幾何距離公式生成資料並回傳 (images, masks)
      
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        """
        """
        raw_image = self.X[idx]
        raw_mask = self.Y[idx]
        # TODO: 實作 Patch 序列化前處理邏輯(切成8*8)
        x_patch = torch.zeros(256, 64, dtype=torch.float32)
        y_patch = torch.zeros(256, 64, dtype=torch.float32)
        return x_patch, y_patch
def get_dataloader(batch_size=16, num_samples=800, 
shuffle=True):
    """ 最終對接接口 打包成 PyTorch DataLoader輸出維度必為 (B, 256, 
64) """
    dataset = 
CircleSegmentationDataset(num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, 
shuffle=shuffle)
