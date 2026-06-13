import torch
import pygame
from torch.utils.data import Dataset, DataLoader
import numpy as np
class CircleSegmentationDataset(Dataset):
    def __init__(self, num_samples=800):
        super(CircleSegmentationDataset, self).__init__()
        self.num_samples = num_samples # 把變數存進自己裡面
        # 1. 依簡報規定每張圖必須隨機產⽣三個｢純紅､純綠､純藍｣且｢互不重疊｣的實心圓
        # 2. 必須使⽤線代兩點距離公式進⾏碰撞偵測d = sqrt(Δx^2 + Δy^2) > r1 + r2 ｡
        # 3. 原始影像維度(num_samples, 128, 128, 3)
        # 4. 原始遮罩維度(num_samples, 128, 128)前景為255，背景為 0
        self.X, self.Y = self._generate_raw_data() # images存進X，mask存進Y
        
  
    def _generate_raw_data(self):
        # TODO: 實作不重疊圓形生成演算法
        # 請在此處初始化畫布利用循環與幾何距離公式生成資料並回傳 (images, masks)
        images = np.zeros((self.num_samples, 128, 128, 3), dtype=np.float32)
        masks = np.zeros((self.num_samples, 128, 128), dtype=np.float32)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # 顏色為純紅、純綠、純藍
      
        for i in range(self.num_samples):
            circles = [] # 記錄圓的position & radius (x, y, r)
            num_circles = np.random.randint(2, 7)

            for _ in range(num_circles):
              color_idx = np.random.randint(0, 3)
              chosen_color = colors[color_idx]

              attempts = 0 # 嘗試次數
              success = False

              while attempts < 100:
                  r = np.random.randint(10, 25) # (自訂)半徑的適當範圍(顯示 & run時比較不會出事)
                  x = np.random.randint(r, 128-r) # 左右都留一個半徑的距離以免超出畫布
                  y = np.random.randint(r, 128-r)
                    
                  overlap = False
                  for cx, cy, cr in circles: # 在畫布上的(原本的)圓
                      distance = np.sqrt((x-cx)**2 + (y-cy)**2)
                      if distance < (r + cr): # 距離小於r1+r2:重疊
                          overlap = True
                          break
                  if not overlap:
                      circles.append((x, y, r)) # 存入符合的圓名單

                      # 建立128*128的棋盤
                      Y, X = np.ogrid[:128, :128] 
                      # 找出所有距離圓心小於 r 的點
                      distance_from_center = np.sqrt((X - x)**2 + (Y - y)**2) 
                      mask_area = (distance_from_center <= r)
                      # 塗色
                      images[i][mask_area] = chosen_color
                      masks[i][mask_area] = 255
                      break
                  attempts += 1
        return torch.tensor(images), torch.tensor(masks)
      
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        """
        """
        raw_image = self.X[idx]
        raw_mask = self.Y[idx]
        # TODO: 實作 Patch 序列化前處理邏輯(切成8*8)

        x_patch = raw_image.reshape(16, 8, 16, 8, 3) # 將128拆成16份的8
        x_patch = x_patch.permute(0, 2, 1, 3, 4) # 調換位置，讓"區塊的座標(行,列)"跟"區塊的像素(行,列)"歸在一起
        x_patch = x_patch.reshape(256, 192) # (16, 16, 8, 8, 3) -> (256, 192)
        y_patch = raw_mask.reshape(16, 8 ,16 ,8)
        y_patch = y_patch.permute(0, 2, 1, 3)
        y_patch = y_patch.reshape(256, 64)
        
        return x_patch, y_patch
def get_dataloader(batch_size=16, num_samples=800, shuffle=True):
    """ 最終對接接口 打包成 PyTorch DataLoader輸出維度必為 (B, 256, 64) """
    dataset = CircleSegmentationDataset(num_samples=num_samples)    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
