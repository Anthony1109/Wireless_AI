import numpy as np
import os

os.makedirs('tools', exist_ok=True)

# 系統預期的通道路徑數量 (通常等於 Cyclic Prefix 的長度)
L = 16  

train_samples = 10000
test_samples = 2000

print("Generating channel data with correct dimensions (16 taps)...")

# 產生形狀為 (樣本數, 16) 的複數矩陣
channel_train = (np.random.randn(train_samples, L) + 1j * np.random.randn(train_samples, L)) / np.sqrt(2)
channel_test = (np.random.randn(test_samples, L) + 1j * np.random.randn(test_samples, L)) / np.sqrt(2)

np.save('tools/channel_train.npy', channel_train)
np.save('tools/channel_test.npy', channel_test)

print("Channel data generated and saved to 'tools/channel_train.npy' and 'tools/channel_test.npy'.")