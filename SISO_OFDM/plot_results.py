import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 1. 定義你的 SNR 範圍 (假設你剛剛是跑 5 到 40 dB，每隔 5 dB 測一次)
SNR_list = np.arange(5, 45, 5) 

# 2. 讀取你剛剛存下來的 4 個 .mat 檔案
# 注意：這裡中括號 ['...'] 裡面的字串，必須是你存 .mat 時給變數取的名稱
try:
    mse_dnn = sio.loadmat('MSE_dnn_4QAM.mat')['MSE_dnn_4QAM'].flatten()
    mse_dnn_cp_free = sio.loadmat('MSE_dnn_4QAM_CP_FREE.mat')['MSE_dnn_4QAM_CP_FREE'].flatten()
    mse_mmse = sio.loadmat('MSE_mmse_4QAM.mat')['MSE_mmse_4QAM'].flatten()
    mse_mmse_cp_free = sio.loadmat('MSE_mmse_4QAM_CP_FREE.mat')['MSE_mmse_4QAM_CP_FREE'].flatten()
except KeyError as e:
    print(f"Error loading .mat files: {e}")
    # 如果報錯，請檢查你存檔時 sio.savemat() 字典裡的 key 是什麼

# 3. 開始畫圖 (使用 semilogy 畫出 Y 軸為對數刻度的圖)
plt.figure(figsize=(10, 7))

# 畫出有 CP 的情況 (實線)
plt.semilogy(SNR_list, mse_mmse, 'b-o', linewidth=2, markersize=8, label='MMSE (With CP)')
plt.semilogy(SNR_list, mse_dnn, 'r-*', linewidth=2, markersize=8, label='DNN (With CP)')

# 畫出無 CP 的情況 (虛線)
plt.semilogy(SNR_list, mse_mmse_cp_free, 'b--s', linewidth=2, markersize=8, label='MMSE (CP-Free)')
plt.semilogy(SNR_list, mse_dnn_cp_free, 'r--^', linewidth=2, markersize=8, label='DNN (CP-Free)')

# 4. 設定圖表格式
plt.xlabel('SNR (dB)', fontsize=14)
plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
plt.title('Channel Estimation Performance (4QAM)', fontsize=16)

# 開啟網格線 (包含主格線與副格線)
plt.grid(True, which="both", ls="--", alpha=0.5)

# 顯示圖例
plt.legend(fontsize=12)

# 顯示圖片 
plt.savefig('result.png', dpi=300)