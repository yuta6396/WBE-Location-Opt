import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib import animation
import imageio
import matplotlib
matplotlib.use('agg')  # Xサーバを使わずに描画
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo

from config import time_interval_sec
"""
history fileがt=3600(sec)しかないためにgifにできないが、分割を増やしてやればいけそう
"""
jst = pytz.timezone('Asia/Tokyo')# 日本時間のタイムゾーンを設定
current_time = datetime.now(jst).strftime("%m-%d-%H-%M")
file_dir = f"result/windfield/{current_time}/"
os.makedirs(file_dir, exist_ok=True)

# netCDFファイルの読み込み
ncfile = "merged_history_3600.pe000000.nc"
ds = nc.Dataset(ncfile)

# データの読み込み
data = {
    #'time': ds.variables['time'][:],
    'x': ds.variables['x'][:],
    'y': ds.variables['y'][:],
    'z': ds.variables['z'][:],
    'V': ds.variables['V'][:],   # Y方向水平速度成分(3次元)
    'W': ds.variables['W'][:],   # 鉛直方向速度成分(3次元)
    'PREC': ds.variables['PREC'][:]  # 降水強度(2次元)
}
# print(f"Shape of data['x']: {data['x'].shape}")
# print(f"Shape of data['y']: {data['y'].shape}")
# print(f"Shape of data['z']: {data['z'].shape}")
print(f"Shape of data['V']: {data['V'].shape}")
print(f"Shape of data['W']: {data['W'].shape}")
# print(f"Shape of data['PREC']: {data['PREC'].shape}")
# y と z のグリッドを作成
Y, Z = np.meshgrid(data['y'], data['z'])

# プロットの準備
fig, ax = plt.subplots()
ax.set_xlim(0, 2e4)
ax.set_ylim(0, 2e4)
ax.set_xlabel('y')
ax.set_ylabel('z')

scale = 10e1  # 矢印の長さの調整
quiver_plot = None

# GIF用のフレーム保存リスト
frames = []

# アニメーションを作成
def update_frame(i):
    global quiver_plot
    ax.clear()  # 以前のプロットを消去
    ax.set_xlim(0, 2e4)
    ax.set_ylim(0, 2e4)
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_title(f'time {i}')
    
    tmpV = np.squeeze(data['V'][ i, :, :, :]) * scale
    tmpW = np.squeeze(data['W'][ i, :, :, :]) * scale

    # プロット（矢印）
    quiver_plot = ax.quiver(Y, Z, tmpV.T, tmpW.T)
    plt.draw()
    
    # フレームを保存
    fig.canvas.draw() #Gif
    fig.savefig(f"{file_dir}frame_{i}.png")  # 各フレームをファイルとして保存
    # image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # frames.append(image)

    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')

    # RGBAとしてリシェイプ (4チャンネル)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frames.append(image)

# フレームごとにプロットを更新
for i in range(data['V'].shape[0]):
    update_frame(i)

# GIFを作成
imageio.mimsave(f'{file_dir}{ncfile}.gif', frames, fps=1)  # FPS=1で遅延時間を調整
