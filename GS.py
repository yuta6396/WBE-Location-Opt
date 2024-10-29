import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
from skopt import gp_minimize
from skopt.space import Integer
import pandas as pd
import seaborn as sns
# 時刻を計測するライブラリ
import time
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo

from optimize import random_search
from analysis import *
from make_directory import make_directory
from config import time_interval_sec, bound

matplotlib.use('Agg')

"""
BORSのシミュレーション
"""

#### User 設定変数 ##############

input_var = "MOMY" # MOMY, RHOT, QVから選択
input_size = -1 # 変更の余地あり
Alg_vec = ["GS"]
num_input_grid = 1 # ある一つの地点を制御
Opt_purpose = "MinSum" #MinSum, MinMax, MaxSum, MaxMinから選択

dpi = 75 # 画像の解像度　スクリーンのみなら75以上　印刷用なら300以上
colors6  = ['#4c72b0', '#f28e2b', '#55a868', '#c44e52'] # 論文用の色
###############################
jst = pytz.timezone('Asia/Tokyo')# 日本時間のタイムゾーンを設定
current_time = datetime.now(jst).strftime("%m-%d-%H-%M")

"""
gp_minimize で獲得関数を指定: acq_func。
gp_minimize の呼び出しにおける主要なオプションは次の通りです。
"EI": Expected Improvement
"PI": Probability of Improvement
"LCB": Lower Confidence Bound
"gp_hedge": これらの獲得関数をランダムに選択し、探索を行う

EI は、探索と活用のバランスを取りたい場合に多く使用されます。
PI は、最速で最良の解を見つけたい場合に適していますが、早期に探索が止まるリスクがあります。
LCB は、解の探索空間が不確実である場合に有効で、保守的に最適化を進める場合に使用されます
"""




nofpe = 2
fny = 2
fnx = 1
run_time = 20

varname = 'PREC'

init_file = "init_00000101-000000.000.pe######.nc"
org_file = "init_00000101-000000.000.pe######.org.nc"
history_file = "history.pe######.nc"

orgfile = f'no-control_{str(time_interval_sec)}.pe######.nc'
file_path = os.path.dirname(os.path.abspath(__file__))
gpyoptfile=f"gpyopt.pe######.nc"


### SCALE-RM関連関数
def prepare_files(pe: int):
    """ファイルの準備と初期化を行う"""
    output_file = f"out-{input_var}.pe######.nc"
    # input file
    init = init_file.replace('######', str(pe).zfill(6))
    org = org_file.replace('######', str(pe).zfill(6))
    history = history_file.replace('######', str(pe).zfill(6))
    output = output_file.replace('######', str(pe).zfill(6))
    history_path = file_path+'/'+history
    if (os.path.isfile(history_path)):
        subprocess.run(["rm", history])
    subprocess.run(["cp", org, init])  # 初期化

    return init, output

def update_netcdf(init: str, output: str, pe: int, input_values):
    """NetCDFファイルの変数を更新する"""
    pe_this_y = 0
    print(input_values)
    Grid_y = input_values[0]
    Grid_z = input_values[1]
    if  Grid_y >= 20:
        pe_this_y = 1
        Grid_y -= 20

    with netCDF4.Dataset(init) as src, netCDF4.Dataset(output, "w") as dst:
        # グローバル属性のコピー
        dst.setncatts(src.__dict__)
        # 次元のコピー
        for name, dimension in src.dimensions.items():
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None))
        # 変数のコピーと更新
        for name, variable in src.variables.items():
            x = dst.createVariable(
                name, variable.datatype, variable.dimensions)
            dst[name].setncatts(src[name].__dict__)
            if name == input_var:
                var = src[name][:]
                if pe == pe_this_y:  # y=Grid_yのときに変更処理
                    var[Grid_y, 0, Grid_z] += input_size # (y,x,z)
                    # var[Grid_y, 0, 0] += input_size # (y,x,z)
                dst[name][:] = var
            else:
                dst[name][:] = src[name][:]

    # outputをinitにコピー
    subprocess.run(["cp", output, init])
    return init

def sim(control_input):
    """
    制御入力決定後に実際にその入力値でシミュレーションする
    """
    #control_input = [0, 0, 0] # 制御なしを見たいとき
    for pe in range(nofpe):
        init, output = prepare_files(pe)
        init = update_netcdf(init, output, pe, control_input)

    subprocess.run(["mpirun", "-n", "2", "./scale-rm", "run_R20kmDX500m.conf"])

    for pe in range(nofpe):
        gpyopt = gpyoptfile.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", history,gpyopt])
    for pe in range(nofpe):  # history処理
        fiy, fix = np.unravel_index(pe, (fny, fnx))
        nc = netCDF4.Dataset(history_file.replace('######', str(pe).zfill(6)))
        onc = netCDF4.Dataset(orgfile.replace('######', str(pe).zfill(6)))
        nt = nc.dimensions['time'].size
        nx = nc.dimensions['x'].size
        ny = nc.dimensions['y'].size
        nz = nc.dimensions['z'].size
        gx1 = nx * fix
        gx2 = nx * (fix + 1)
        gy1 = ny * fiy
        gy2 = ny * (fiy + 1)
        if pe == 0:
            dat = np.zeros((nt, nz, fny*ny, fnx*nx))
            odat = np.zeros((nt, nz, fny*ny, fnx*nx))
        dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
        odat[:, 0, gy1:gy2, gx1:gx2] = onc[varname][:]

    sum_co=np.zeros(40) #制御後の累積降水量
    sum_no=np.zeros(40) #制御前の累積降水量
    for y_i in range(40):
        for t_j in range(nt):
            if t_j > 0:
                sum_co[y_i] += dat[t_j,0,y_i,0]*time_interval_sec
                sum_no[y_i] += odat[t_j,0,y_i,0]*time_interval_sec

    sum_prec = 0
    for y_i in range(40):
        sum_prec += sum_co[y_i]
    return sum_prec



def grid_search(objective_function):
    best_score = float('inf')
    best_params = None
    # 結果を保存するためのリスト
    results = []
    
    # 各組み合わせについて評価
    cnt = 0
    for y_i in range(0, 40):
        for z_i in range(0,97):
            score = objective_function([y_i, z_i])
            score = 100*score/96.50 # 大体で制御なし⇒100%
            results.append({'Y': y_i, 'Z': z_i, 'score': score})

            cnt += 1
            if score < best_score:
                best_score = score
                best_params = [y_i, z_i]
                f.write(f"\ncnt={cnt}: params=[{y_i}, {z_i}],  best_score={best_score}")
    # 結果をデータフレームに変換
    results_df = pd.DataFrame(results)
    # ピボットテーブルの作成
    scores_pivot = results_df.pivot(index='Z', columns='Y', values='score')
    scores_pivot = scores_pivot.iloc[::-1]
    return best_params, best_score, scores_pivot


###実行
dirname = f"result/GS/{input_var}={input_size}_{current_time}"
os.makedirs(dirname, exist_ok=True)
output_file_path = os.path.join(dirname, f'summary.txt')
f = open(output_file_path, 'w')

# グリッドサーチの実行
best_params, best_score, scores_pivot = grid_search(sim)

f.write(f"\nBest parameters: {best_params}\n")
print(f"Best score: {best_score}")
# ヒートマップの描画
plt.figure(figsize=(8, 6))
sns.heatmap(scores_pivot, annot=False, cmap='viridis_r') # annot=Trueだと具体的な値表示
plt.title(f'Grid Search Accumulated PREC (%) Input:{input_var}={input_size}')
plt.xlabel('Y')
plt.ylabel('Z')
plt.savefig(f"{dirname}/heatmap.png", dpi = 300)

f.close()
