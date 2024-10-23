import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
from skopt import gp_minimize
from skopt.space import Integer
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
input_size = 10 # 変更の余地あり
Alg_vec = ["BO", "RS"]
num_input_grid = 1 # ある一つの地点を制御
Opt_purpose = "MinSum" #MinSum, MinMax, MaxSum, MaxMinから選択

initial_design_numdata_vec = [1] #BOのRS回数
max_iter_vec = [2]            #{10, 20, 20, 50]=10, 30, 50, 100と同値
random_iter_vec = max_iter_vec

trial_num = 1  #箱ひげ図作成時の繰り返し回数
trial_base = 0

dpi = 75 # 画像の解像度　スクリーンのみなら75以上　印刷用なら300以上
colors6  = ['#4c72b0', '#f28e2b', '#55a868', '#c44e52'] # 論文用の色
###############################
jst = pytz.timezone('Asia/Tokyo')# 日本時間のタイムゾーンを設定
current_time = datetime.now(jst).strftime("%m-%d-%H-%M")
base_dir = f"result/BORS/{current_time}/"

cnt_vec = np.zeros(len(max_iter_vec))
for i in range(len(max_iter_vec)):
    if i == 0:
        cnt_vec[i] = int(max_iter_vec[i])
    else :
        cnt_vec[i] = int(cnt_vec[i-1] + max_iter_vec[i])
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
    control_input = [18, 7] # 
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
            # MOMY_dat = np.zeros((nt, nz, fny*ny, fnx*nx))
            # MOMY_no_dat = np.zeros((nt, nz, fny*ny, fnx*nx)) 
            # QHYD_dat = np.zeros((nt, nz, fny*ny, fnx*nx))
            # QHYD_no_dat = np.zeros((nt, nz, fny*ny, fnx*nx)) 
        # print(nc.variables.keys()) 
        dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
        odat[:, 0, gy1:gy2, gx1:gx2] = onc[varname][:]
        # MOMYの時.ncには'V'で格納される
        # MOMY_dat[:, :, gy1:gy2, gx1:gx2] = nc['V'][:]
        # MOMY_no_dat[:, :, gy1:gy2, gx1:gx2] = onc['V'][:]

        # QHYD_dat[:, :, gy1:gy2, gx1:gx2] = nc['QHYD'][:]
        # QHYD_no_dat[:, :, gy1:gy2, gx1:gx2] = onc['QHYD'][:]
    # 各時刻までの平均累積降水量をplot 
    # print(nc[varname].shape)
    # print(nc['V'].shape)
    # figure_time_lapse(control_input, base_dir, odat, dat, nt, varname)
    # figure_time_lapse(control_input, base_dir, MOMY_no_dat, MOMY_dat, nt, input_var)
    # figure_time_lapse(control_input, base_dir, QHYD_no_dat, QHYD_dat, nt, "QHYD")
    # merged_history の作成
    # subprocess.run(["mpirun", "-n", "2", "./sno", "sno_R20kmDX500m.conf"])
    # anim_exp(base_dir, control_input)

    sum_co=np.zeros(40) #制御後の累積降水量
    sum_no=np.zeros(40) #制御前の累積降水量
    for y_i in range(40):
        for t_j in range(nt):
            if t_j > 0:
                sum_co[y_i] += dat[t_j,0,y_i,0]*time_interval_sec
                sum_no[y_i] += odat[t_j,0,y_i,0]*time_interval_sec
    #print(sum_co-sum_no)
    return sum_co, sum_no

def black_box_function(control_input):
    """
    制御入力値列を入れると、制御結果となる目的関数値を返す
    """
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
        dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]

        sum_co=np.zeros(40) #制御後の累積降水量
        for y_i in range(40):
            for t_j in range(nt):
                if t_j > 0: #なぜかt_j=0に　-1*10^30くらいの小さな値が入っているため除外　
                    sum_co[y_i] += dat[t_j,0,y_i,0]*time_interval_sec
    objective_val = calculate_objective_func_val(sum_co, Opt_purpose)

    return objective_val


###実行
make_directory(base_dir)
 
filename = f"config.txt"
config_file_path = os.path.join(base_dir, filename)  # 修正ポイント
f = open(config_file_path, 'w')
##設定メモ##
f.write(f"input_var ={input_var}")
f.write(f"{input_size=}")
f.write(f"\nAlg_vec ={Alg_vec}")
f.write(f"\nnum_input_grid ={num_input_grid}")
f.write(f"\nOpt_purpose ={Opt_purpose}")
f.write(f"\ninitial_design_numdata_vec = {initial_design_numdata_vec}")
f.write(f"\nmax_iter_vec = {max_iter_vec}")
f.write(f"\nrandom_iter_vec = {random_iter_vec}")
f.write(f"\ntrial_num = {trial_num}")
f.write(f"\n{time_interval_sec=}")
################
f.close()

BO_ratio_matrix = np.zeros((len(max_iter_vec), trial_num)) # iterの組み合わせ, 試行回数
RS_ratio_matrix = np.zeros((len(max_iter_vec), trial_num))
BO_time_matrix = np.zeros((len(max_iter_vec), trial_num)) 
RS_time_matrix = np.zeros((len(max_iter_vec), trial_num))

BO_file = os.path.join(base_dir, "summary", f"{Alg_vec[0]}.txt")
RS_file = os.path.join(base_dir, "summary", f"{Alg_vec[1]}.txt")
progress_file = os.path.join(base_dir, "progress.txt")

# bounds に整数の範囲を指定する highまで探索範囲であることに注意
bounds = [Integer(low=0, high=39, prior='uniform', transform='normalize'),  # Y次元目: 0以上40未満の整数 (0～39)
          Integer(low=0, high=96, prior='uniform', transform='normalize')]  # Z次元目: 0以上97未満の整数 (0～96)

# # bounds に整数の範囲を指定する
# bounds = [Integer(low=0, high=39, prior='uniform', transform='normalize')]  # Y次元目: 0以上40未満の整数 (0～39)
with open(BO_file, 'w') as f_BO, open(RS_file, 'w') as f_RS,  open(progress_file, 'w') as f_progress:
    for trial_i in range(trial_num):
        f_progress.write(f"\n\n{trial_i=}\n")
        cnt_base = 0
        for exp_i in range(len(max_iter_vec)):
            f_progress.write(f"{exp_i=},  ")
            if exp_i > 0:
                cnt_base  = cnt_vec[exp_i - 1]

            ###BO
            random_reset(trial_i+trial_base)


            start = time.time()  # 現在時刻（処理開始前）を取得
            # ベイズ最適化の実行
            if exp_i == 0:
                result = gp_minimize(
                    func=black_box_function,        # 最小化する関数
                    dimensions=bounds,              # 探索するパラメータの範囲
                    acq_func="EI",
                    n_calls=max_iter_vec[exp_i],    # 最適化の反復回数
                    n_initial_points=initial_design_numdata_vec[exp_i],  # 初期探索点の数
                    verbose=True,                   # 最適化の進行状況を表示
                    initial_point_generator = "random",
                    random_state = trial_i
                )
            else:
                result = gp_minimize(
                    func=black_box_function,        # 最小化する関数
                    dimensions=bounds,              # 探索するパラメータの範囲
                    acq_func="EI",
                    n_calls=max_iter_vec[exp_i],    # 最適化の反復回数
                    n_initial_points=0,  # 初期探索点の数
                    verbose=True,                   # 最適化の進行状況を表示
                    initial_point_generator = "random",
                    random_state = trial_i,
                    x0=initial_x_iters,
                    y0=initial_y_iters
                )           
            end = time.time()  # 現在時刻（処理完了後）を取得
            time_diff = end - start
            # 最適解の取得
            min_value = result.fun
            min_input = result.x
            initial_x_iters = result.x_iters
            initial_y_iters = result.func_vals
            f_BO.write(f"\n input\n{result.x_iters}")
            f_BO.write(f"\n output\n {result.func_vals}")
            f_BO.write(f"\n最小値:{min_value}")
            f_BO.write(f"\n入力値:{min_input}")
            f_BO.write(f"\n経過時間:{time_diff}sec")
            f_BO.write(f"\nnum_evaluation of BBF = {cnt_vec[exp_i]}")
            sum_co, sum_no = sim(min_input)
            SUM_no = sum_no
            BO_ratio_matrix[exp_i, trial_i] = calculate_PREC_rate(sum_co, sum_no)
            print(BO_ratio_matrix[exp_i, trial_i])
            BO_time_matrix[exp_i, trial_i] = time_diff


            ###RS
            random_reset(trial_i+trial_base)
            # パラメータの設定
            # bounds_MOMY = [(-max_input, max_input)]*num_input_grid  # 探索範囲
            start = time.time()  # 現在時刻（処理開始前）を取得
            if exp_i == 0:
                best_params, best_score = random_search(black_box_function, bounds, random_iter_vec[exp_i], f_RS, num_input_grid)
            else:
                np.random.rand(int(cnt_base*num_input_grid)) #同じ乱数列の続きを利用したい
                best_params, best_score = random_search(black_box_function, bounds, random_iter_vec[exp_i], f_RS, num_input_grid, previous_best=(best_params, best_score))
            end = time.time()  # 現在時刻（処理完了後）を取得
            time_diff = end - start

            f_RS.write(f"\n最小値:{best_score}")
            f_RS.write(f"\n入力値:{best_params}")
            f_RS.write(f"\n経過時間:{time_diff}sec")
            f_RS.write(f"\nnum_evaluation of BBF = {cnt_vec[exp_i]}")
            sum_co, sum_no = sim(best_params)
            sum_RS_MOMY = sum_co
            RS_ratio_matrix[exp_i, trial_i] =  calculate_PREC_rate(sum_co, sum_no)
            RS_time_matrix[exp_i, trial_i] = time_diff

#シミュレーション結果の可視化
filename = f"summary.txt"
config_file_path = os.path.join(base_dir, "summary", filename)  
f = open(config_file_path, 'w')

vizualize_simulation(BO_ratio_matrix, RS_ratio_matrix, BO_time_matrix, RS_time_matrix, max_iter_vec,
         f, base_dir, dpi, Alg_vec, colors6, trial_num, cnt_vec)
f.close()
