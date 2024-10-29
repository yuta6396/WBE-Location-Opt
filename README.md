# WBE-Location-Opt
## 概要
Warm Bubble Experimentを用いたシミュレーション. 目的関数:累積降水量(2024/10/19).

初期値介入による制御問題をBO, RS　もしくはPSO, GAの2手法をsimulateできる（ユーザーはsim_OO.pyを実行）.

基本実装：t=0の時の変数MOMYの変更量は固定　介入場所40*97グリッドの中から一つ選ぶ最適化
          組み合わせ最適化問題

## ファイル構造
- ~/scale-5.5.1/scale-rm/test/tutorial/ideal/WarmBubbleExperiment/WBE-AutomaticControlUnion
    - sim_BORS.py

    - sim_PSOGA.py

    - GS.py グリッドサーチ

    - optimize.py ブラックボックス最適化手法の実装

    - analysis.py シミュレーション後の処理

    - make_directory ディレクトリ階層構造を作成

    - config.py ブラックボックス最適化手法のハイパーパラメータ

    - calc_object_val.py 目的関数の計算　どんな目的関数にするか！

    - visualize_input.py 入力値の探索過程の可視化（sim_OO.pyには含まれない）

    - results/              # グラフや結果を保存
        - BORS
            - シミュレーション時間ごとのファイル
                - Accumulated-PREC-BarPlot
                - Line-Graph
                - Time-BarPlot
                - Time_lapse 各時間帯の累積降水量
                - BO_fig BO探索過程の可視化
                - summary
        - PSOGA

## 可能な実験設定
- 制御変数の変更
    - MOMY  (-30~30)
    - RHOT  (-10~10)
    - QV    (-0.1~0.1)程度


- 制御目的の変更
    - 観測できる全範囲（y=0~39）の累積降水量の（最小化/最大化）
    - 最大累積降水量地点（y=y'）の累積降水量（最小化/最大化）

## ブラックボックス最適化手法の実装方法
乱数種を10種類用意しシミュレーションを実行。
        np.random.seed(trial_i) 
        random.seed(trial_i) 
     

### ベイズ最適化
Scikit-Optimize ライブラリのgp minimize 関数

### 粒子群最適化
LDIWMを採用（w_max=0.9, w_min=0.4, c1=c2=2.0）

### 遺伝的アルゴリズム
実数値コーディングを採用

選択方式：トーナメント選択（トーナメントサイズ=3）

交叉方法：ブレンド交叉（交叉率=0.8，ブレンド係数α=0.5）

突然変異方法：ランダムな実数値への置き換え（突然変異率=0.05）

### ランダムサーチ
特になし


## 注意事項
- run_R20kmDX500m.confに#でコメントしようとするとエラーが起こる
- confファイルで&HISTORY_ITEM name='PREC', taverage=.true.としているので、PRECの値はTINTERVALの平均PRECで定義されます。TINTERVAL=3600なので、3600倍することで一時間の累積降水量が算出できます。
## 寄稿者

## 参照

## メモ
- 制御前とinput=0の制御後のPRECの値が異なるのはなんでか
- 12個あるはずの.mp4が6個しかない　BOかRSどっちかしか保存されていないor上書き
- test_result とresultの使い分け
