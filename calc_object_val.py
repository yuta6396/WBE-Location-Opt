import numpy as np

def calculate_objective_func_val(sum_co, Opt_purpose:str):
    """
    得られた各地点の累積降水量予測値(各Y-grid)から目的関数の値を導出する
    """
    represent_prec = 0
    if Opt_purpose == "MinSum" or Opt_purpose == "MaxSum":
        represent_prec = np.sum(sum_co)*100/96.50

    elif Opt_purpose == "MinMax" or Opt_purpose == "MaxMax":
        represent_prec = 0
        for j in range(40):  
            if sum_co[j] > represent_prec:
                represent_prec = sum_co[j] # 最大の累積降水量地点
        represent_prec= represent_prec*100/6.2
    else:
        raise ValueError(f"予期しないOpt_purposeの値: {Opt_purpose}")

    if Opt_purpose == "MaxSum" or Opt_purpose == "MaxMax":
        represent_prec = -represent_prec # 目的関数の最小化問題に統一   
    print(represent_prec)
    return represent_prec