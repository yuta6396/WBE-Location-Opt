time_interval_sec = 300 # 何秒おきに計算するか  .confのFILE_HISTORY_DEFAULT_TINTERVALと同値にする
bound = 30 #20240830現在ではMOMY=30, RHOT=10, QV=0.1にしている
#PSO LDWIM
w_max = 0.9
w_min = 0.4

# GA Parameters

gene_length = 3  # Number of genes per individual 制御入力grid数
crossover_rate = 0.8  # Crossover rate
mutation_rate = 0.05  # Mutation rate
lower_bound = -bound  # Lower bound of gene values
upper_bound = bound  # Upper bound of gene values
alpha = 0.5  # BLX-alpha parameter
tournament_size = 3 #選択数なので population以下にする必要あり