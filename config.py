time_interval_sec = 300 # 何秒おきに計算するか  .confのFILE_HISTORY_DEFAULT_TINTERVALと同値にする
#PSO LDWIM
w_max = 0.9
w_min = 0.4

# GA Parameters

gene_length = 3  # Number of genes per individual 制御入力grid数
crossover_rate = 0.8  # Crossover rate
mutation_rate = 0.05  # Mutation rate
alpha = 0.5  # BLX-alpha parameter
tournament_size = 3 #選択数なので population以下にする必要あり