from TICC_solver import TICC
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import sys
import matplotlib.pyplot as plt
# fname = "example_data.txt"

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
#
# Data = np.loadtxt(fname, delimiter=",")
# P = np.loadtxt("Results22.txt")
# start = 102
# mark = 134
# (m, n) = Data.shape  # m: num of observations, n: size of observation vector
# DataP = np.array(P)
# print(DataP)
# Data=np.array(Data)
# Data_pro2K=Data[start:mark,:]
# print(Data_pro2K.shape)
# print(P.shape)
#
# index=range(start+1,mark+1)
# for i in range(0,3):
#  plt.plot(index,Data_pro2K[:,i])
# plt.plot(index,P[start:mark])
# plt.show()


fname = 'E:\exam\linfentest\extract_date\\013_2.txt'
num_clu = 10
ticc = TICC(window_size=3, number_of_clusters=num_clu, lambda_parameter=11e-2, beta=1, maxIters=60, threshold=2e-5,
            write_out_file=False, prefix_string="output_folder/", num_proc=1,compute_BIC=True)
(cluster_assignment, cluster_MRFs,bic,lle_all_points_clusters) = ticc.fit(input_file=fname)


np.savetxt('F:\TICC\TICC-master\output_res\Results71.txt', cluster_assignment, fmt='%d', delimiter=',')


mrf_value = np.array(list(cluster_MRFs.values()))
cluster_inner_distance = np.arange(float(num_clu))#
for i in range(num_clu):
    cluster_temp = lle_all_points_clusters[np.where(cluster_assignment==i)]
    clusteri = cluster_temp[:, i].mean()
    cluster_inner_distance[i] = clusteri
cluster_inner_distance = normalization(cluster_inner_distance)
cluster_dis = np.zeros((num_clu,num_clu))
Rij = np.zeros((num_clu,num_clu))
for i in range(num_clu):
    for j in range(num_clu):
        cluster_dis[i,j] = np.linalg.norm(cluster_MRFs[i]-cluster_MRFs[j])
print(cluster_dis)

for i in range(num_clu):
    for j in range(num_clu):
     Si = cluster_inner_distance[i]
     Sj = cluster_inner_distance[j]
     if cluster_dis[i, j] == 0:
        Rij[i, j] = 0
        continue
     Rij[i, j] = (Si + Sj) / cluster_dis[i, j]

DBI = np.max(Rij, axis=1).sum()/num_clu
print('--------DBI-----')
print(DBI)


print('------BIC-------')
print(bic)


