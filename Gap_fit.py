########################################################################################################################################                                                    
# This is the Multi-kernel Clustering of Amorphous systems and Machine-Learned Interatomic Potentials by active learning
#                                                            MK-CAMLIP
#
#
# This code has been written and is copyright (c) 2021-2024 of the following authors:
# Firas Shuaib, Guido Ori, Philippe Thomas, Olivier Masson, and Assil Bouzid. 
#
# Refer to the file LICENSE.md for license information and the README.md file for practical installation 
#
# MK-CAMLIP  workflow has two main files 
#     1. Gap_fit.py:   Takes all input parameters and implements functions for computing The global similarity between configurations, Active learning clustering, select the best value of Delta, bulid training and test data samples, fitting the GAP model until the desired level of accuracy is attained, ... etc. 
#     2. Functions.py: defined all the funcions that used in the main python file "Gap_fit.py"

#
#                                             The final code repository is  
#                                         https://github.com/ASM2C-group/MK-CAMLIP
# A python workflow to run active learning for Clustering and GAP is implemented through the script *Gap_fit.py*.  The script's input parameters are defined below.

#                                                  Main arguments:
#
# filename			        Trajectory filename (extxyz formate)
# single_atom_file		        Single atom trajectory  filename (extxyz formate)
# r_cut, n_max, l_max, atom_sigma       Cutt off radious,Nmax, Lmax, and atom_sigma values to build SOAP descriptor for the aim to have local similarity measure between two configurations.
# Delta 				Hyperparameter  controls the relative weight of SOAP and Coulomb  kernels. At the aim of tunning Delta (AL-Clustering step) It shoud be defined as "None" lead to repeating the Clustering procedure for different value of Delta varied from 0 to 1 with a step of 0.1. Once the Clustering procedure has done, the best value of Delta will be selected. Delta value can also defined by the user.     
# d_tol 	                        Distance convergence threshold for active learning clustering step.
# min_cluster			        Number of minimum clusters to be obtained.
# I_perc				Starting fraction of configurations from each uncorrelated cluster.
# n                                     Increment fraction of configurations from each uncorrelated cluster
# cutoff				Cutt off range for SOAP Usage: [min max] (for GAP fitt task)
# sparse_2d			        Nsparse range for 2D descriptor Usage: [min max] (for GAP fitt task)
# sparse				Nsparse range for SOAP Usage: [min max] (for GAP fitt task)
# lmax, nmax			        Nmax, Lmax range. Usage:  [min max] (for GAP fitt task)
# sigma_atom			        Range of hyperparameter sigma_atom for SOAP Usage: [min max] (for GAP fitt task)
# Nopt				        Number of exploration and optimization steps for BO Usage: [exploration optimization] (used [30 20])
# rmse_target			        Root mean square error (RMSE), energy convergence threshold  (eg. 0.002)
# restart     			        If "yes" the active loop will start from the last loop.        
#######################################################################################################################################

import numpy as np
from Functions import get_species,get_soap_coulomb_simlarity_matrix, hyper_kernal,get_clusters,get_train_valid_set,print_clusters, Evaluate,formation_energy_force,Energy_plot,Histogram_plot, cluster_details, MDS_2d_plot, get_std
import scipy.sparse
import matplotlib.pyplot as plt
import os , subprocess,time
import ase.io, ase, random, json, shutil
from dscribe.descriptors import SOAP, CoulombMatrix
from dscribe.kernels import AverageKernel,REMatchKernel  #can be used to calculate this similarity
from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel, polynomial_kernel
from sklearn.cluster import SpectralClustering
from itertools import cycle, islice
import GPy, GPyOpt
from GPyOpt.methods import BayesianOptimization
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part I: Define your System input parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define input files name for your dataset
filename, single_atom_file         = 'DB_AsTe3_140Ry.xyz', 'one.xyz'  

#Define input parameters for your SOAP descriptor for the aim of active learning clustering step 
r_cut, n_max, l_max, atom_sigma = 5.5, 8, 8, 0.5

#Define your Similarity Kernels to measure the simlarity between config
metric_soap = AverageKernel(metric="polynomial",gamma=None, degree=4, coef0=1.0, normalize_kernel=True)    # For local (SOAP) similarity measurement
metric_coulomb = AverageKernel(metric="laplacian",gamma=None, degree=4, coef0=1.0, normalize_kernel=True)  # For long-range (Coulomb) similarity measurement

# Define the best value of 𝛿. In the active learning clustering step the user must set 𝛿 = None.
Delta       = None

#Define the desired threshold for the change in the average distance between obtained clusters
d_tol       = 0.0001

#Define the minimum number of clusters that must be obtained
min_cluster = 2

#Define the starting and Increment fractions (I_perc and n respectively) of configurations from each uncorrelated cluster.
I_perc      = 30
n           = 10

# Define the Main arguments for the Gap-fit procedure
cutoff      = [4, 5.5]
sparse_2d   = [10, 100]
sparse      = [1500, 2500]
lmax        = [5,8]
nmax        = [5,8]
sigma_atom  = [0.1, 1.0]

# Define the number of exploration and optimization steps for the Bayesian Optimization step
Nopt        = [30, 20]

# Define the desired RMSE for the energy on the test set
rmse_target = 0.002

# If yes means all snapshots in a given dataset have the same number of atoms.
All_config_same = "yes"

# If "yes" All_config_same = "yes",  the active learning MLIP process will start using the best value of Delta found from the active learning clustering step, otherwise the workflow will stop after the clustering step. 
Do_MLIP = "no" 

# If "yes" the active learning MLIP process will start from the last step. 
restart     = "yes"

traj_reader, traj_reader_single        = ase.io.read(filename,':'), ase.io.read(single_atom_file,':')
print("Number of snapshots in the input file = {}\n".format(len(traj_reader)),"Number of snapshots in the single atom file = {}\n ".format(len(traj_reader_single)))




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Part II Load/build SOAP, Coulomb, hybrid, and distance simlarity matrices between configurations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
    Here we build a local (kernel_soap) and long-range (kernel_coulomb) similarity matrices between various configurations in a given dataset (an extxyz file), based on the average kernel method. Then these matrices will be saved in ".npz" binary format, in the current directory. This will reduce the storage memory needed to run the workflow as well as enable these binary files to be called later in order to construct a global (hybrid) similarity kernel. 
    
    We propose a hybrid kernel approach that base on a parameter 𝛿, which combines both the SOAP and Coulomb average kernels in a way to take advantage of the local and long-range descriptions of a target system. The hybrid kernel can be defined as follows: 
                                                           hyprid_kernal(𝐴, 𝐵) = (1 − 𝛿) × hyprid_kernal(𝐴, 𝐵) + 𝛿 × kernel_coulomb(𝐴, 𝐵)
   
    Then this kernel is converted into a distance metric as:
                                                          distmat = √2 − 2*hyprid_kernal(𝐴, 𝐵)
    
     If the user has defined the optimum value of Delta, the corresponding hybrid and distance matrices will be calculated and saved in a binary format ".npz", otherwise only soap and Coulomb matrices will be obtained and saved. Once one of the hybrid, distance, kernel_soap, and kernel_coulomb  matrices is exists in the the current directory, the code will call the corresponding binary file and continue computing the rest matrices.
"""

if not os.path.isfile("distance_matrix_{}.npz".format(Delta)):
    if os.path.isfile('coulomb_simlarity_matrix.npz'):
        #Load the Soap and Coulomb simlarity Matrix.
        kernel_soap = scipy.sparse.load_npz("soap_simlarity_matrix.npz").toarray()
        kernel_coulomb   =  scipy.sparse.load_npz("coulomb_simlarity_matrix.npz").toarray()
        print(" Loading Soap and Coulomb simlarity Matrix is done")
    else:
        print("Start building Soap,coulomb descriptors and their similarity matrix")
        kernel_soap, kernel_coulomb = get_soap_coulomb_simlarity_matrix(traj_reader,r_cut,n_max,l_max,atom_sigma,metric_soap,metric_coulomb)
        for y in range(len(kernel_soap)):
            for p in range(len(kernel_soap[y])):
                if kernel_soap[y][p] > 1.00 :
                    kernel_soap[y][p] = 1.00  # To avoid any numerical error
                elif kernel_coulomb[y][p] > 1.00 :
                    kernel_coulomb[y][p] = 1.00
    # Building hyper simlarity and distance matrix
    if not Delta == None:
        distmat, simmat    = np.empty((len(kernel_soap), len(kernel_soap))),np.empty((len(kernel_soap), len(kernel_soap)))
        for config in range(len(kernel_soap)):
            hyprid_kernal  = hyper_kernal(kernel_soap[config], kernel_coulomb[config], Delta)
            #print(len(hyprid_kernal))
            for w in range(len(hyprid_kernal)):
                if hyprid_kernal[w] > 1.00 :
                    hyprid_kernal[w] = 1.00 
                    #print(hyprid_kernal[w])
            simmat[config] = hyprid_kernal
            d = np.sqrt(2-(2*hyprid_kernal))
            distmat[config]= d
        print(" Building hyper simlarity and distance matrix is done")
        #Save the data in a binary file
        scipy.sparse.save_npz("hyprid_matrix_{}.npz".format(Delta), scipy.sparse.csc_matrix(simmat))
        scipy.sparse.save_npz("distance_matrix_{}.npz".format(Delta), scipy.sparse.csc_matrix(distmat))
elif os.path.isfile("distance_matrix_{}.npz".format(Delta)):
    #Load the hyper Matrix.
    simmat  = scipy.sparse.load_npz("hyprid_matrix_{}.npz".format(Delta)).toarray()
    distmat = scipy.sparse.load_npz("distance_matrix_{}.npz".format(Delta)).toarray()
    print(" Loading hyper simlarity and distance matrix is done")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part III: Active learning clustering step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
    In part III, if the user did not define the optimum value of Delta (set as Delta = None), the workflow will start the Active learning clustering step by using the local (kernel_soap) and long-range (kernel_coulomb) similarity matrices computed from part II to build the hybrid kernel: 
                                                           hyprid_kernal(𝐴, 𝐵) = (1 − 𝛿) × hyprid_kernal(𝐴, 𝐵) + 𝛿 × kernel_coulomb(𝐴, 𝐵)
    
    Here the value of 𝛿 is varied from 0 to 1 with a step of 0.1. For each value of 𝛿 the hybrid kernel is converted into a distance metric as:
                                                          distmat = √2 − 2*hyprid_kernal(𝐴, 𝐵)
    
    The distance matrix is provided as input to the Spectral Clustering algorithm iteratively until the change in the average distance between obtained clusters is attained the desired threshold (d_tol). This Clustering procedure is repeated for different value of 𝛿. For each value of delta an output file is printed. This file contain information such as clust_id, Energy, and Temp of each config_id. Furthermore, the obtained list of uncorrelated clusters are visualized in a 2D map by resorting to the Multidimensional Scaling (MDS) method. The workflow will print (.dat, and .eps) file, which show the 2D map corresponding to each value of 𝛿. 
        
    Please note that both hybrid and distance matrices wil not be saved for each delta value of.
    
    If the user set Do_MLIP == "yes" and All_config_same == "yes", the workflow will caculate the optimum value of Delta. Next it will compute the corresponding hybrid and distance matrices. sequentially, The active learning MLIP step will start basd on the computed matrices.
    
    else
        The workflow will stop and only  Active learning clustering step will be achieved. The user has to defined the the optimum value of Delta and then re-execute the workflow.
"""

while Delta == None:
    weightt, w_dist_cluster, Aver_std, Max_std, Num_stds = [], [], [], [], []
    for w in range(0,11,1):
        Delta = w/10
        weightt.append(Delta)
        for y in range(len(kernel_soap)):
            for p in range(len(kernel_soap[y])):
                if kernel_soap[y][p] > 1.00 :
                    kernel_soap[y][p] = 1.00  # To avoid any numerical error
                elif kernel_coulomb[y][p] > 1.00 :
                    kernel_coulomb[y][p] = 1.00
        distmat, simmat    = np.empty((len(kernel_soap), len(kernel_soap))), np.empty((len(kernel_soap), len(kernel_soap)))
        for config in range(len(kernel_soap)):
            hyprid_kernal  = hyper_kernal(kernel_soap[config], kernel_coulomb[config], Delta)
            for w in range(len(hyprid_kernal)):
                if hyprid_kernal[w] > 1.00 :
                    print("have hyper similarity highter than one in config {}".format(w),hyprid_kernal[w])
            simmat[config] = hyprid_kernal
            d = np.sqrt(2-(2*hyprid_kernal))
            distmat[config]= d
        label_config_distance,n_noise_config_distance,n_clustdict_config_distance,best_n_cluster_config_distance = get_clusters(distmat,simmat,min_cluster,d_tol)
        result_config_distance = cluster_details(traj_reader,n_clustdict_config_distance,Delta)
        energy = Energy_plot(result_config_distance, Delta)
        hist_clusters = Histogram_plot(n_clustdict_config_distance, Delta)
        w_dist_cluster.append(best_n_cluster_config_distance)
        aver_std_delta, max_std_delta, n_std_delta = get_std(result_config_distance, traj_reader[1].get_number_of_atoms(), Delta) # This works only if all snapshots has the same number of atoms
        Aver_std.append(aver_std_delta), Max_std.append(max_std_delta), Num_stds.append(n_std_delta)
        get_2d_plot        = MDS_2d_plot(simmat, Delta, best_n_cluster_config_distance, n_clustdict_config_distance )
    with open('./n_cluster_std_VS_\u03B4.dat', 'w') as nst:
                        nst.write("{} \t {}\t {}\t  {}\n".format("#Delta", "max_std", "average_std", "n_clusters"))
                        for data in range(len(weightt)):
                            nst.write("{} \t {}\t {}\t  {}\n".format(weightt[data], Max_std[data], Aver_std[data], Num_stds[data]))
    if All_config_same == "yes":
       Best_delta = weightt[np.argmin(Aver_std)]
       Delta = Best_delta
       #print(Best_delta)
       distmat, simmat    = np.empty((len(kernel_soap), len(kernel_soap))),np.empty((len(kernel_soap), len(kernel_soap)))
       for config in range(len(kernel_soap)):
            hyprid_kernal  = hyper_kernal(kernel_soap[config], kernel_coulomb[config], Best_delta)
            #print(len(hyprid_kernal))
            for w in range(len(hyprid_kernal)):
                if hyprid_kernal[w] > 1.00 :
                    hyprid_kernal[w] = 1.00 
                    #print(hyprid_kernal[w])
            simmat[config] = hyprid_kernal
            d = np.sqrt(2-(2*hyprid_kernal))
            distmat[config]= d
       print(" Building hyper simlarity and distance matrix is done")
       #Save the data in a binary file
       scipy.sparse.save_npz("hyprid_matrix_{}.npz".format(Delta), scipy.sparse.csc_matrix(simmat))
       scipy.sparse.save_npz("distance_matrix_{}.npz".format(Delta), scipy.sparse.csc_matrix(distmat))
    plt.figure(figsize=(15, 5))
    plt.subplot(1,1,1)
    plt.plot(weightt,w_dist_cluster, label= "config_distance")
    plt.xticks(np.arange(0, 1.2, 0.1))
    plt.xlim(0, 1.0)
    plt.ylim(0,)
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
    plt.xlabel('Delta')
    plt.ylabel('n_cluster')
    plt.legend(loc='best', shadow=True, fontsize=9)
    #plt.title("Config distance output",fontsize=16, fontweight='bold')
    plt.savefig('n_cluster_VS_\u03B4.eps', dpi=300, bbox_inches='tight')
    if Do_MLIP == "yes" and All_config_same == "yes":
    	print("The active learning clustering step has finished successfully. The best value of \u03B4 found to be {}. Now, Active learning clustering step.Now the ctive learning MLIP step will start. If you would like to change the value of Delta, please stop the run and define the value of Delta manually. Enjoy !!!!" .format(Best_delta))
    elif Do_MLIP != "yes" and All_config_same != "yes":
         print("The active learning clustering step has finished successfully.")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part IV The active learning MLIP (AL-MLIP) step  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
     In Part IV  we build the training and test data samples are built from the the selected 𝛿 clusters. 
              1/ We extract from each uncorrelated cluster a fraction I_perc (defined by the user) of configurations.
              
              2/ The small database is split into a training set (70% of the data) and a validation set (30% of the data) that will serve as a measure of the accuracy of the achieved MLIP through the calculation of the Root Mean Square Error (RMSE) on the predicted energies compared to the reference FPMD data.   
              
              3/ The training set is used to train the GAP model. Bayesian optimization (BO) is uesd to tune 7 hyperparameters of the GAP model, including: cutoff, gap-delta, 2d-number of sparse points, SOAP number of sparse, l_max, n_max, and  atom_sigma.  the number of BO iterations it depends on the number of exploration and optimization steps defind by the user, the obtained MLIP at each BO-iteration can found in a directory called test with a tail corresponding to the active learning MLIP loop as test_{AL-loop}. 
              
              4/ If the GAP model refined using BO fails to meet the desired level of accuracy (rmse_target), the training set will be increased by a fraction of n (defind by the user), and another loop of AL-MLIP is carried out.
              
              5/ Step 4 will be repeated, untill reaches the pre-defined RMSE in energy prediction for GAP model or when the 100% of the reference dataset is used.
              
              6/ Upon convergence, the obtained training, and test database itogether with the optimal GAP hyperparameters (report file) that are required to achieve the target accurcy of GAP MLIP potential will saved in test_{last_AL-loop}.
"""

if Do_MLIP == "yes" and All_config_same == "yes":
   print("The active learning MLIP step has started with \u03B4 value = {}. This value is either obtained from the active learning clustering step or defined by the user." .format(Delta))
   target      = "RMSE_E_testset"
   label,n_noise,n_clustdict,best_n_cluster = get_clusters(distmat,simmat,min_cluster,d_tol)
   see_clusters = print_clusters(n_clustdict,traj_reader)
   iteration, RMS = [],[]
   opt_rmse_best = 100.0
   for ff in range(100):
        if os.path.isfile('test_{}/train.xyz'.format(ff)):
            print("The {} trial has finished successfully. Find the report file for the RMSE_E on the testset in the folder test_{}\n".format(ff, ff))
            iteration.append(ff)
            BO_iteration, BO_rms_En_t, BO_rms_En_v, BO_rms_F_t, BO_rms_F_v = [], [], [], [], []
            for b in range(Nopt[0]+Nopt[1]):
                if os.path.isfile('test_{}/BO_{}/train_Gap.xyz'.format(ff,b)):
                    BO_iteration.append(b)
                    rms_ener_train, std_ener_train, rms_force_train, std_force_train = formation_energy_force('test_{}/train.xyz'.format(ff),'test_{}/BO_{}/train_Gap.xyz'.format(ff,b))
                    rms_ener_valid, std_ener_valid, rms_force_valid, std_force_valid = formation_energy_force('test_{}/valid.xyz'.format(ff),'test_{}/BO_{}/valid_Gap.xyz'.format(ff,b))
                    BO_rms_En_t.append(rms_ener_train), BO_rms_En_v.append(rms_ener_valid), BO_rms_F_t.append(rms_force_train), BO_rms_F_v.append(rms_force_valid)
            print(BO_iteration, BO_rms_En_t, BO_rms_En_v, BO_rms_F_t, BO_rms_F_v, ff)
            os.system("mv RMS_build_train_n_* ./test_{}".format(ff))
        elif (I_perc + n*(ff)) > 100:
            print('All the dataset is used in the previous iteration. The predefined RMSE is not reached')
            break;
        elif not os.path.isfile('test_{}/train.xyz'.format(ff)) and restart == "yes":    # To restart the BO from the last step
            os.makedirs('test_{}'.format(ff), exist_ok = True)
            trainset,validset = get_train_valid_set(n_clustdict, traj_reader,I_perc+(n*(ff)),traj_reader_single)
            print("In {} iteration the number of configs in the training and test set are: {} , {}".format(ff,len(trainset), len(validset) ) )
            bounds = [{'name': 'cutoff',          'type': 'continuous',  'domain': (cutoff[0], cutoff[1])},\
                {'name':'delta',            'type': 'discrete',  'domain':np.arange(0.01,1.0,0.01)},\
                {'name': 'n_sparse_2d',        'type': 'discrete',  'domain': np.arange(sparse_2d[0],sparse_2d[1]+1,10)},\
                {'name': 'n_sparse',        'type': 'discrete',  'domain': np.arange(sparse[0],sparse[1]+1,100)},\
                {'name':'n_lmax',           'type': 'discrete',  'domain': np.arange(lmax[0],lmax[1]+1)},\
                {'name':'n_nmax',           'type': 'discrete',  'domain': np.arange(nmax[0],nmax[1]+1)},\
                {'name':'atom_sigma',       'type': 'discrete',  'domain': np.arange(sigma_atom[0],sigma_atom[1],0.1)},]
            # Optimize the quip parameters
            opt_gap = BayesianOptimization(f=Evaluate, domain=bounds,initial_design_numdata = int(Nopt[0]),
                                             model_type="GP_MCMC",
                                             acquisition_type='EI_MCMC', #EI
                                             evaluator_type="predictive",  # Expected Improvement
                                             exact_feval = False,
                                             model_optimize_restarts = True,
                                             verbosity=1,
                                             maximize=False)  
            print("Building the initial points for Bayesian Optimization has started")
            #Run Bayesian Optimization
            print(" \nBayesian Optimization has started \n")
            opt_gap.run_optimization(max_iter=int(Nopt[1]),verbosity=2,report_file='repot.dat',evaluations_file='evalution.dat', models_file='modle.dat')
            opt_gap.plot_convergence("convergence.png")
            hyper_parameters = {}
            for parameter in range(len(bounds)):
                    hyper_parameters[bounds[parameter]["name"] ] = opt_gap.x_opt[parameter]
            hyper_parameters[target] = opt_gap.fx_opt
            RMS.append(opt_gap.fx_opt)
            os.system("mv train.xyz valid.xyz ./test_{}".format(ff)+"&& cp *.dat *.png  ./test_{}".format(ff))
            print("The obtain hyper parameters from Bayesian Optimization for {} AL-loop are {}\n ".format(ff, hyper_parameters))
            opt_rmse_best = opt_gap.fx_opt
            print("\n Target RMSE lowered in {} trial: {} eV/Atom".format(ff,opt_rmse_best))
            best_hyper = hyper_parameters
            if ff != 0  and opt_rmse_best  < rmse_target:
                    plot_final(iteration,RMS)
                    print("\n the smallest possible training database can be found in the folder test_{} with hyper parameters : {}\n Bye.".format(ff, best_hyper))
                    break;
