import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import shutil,time, os, random
import ase.io,ase, subprocess,json
from dscribe.descriptors import SOAP, CoulombMatrix
from dscribe.kernels import AverageKernel, REMatchKernel #can be used to calculate this similarity
from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel, polynomial_kernel, pairwise_kernels
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
from itertools import cycle, islice
import GPy, GPyOpt
from GPyOpt.methods import BayesianOptimization
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_species(configlist):
    structures,symbols = [configlist],set()
    for s in range(len(structures)):
        for i in range(len(structures[s])):
            #print(structures[s][i])
            symbols.update(structures[s][i].get_chemical_symbols())
    return(symbols)

def hyper_kernal(soap, coulomb, Delta):
    K_new = (1-Delta)*soap + Delta*coulomb
    return K_new

def get_soap_coulomb_simlarity_matrix(traj_reader,r_cut,n_max,l_max,atom_sigma,soap_metric,coulomb_metric):
    # Build SOAP.
    print('Compute SOAP descriptors')
    soap_features = SOAP(species=get_species(traj_reader), rcut=r_cut, nmax=n_max, lmax=l_max, sigma=atom_sigma, periodic=True, rbf='polynomial',crossover=True, sparse=False).create(traj_reader)
    #Build Coulomb Matrix
    print('Compute Coulomb descriptors')
    used_atoms = 0
    for n in range(len(soap_features)):
        n_atoms = len(soap_features[n])
        if n_atoms > used_atoms:
            used_atoms = n_atoms
    coulomb_features = CoulombMatrix(n_atoms_max=used_atoms, flatten=False,permutation='none').create(traj_reader)
    #get the soap and coulomb simlarity Matrix
    sim_kernel_soap    = soap_metric.create(soap_features)
    sim_kernel_coulomb = coulomb_metric.create(coulomb_features)
    # Save the output to disk.
    scipy.sparse.save_npz("soap_simlarity_matrix.npz", scipy.sparse.csc_matrix(sim_kernel_soap))
    scipy.sparse.save_npz("coulomb_simlarity_matrix.npz", scipy.sparse.csc_matrix(sim_kernel_coulomb))
    return(sim_kernel_soap, sim_kernel_coulomb)

def get_clusters(distmat,simmat,min_cluster,tol):
    averg, cluster, evolution, itera  = [],[],[],[]
    min_dist, best_n_cluster          = 0.0, 0
    for n_clusters in range(min_cluster,len(simmat)):
        dis_clust           = []
        joint,number        = 0.0, 0
        output_clusters     = SpectralClustering(n_clusters=n_clusters, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0, affinity='precomputed', n_neighbors=10, eigen_tol=0.0, assign_labels='cluster_qr', degree=4, coef0=1.0, kernel_params=None, n_jobs=None, verbose=False).fit(simmat)
        cluster.append(n_clusters)
        #pick only label clusters
        label = output_clusters.labels_
        n_clusters_Spectral = len(set(label)) - (1 if -1 in label else 0)
        n_noise_Spectral    = list(label).count(-1)
        # group the clusters
        n_clustdict = {}
        for index, clustnum in enumerate(label):
            if clustnum != -1 :
                if clustnum not  in n_clustdict.keys():
                    n_clustdict[clustnum] = []
                n_clustdict[clustnum].append(index)
       # print('When we have {} clusters only'.format(len(n_clustdict)))
        one_all_diff, count = 0.0, 0
        for clust in range(len(n_clustdict)):
#           print('We have in cluster_{}, number of {} items'.format(clust,len(n_clustdict[clust])))
            for others in range(clust+1,len(n_clustdict)):
                diff_item = []
                for ind1 in range(len(n_clustdict[clust])):
                    diff = []
                    for ind2 in range(len(n_clustdict[others])):
                        diff.append(distmat[n_clustdict[clust][ind1]][n_clustdict[others][ind2]])
                    diff_item.append(np.mean(diff))
                diff_clust = (np.mean(diff_item))
                dis_clust.append(np.mean(diff_item))
                count +=1
        averg_diff = np.mean(dis_clust)
        #print('The average distance between all the {} clusters = {}'.format(count, averg_diff))
        averg.append(averg_diff)
#       print('average distance list = ',averg,'cluster list = ',cluster)
        if n_clusters > min_cluster:
            #print(averg[n_clusters-(min_cluster+1)],averg_diff)
            evolution.append(abs(averg[n_clusters-(min_cluster+1)]-averg_diff))
            itera.append(n_clusters)
            if abs(averg[n_clusters-(min_cluster+1)]-averg_diff) <= tol:
                min_dist = abs(averg[n_clusters-(min_cluster+1)]-averg_diff)
                best_n_cluster = n_clusters
                break;
            else:
                min_dist = abs(averg[n_clusters-(min_cluster+1)]-averg_diff)
                best_n_cluster = n_clusters
    return(label,n_noise_Spectral,n_clustdict,best_n_cluster)


def cluster_details(traj_reader, n_clustdict, weight):
    with open("cluster_details.dat", 'w') as d:
        for key in range(len(n_clustdict)):
            configlis = []
            for at in range(len(traj_reader)):
                if at in  n_clustdict[key]:
                    configlis.append(traj_reader[at])
                    d.write("{}\t {}\t {}\n".format(key, at, traj_reader[at].get_potential_energy()))
            os.makedirs('Clusters/cluster_num{}/data'.format(key), exist_ok = True)
            ase.io.write('Clusters/cluster_num{}/data/result.xyz'.format(key), configlis, 'extxyz')
            os.system("grep -o -P '(?<=config_type=).*(?=v)' ./Clusters/cluster_num{}/data/result.xyz >> Temp".format(key))
        d.write("{} {}\t {}\t        {}\n".format("#clust_id", "conf_id", "Energy", "Temp"))
    os.system("paste cluster_details.dat Temp > JOINT"+ "&& tail -1 JOINT >"+"Clusters_for_\u03B4_{}".format(weight)+" && head -n -1 JOINT >>"+"Clusters_for_\u03B4_{}".format(weight)+"&& rm -r Clusters Temp JOINT cluster_details.dat")
    end = "Clusters_for_\u03B4_{}" 
    return(end)

def Energy_plot(config_details_file,weight):
    colors = ['deeppink', 'blue', 'darkorange', 'green',  'gray', "purple","black","darkviolet",'brown','red', 'crimson']
    marks = ['o','P','s', 'D',  'H', 'X', 'v','p','*','h','>','<', '8']
    config = open(config_details_file.format(weight),'r').read().splitlines()
    cluster_config_id, config_config_id, ener_config, temp_config = [], [], [],[]
    start_config, mark_config, color_config = 0, 0, 0
    plt.figure(figsize=(15, 15))
    for line2 in range(1,len(config)):
        if  float(config[line2].split()[0]) == start_config:
            cluster_config_id.append(float(config[line2].split()[0])), config_config_id.append(float(config[line2].split()[1])), ener_config.append(float(config[line2].split()[2]))
            if line2 == (len(config)-1):
                plt.scatter(config_config_id,ener_config, marker=marks[mark_config], c= colors[color_config], s=40)
        else:
            plt.subplot(1, 1, 1)
            plt.scatter(config_config_id,ener_config, marker=marks[mark_config], c= colors[color_config], s=40)
            color_config+=1
            if color_config == len(colors):
                mark_config+=1
                color_config =0
            start_config = float(config[line2].split()[0])
            cluster_config_id, config_config_id, ener_config, temp_config = [], [], [],[]
    plt.xlim(0, len(config)-1)
    plt.xlabel('Snap_id',fontsize=20)
    plt.ylabel('E$^{DFT}$ (eV/atom)',fontsize=20)
    plt.title('\u03B4 = {}'.format(weight),fontsize=16, fontweight='bold')
    plt.savefig('Output_cluster_\u03B4_{}.eps'.format(weight), dpi=300, bbox_inches='tight')
    finish = "done"
    return(finish)

def Histogram_plot(n_clustdict2, weight):
    key_met2 = []
    for k2 in range(len(n_clustdict2)):
        for c2 in range(len(n_clustdict2[k2])):
            key_met2.append(k2)
    plt.figure(figsize=(18, 5))
    plt.subplot(1,1,1)
    nn, nnbins, npatches = plt.hist( key_met2, bins=np.linspace(np.min(key_met2),np.max(key_met2),len(n_clustdict2)+1), label= "\u03B4 = {}".format(weight))
    plt.xticks(np.arange(min(key_met2), max(key_met2)+2, 4.0))
    #plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
    plt.xlim(min(key_met2),max(key_met2))
    plt.ylim(0,)
    plt.xlabel('Cluster_id',fontsize=20)
    plt.ylabel('N_configs',fontsize=20)
    plt.legend(loc='best', shadow=True, fontsize=14)
    plt.savefig('Histogram_\u03B4_{}.eps'.format(weight), dpi=300, bbox_inches='tight')
    fi = "done"
    return(fi)

def MDS_2d_plot(similarity_matrix, weight, best_n_cluster, n_clustdict):
    embedding = MDS(n_components=2,random_state=best_n_cluster, normalized_stress='auto',dissimilarity='euclidean', eps=0.000001, n_jobs=-1, n_init=100)
    X_transformed = embedding.fit_transform(similarity_matrix)
    colors = ['deeppink', 'blue', 'darkorange', 'green',  'gray', "purple","black","darkviolet",'brown','red']
    marks = ['o','P','s', 'D',  'H', 'X', 'v','p','*','h','>','<', '8','^','1','3', '.']
    start1,  mark1, color1 = 0, 0, 0
    plt.figure(figsize=(16, 8))
    with open("MDS_2d_plot_{}.dat".format(weight), 'w') as DP :
            DP.write("{}\t {}\n".format("#x_axis", "y_axis"))
            for l in range(len(n_clustdict)):
                x_axis, y_axis = [], []
                if  l == start1:
                    for k, y in enumerate(X_transformed[:, 0]):
                        if k in n_clustdict[l]:
                            x_axis.append(y),  y_axis.append(X_transformed[:, 1][k])
                            DP.write("{}\t {}\n".format( y, X_transformed[:, 1][k]))
                    plt.scatter(x_axis,y_axis, marker=marks[mark1], c= colors[color1], s=40)
                    plt.xticks([]),           plt.yticks([])
                    plt.xticks(color='w'),    plt.yticks(color='w')
                    start1+=1
                    DP.write("\n")
                    color1+=1
                    if color1 == len(colors):
                         mark1  +=1
                         color1 = 0
    plt.title("{} clusters with \u03B4 = {}".format(best_n_cluster, weight), fontsize=18, fontweight='bold')
    plt.savefig('MDS_2d_plot_\u03B4_{}.eps'.format(weight), dpi=300, bbox_inches='tight')
    Done = "finish"
    return(Done)


def print_clusters(n_clustdict,traj_reader):
	direct      = 'Clusters/cluster_num{}/data'
	direct_file = 'Clusters/cluster_num{}'
	output      = 'Clusters/cluster_num{}/data/result.xyz'
	files       = ['options']
	for key, trjlist in n_clustdict.items():
    		configlis = []
    		for at in range(len(traj_reader)):
        		if at in  n_clustdict[key]:
            			configlis.append(traj_reader[at])
    		os.makedirs(direct.format(key), exist_ok = True)
    		ase.io.write(output.format(key), configlis, 'extxyz')
	finish = 'Done'
	return(finish)

def get_std(cluster_file_name, n_atoms, weight):
              clust_energies, std_clust, clust_id = [], [], 0
              clusters_file = open(cluster_file_name.format(weight),'r').read().splitlines()
              for line in range(1,len(clusters_file)):
                  if  float(clusters_file[line].split()[0]) == clust_id:
                      clust_energies.append((float(clusters_file[line].split()[2]))/n_atoms)
                      if line == (len(clusters_file)-1):
                         std = np.std(clust_energies)
                         std_clust.append(std*1000)
                  else:
                      std = np.std(clust_energies)
                      std_clust.append(std*1000)
                      #print(len(std_clust), len(clust_ener))
                      clust_id = float(clusters_file[line].split()[0])
                      clust_energies = []
              return(np.max(std_clust), np.average(std_clust), len(std_clust))
def Evaluate(x):
    rms_ener_train,rms_ener_test,rms_force_train,rms_force_test = gap_setup(cutoff = float(x[:,0]), delta = float(x[:,1]), n_sparse_2d = int(x[:,2]) ,n_sparse = int(x[:,3]), n_lmax = int(x[:,4]), n_nmax = int(x[:,5]), atom_sigma = float(x[:,6]))
    print("\nParam: cutoff = {},delta = {}, n_sparse_2d = {}, ,n_sparse = {}, n_lmax = {}, n_nmax = {}, atom_sigma = {}  |  RMS_ener_train : {}, RMS_ener_test: {}, RMS_force_train: {}, RMS_force_test: {}".format(float(x[:,0]),float(x[:,1]), int(x[:,2]), int(x[:,3]),float(x[:,4]),float(x[:,5]),float(x[:,6]) ,rms_ener_train,rms_ener_test,rms_force_train,rms_force_test))
    return(rms_ener_test)

def gap_setup(cutoff=5.0,delta=1.0,n_sparse_2d=50,n_sparse=100,n_lmax=4,n_nmax=4,atom_sigma=0.5):
    binn     ="Insert the whole path of the `quip` and `gap_fit` binary files"
    gap_file = " gp_file=test.xml > outputt"
    fit1     = binn+'/quip E=T F=T atoms_filename=train.xyz param_filename=test.xml | grep AT | sed "s/AT//" >> train_Gap.xyz'
    fit2     = binn+'/quip E=T F=T atoms_filename=valid.xyz param_filename=test.xml | grep AT | sed "s/AT//" >> valid_Gap.xyz'
    gap_fit  = binn+"/gap_fit at_file=train.xyz gap={distance_Nb order=2 compact_clusters=T cutoff="+str(round(cutoff,2))+" cutoff_transition_width=1 n_sparse="+str(n_sparse_2d)+" covariance_type=ard_se delta="+str(delta)+" theta_uniform=1.0 sparse_method=uniform add_species=T:soap l_max="+str(n_lmax)+" n_max="+str(n_nmax)+" cutoff="+str(round(cutoff,2))+" cutoff_transition_width=1.0 delta="+str(round(delta,2))+" atom_sigma="+str(round(atom_sigma,2))+" zeta=4 add_species=T config_type_n_sparse={same:"+str(n_sparse)+":Tellurium:1:Oxygen:1} sparse_method=cur_points covariance_type=dot_product} e0_method="+"isolated"+" sparse_jitter=1e-12 default_sigma={0.001 0.05 0.05 0.05} config_type_sigma={Tellurium:0.0001:0.05:0.05:0.05:Oxygen:0.0001:0.05:0.05:0.05}"+gap_file
    output_gap = subprocess.check_output(gap_fit,stderr=subprocess.STDOUT, shell=True)
    fit1_out   = subprocess.check_output(fit1,stderr=subprocess.STDOUT, shell=True)
    fit2_out   = subprocess.check_output(fit2,stderr=subprocess.STDOUT, shell=True)
    print(output_gap, fit1_out, fit2_out)
    while os.path.getsize("train_Gap.xyz") == 0:
        fit1_out   = subprocess.check_output(fit1,stderr=subprocess.STDOUT, shell=True)
        print(fit1_out)
        time.sleep(300)
    while os.path.getsize("valid_Gap.xyz") == 0:
        fit2_out   = subprocess.check_output(fit2,stderr=subprocess.STDOUT, shell=True)
        print(fit2_out)
        time.sleep(300)
    rms_ener_train, std_ener_train, rms_force_train, std_force_train   = formation_energy_force('./train.xyz','./train_Gap.xyz')
    rms_ener_test, std_ener_test, rms_force_test, std_force_test       = formation_energy_force('./valid.xyz','./valid_Gap.xyz')
    for f in range(1000):
        if not os.path.isfile('test_{}/valid.xyz'.format(f)):
            for b in range(1000):
                if not os.path.isfile('test_{}/BO_{}/valid_Gap.xyz'.format(f,b)):
                    os.makedirs('test_{}/BO_{}'.format(f,b), exist_ok = True)
                    os.system("mv train_Gap.xyz valid_Gap.xyz train.xyz.idx valid.xyz.idx  outputt log *.xml.* *.xml ./test_{}/BO_{}".format(f,b))
                    break;
            break;
    return(rms_ener_train,rms_ener_test,rms_force_train,rms_force_test)
    
def get_train_valid_set(n_clustdict, traj_reader,select,traj_reader_single):
    train_set, trainlist, valid_set, validlist = [],[],[],[]
    for key, clusterlist in n_clustdict.items():
        nused, nvalid = list(),list()
        n_used  = round((select/100)*len(clusterlist))
        n_valid = round((30/100)*n_used)
        nused   = random.sample(clusterlist,n_used)
        nvalid  = random.sample(nused,n_valid)
    #     print(ntrain,nvalid,len(clusterlist),length)
        for v in nvalid:
            valid_set.append(v)
        for t in nused:
            if t not in nvalid:
                train_set.append(t)
    #print('training set is with {} items'.format(len(train_set)))
    #print('validation set is with {} items'.format(len(valid_set)))
    for conf in range(len(traj_reader)):
            if conf in  train_set:
                trainlist.append(traj_reader[conf])
            elif conf in  valid_set:
                validlist.append(traj_reader[conf])
    for sing in range(len(traj_reader_single)):
        trainlist.append(traj_reader_single[sing])
        validlist.append(traj_reader_single[sing])
    ase.io.write("train.xyz", trainlist, 'extxyz')
    ase.io.write("valid.xyz", validlist, 'extxyz')
    return(trainlist,validlist)

def rms_dict(x_DFT, x_Gap):
    """ Takes two datasets of the same shape and returns a dictionary containing RMS error data"""

    x_ref = np.array(x_DFT)
    x_pred = np.array(x_Gap)

    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError('WARNING: not matching shapes in rms')

    error_2 = (x_ref - x_pred)** 2

    RMS = np.sqrt(np.average(error_2))
    std = np.sqrt(np.var(error_2))
    return (RMS, std)
    

def formation_energy_force(in_file, out_file):
        train_file, gap_file    = ase.io.read(in_file,':'), ase.io.read(out_file,':')
        file_in, file_out = open(in_file, "r").readlines(), open(out_file, "r").readlines()
        files         = [train_file,gap_file]
        species, atom, E_form_in, E_form_out = set(), [], [],[]
        force_in, force_out, skip = [], [], 0
        for c in range(len(files)):
                for s in range(len(files[c])):
                        species.update(files[c][s].get_chemical_symbols())
        for a in species: atom.append(a)
        for at in range(len(train_file)-len(atom)):
                symbols_in    = train_file[at].get_chemical_symbols()
                symbols_out   = gap_file[at].get_chemical_symbols()
                ener_in,ener_out   = 0.0, 0.0
                for o in range(len(train_file)-len(atom),len(train_file)):
                        n_sin_atom_in  = symbols_in.count(train_file[o].get_chemical_symbols()[0])
                        n_sin_atom_out = symbols_out.count(gap_file[o].get_chemical_symbols()[0])
#                         print(n_sin_atom_in,n_sin_atom_out,train_file[o].get_chemical_symbols()[0])
                        ener_in  = ener_in -(n_sin_atom_in* float(train_file[o].get_potential_energy()))
                        ener_out = ener_out - (n_sin_atom_out* float(gap_file[o].get_potential_energy()))
#                 print(ener_in, ener_out)
                formation_energy_in_atom       = (train_file[at].get_potential_energy()+ ener_in)/float(train_file[at].get_number_of_atoms())
                formation_energy_out_atom      = (gap_file[at].get_potential_energy()+ ener_out)/float(gap_file[at].get_number_of_atoms())
                E_form_in.append(formation_energy_in_atom), E_form_out.append(formation_energy_out_atom)
#                 print(formation_energy_out_atom)
                for j in range(skip+2, skip+(train_file[at].get_number_of_atoms())+2):
                        #atom = file_in[j].split()[0]
                        force_in.append([float(file_in[j].split()[4]), float(file_in[j].split()[5]), float(file_in[j].split()[6])])
                        force_out.append([float(file_out[j].split()[4]),float(file_out[j].split()[5]),float(file_out[j].split()[6])])
                skip +=((train_file[at].get_number_of_atoms())+2)
#                 print(at, (train_file[at].get_number_of_atoms()), skip)
#         print(len(force_in), len(force_out))
        ener_rms, ener_std = rms_dict(E_form_in, E_form_out)
        F_rms, F_std = rms_dict(force_in, force_out)
        return(ener_rms, ener_std, F_rms, F_std)
