import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
import scipy
import freud
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_1samp
from scipy.stats import permutation_test
import seaborn as sns
import matplotlib.style as style

def calc_dist_mat(positions):
    """
    Compute a distance matrix for a given array of positions
    """
    return np.linalg.norm(np.abs(np.reshape(positions,(positions.shape[0],1,positions.shape[1]))-np.reshape(positions,(1,positions.shape[0],positions.shape[1]))),axis=2)

def gyration_tensor(positions,box_dimensions):
    """
    Compute the gyration tensor for a set of particles
    """
    #move center of mass of the particles to zero
    positions = center(positions,box_dimensions)
    return np.einsum('im,in->mn', positions,positions)/len(positions)

def center(positions, box_dimensions):
    """
    Center a list of positions at (0,0,0) taking into account periodic
        boundary conditions
    Inputs:
        positions: the positions to center
        box_dimensions: the dimensions of the periodic box
    """
    angular_positions = positions / box_dimensions * 2 * np.pi
    sines = np.sin(angular_positions)
    cosines = np.cos(angular_positions)
    angle_mean = np.arctan2(-np.mean(sines,axis=0),-np.mean(cosines,axis=0)) + np.pi
    com = box_dimensions * angle_mean / (2*np.pi)
    positions = positions - com
    return np.where(np.abs(positions)<box_dimensions/2,positions,positions-np.sign(positions)*box_dimensions)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
    
def compute_vec_periodic(point1,point2, box_dimensions):
    """
    Utility function: Compute the vector between 2 points keeping in mind a periodic box of dimensions box_dimensions
    Returns the vector between the two points in question
    """
    distance = point2-point1
    abs_dist = np.abs(distance)
    alt_dist = (box_dimensions-abs_dist)*np.sign(distance)*-1
#     print(distance)
#     print(box_dimensions-np.abs(distance))
#     print(np.where(abs_dist <= box_dimensions-abs_dist, distance, alt_dist))
#     print('\n\n')
    return np.where(abs_dist <= box_dimensions-abs_dist, distance, alt_dist)

def compute_cog_periodic(points, box_dimensions):
    """
    Utility function: Compute the center of gravity of a set of 2 points keeping in mind a periodic box of dimensions box_dimensions
    Returns the com of the points in question
    """
    distance = points[1]-points[0]
    if distance[0] > box_dimensions[0] / 2:
        points[1,0] = points[1,0]-box_dimensions[0]
    elif distance[0] < -box_dimensions[0] / 2:
        points[0,0] = points[0,0]-box_dimensions[0]
    if distance[1] > box_dimensions[1] / 2:
        points[1,1] = points[1,1]-box_dimensions[1]
    elif distance[1] < -box_dimensions[1] / 2:
        points[0,1] = points[0,1]-box_dimensions[1]
    if distance[2] > box_dimensions[2] / 2:
        points[1,2] = points[1,2]-box_dimensions[2]
    elif distance[2] < -box_dimensions[2] / 2:
        points[0,2] = points[0,2]-box_dimensions[2]    
    return np.mean(points,axis=0)

def cos_diff_mat(points):
    """
    Utility function: calculates a cosine difference matrix
    Inputs:
        points - the points to calculate the cosine difference of
    """
    norm = np.linalg.norm(points,axis=1)
    return  np.around(1 - np.dot(points, points.T) / np.outer(norm,norm),4)

def calc_dist_mat_periodic(positions, box_dimensions):
    """
    Utility function: compute a distance matrix for a given array of positions in a 3D periodic box
    Inputs:
        positions: the positions for which to compute the distance matrix
        box_dimensions: the dimensions of the periodic box
    """
    vecdist=np.abs(np.reshape(positions,(positions.shape[0],1,positions.shape[1]))-np.reshape(positions,(1,positions.shape[0],positions.shape[1])))
    return np.linalg.norm(np.minimum(vecdist,box_dimensions-vecdist),axis=2)

def calc_dist_mat_periodic_2pos(positions1, positions2, box_dimensions):
    """
    Utility function: compute a distance matrix for 2 given arrays of positions in a 3D periodic box
    Inputs:
        positions: the positions for which to compute the distance matrix
        box_dimensions: the dimensions of the periodic box
    """
    vecdist=np.abs(np.reshape(positions1,(positions1.shape[0],1,positions1.shape[1]))-np.reshape(positions2,(1,positions2.shape[0],positions2.shape[1])))
    return np.linalg.norm(np.minimum(vecdist,box_dimensions-vecdist),axis=2)

def plane_least_squares(x,y):
    """
    Takes a 2-dimensional input array of points (x,y) and a target array f(x,y)
    Calculates a planar fit of the form f = a + bx + cy
    Returns the least sum of squares
    Inputs:
        x: array with shape (N,2)
        y: array with shape (N,)
    Outputs: least sum of squares
    """
    #make the array of polynomial features with order 2
    a1=np.ones((len(x),1))
    a2=x
    A=np.concatenate((a1,a2),axis=1)
    sol=scipy.linalg.lstsq(A,y)
    return np.around(sol[1],4)/len(x)

def compute_gyration_tensor(x):
    """
    Compute the gyration tensor for an input array of points x
    Inputs:
        x: nparray which contains a list of points
    Returns:
        gyration tensor G as a 3x3 array
    """
    #move coordinates to center of mass
    demeaned_coords = x - compute_com(x)
    
    #calculate gyration tensor
    g=np.zeros((3,3))
    for i in range(len(demeaned_coords)):
        g[0,0] += demeaned_coords[i,0]*demeaned_coords[i,0]
        g[0,1] += demeaned_coords[i,0]*demeaned_coords[i,1]
        g[0,2] += demeaned_coords[i,0]*demeaned_coords[i,2]
        g[1,0] += demeaned_coords[i,1]*demeaned_coords[i,0]
        g[1,1] += demeaned_coords[i,1]*demeaned_coords[i,1]
        g[1,2] += demeaned_coords[i,1]*demeaned_coords[i,2]
        g[2,0] += demeaned_coords[i,2]*demeaned_coords[i,0]
        g[2,1] += demeaned_coords[i,2]*demeaned_coords[i,1]
        g[2,2] += demeaned_coords[i,2]*demeaned_coords[i,2]
    return g / len(demeaned_coords)
    
#THIS CALCULATES THE CENTROSYMMETRY PARAMETER BUT IN 3D AND NOT WEIGHTING THE DISTANCES OTHER THAN TO DETECT NEIGHBORS
#updated on 6/16/25 to return the entire distribution of centrosymmetries found
def calc_centrosymmetry_props_3d(nsteps,phi,charge_frac,test,trajnr):
    """
    Function to calculate the number of clusters as well as cluster centrosymmetry parameters
    Inputs:
        nsteps: the number of trajectory steps to consider when taking averages
        phi: the fraction of free DA in the system
        charge_frac: the fraction of charge on the polymer
        test: if this is True then do some printouts, hide the printouts if false
        trajnr: the trajectory number to read
    """
    
    #GET BASIC INFORMATION AND LOAD THE DATA
    n_DA = 500 #hard-code the total number of DA chromophores
    index_data = np.linspace(0,n_DA-1,n_DA,dtype='int') #data showing indices of DA molecules
    n_polymer = int(np.round((1-float(phi))*n_DA/10)) #number of polymer chains
    trajfile=fr'C:\Users\nicho\Documents\Science\ferroelectricity project\scripts\phi_{phi}_{charge_frac}\traj_{trajnr}.lammpstrj'
    datafile=fr'C:\Users\nicho\Documents\Science\ferroelectricity project\scripts\lammps_inputs\Sys_{phi}_{charge_frac}.data'
    u = mda.Universe(datafile, trajfile, format='LAMMPSDUMP')
    avg_num_clusters = np.zeros(nsteps)
    avg_cluster_size = np.zeros(nsteps)
    avg_assembled = np.zeros(nsteps)
    avg_centrosymmetry = np.zeros(nsteps)
    avg_centrosymmetry_std = np.zeros(nsteps)
    centrosymmetry_vals=np.zeros(n_DA*nsteps) #array to contain all of the different P values for each chromophore calculated in each timestep
    n_centrosymmetry_vals=0 #parameter to keep track of the number of total P values we have calculated
#     polymer_proximities=np.zeros(nsteps)
#     electrostatic_forces=np.zeros(nsteps)
#     avg_cos_diff = np.zeros(nsteps)

    #LOOP THROUGH ALL TIMESTEPS REQUESTED
    for t in range(len(u.trajectory)-nsteps+9,len(u.trajectory),5): #only perform analysis every 5 frames (based on autocorrelation data)
        
        #PERFORM CLUSTERING PROCESS
#         if t%10==0:
#             print(t)
        print(t)
        
        u.trajectory[t] #go to the appropriate trajectory frame
        #get resids for free DAs
        DA_indices=np.linspace(n_polymer,n_polymer+n_DA-1,n_DA,dtype=int)
        free_DAs=u.select_atoms('type 6')
        free_ids=free_DAs.resids-1-n_polymer
        #get resids for DAs which are attached to the polymer
        bound_ids=np.setdiff1d(DA_indices,free_ids)
        #some stuff
        DA_vecs=np.zeros((n_DA,3))
        #get vectors from acceptor to donor group
        for j in range(n_polymer,n_polymer+n_DA):
            sel1=u.select_atoms(f'resid {j+1} and (type 3)')
            sel2=u.select_atoms(f'resid {j+1} and (type 4)')
            DA_vecs[j-n_polymer]=compute_vec_periodic(sel1.positions[0],sel2.positions[0],u.dimensions[:3])
        DA_vecs_raw = np.copy(DA_vecs) #save a copy with the vectors pointing their original directions - we will use this to get the polarization

        #get cog data for DA molecules
        cog_data=np.zeros((n_DA,3)) #array containing center of gravity data of the DA molecules
        for j in range(n_polymer,n_polymer+n_DA):
            selection=u.select_atoms(f'resid {j+1} and (type 3 or type 4)')
            cog_data[j-n_polymer]=compute_cog_periodic(selection.positions,u.dimensions[:3])

        DA_data=np.zeros((n_DA,3)) #contains the theta and phi values corresponding to each DA vector np.arctan2(DA_vecs[i,1],DA_vecs[i,0])+np.pi
        for i in range(n_DA):
            DA_data[i] = np.array([np.arctan2(DA_vecs_raw[i,1],DA_vecs_raw[i,0]), np.arccos(np.abs(DA_vecs_raw[i,2])/np.linalg.norm(DA_vecs_raw[i])),0])
        #do cluster analysis on the theta and phi values
        box1 = freud.box.Box(np.pi,np.pi,0)
        system1 = freud.AABBQuery(box1, DA_data)
        cl1 = freud.cluster.Cluster()
        cl1.compute(system1, neighbors={"mode": 'ball', "r_max": 0.2})
        if test and t == len(u.trajectory)-1:
            #print out a plot of the clusters as well as some information if test is selected
            fig = plt.figure(figsize = (20, 10))
            for cluster_id in range(cl1.num_clusters):
                cluster_system = freud.AABBQuery(system1.box, system1.points[cl1.cluster_keys[cluster_id]])
                if len(cl1.cluster_keys[cluster_id])>5:
                    plt.scatter(cluster_system.points[:,0], cluster_system.points[:,1],label=f'Cluster {cluster_id}')
                    print(
                        f"There are {len(cl1.cluster_keys[cluster_id])} points in cluster {cluster_id}."
                    )
                    intracluster_distances=calc_dist_mat_periodic(DA_data[cl1.cluster_idx==cluster_id],np.array([np.pi,np.pi,0]))
                    intracluster_pos_distances=calc_dist_mat_periodic(cog_data[cl1.cluster_idx==cluster_id],u.dimensions[:3])
                    print(len(intracluster_distances[(intracluster_distances!=0) & (intracluster_pos_distances<=3.5)]))
                    relevant_intracluster_distances=intracluster_distances[(intracluster_distances!=0) & (intracluster_pos_distances<=3.5)]
#                     for i in range(len(cog_data)):
#                         phi_theta_diffs = np.zeros(len(dist_mat[(dist_mat>0) & (dist_mat<=3.5)]))
#                         current_index=0
#                         loc_dists = dist_mat[i]
#                         relevant_data = DA_data[(loc_dists > 0) & (loc_dists <= 3.5)]
#                         phi_theta_diffs[current_index:current_index+len(relevant_data)] = np.linalg.norm(DA_data[i] - relevant_data, axis=1)
#                         current_index += len(relevant_data)
#                     print('50th percentile: ', np.percentile(relevant_intracluster_distances[relevant_intracluster_distances<0.5],50))
#                     print('90th percentile: ', np.percentile(relevant_intracluster_distances[relevant_intracluster_distances<0.5],90))
#                     print('95th percentile: ', np.percentile(relevant_intracluster_distances[relevant_intracluster_distances<0.5],95))
            plt.title("Clusters identified", fontsize=20)
            plt.xlim(-np.pi,np.pi)
            plt.ylim(0,np.pi)
            plt.legend()
            # ax.tick_params(axis="both", which="both", labelsize=14, size=8)
            plt.show()



        #compute proximity data real quick
#             polymer_positions=u.select_atoms('type 7 or type 8').positions
#             polymer_da_distances = calc_dist_mat_periodic_2pos(polymer_positions, cog_data, u.dimensions[:3])
#             polymer_proximities[t-(len(u.trajectory)-nsteps)] = np.sum(polymer_da_distances)/(polymer_da_distances.shape[0]*polymer_da_distances.shape[1])
#             da_charge_positions=u.select_atoms('type 6').positions
#             polymer_charge_positions=u.select_atoms('type 8').positions
#             polymer_charge_distances = calc_dist_mat_periodic_2pos(polymer_charge_positions, da_charge_positions, u.dimensions[:3])
#             electrostatic_forces[t-(len(u.trajectory)-nsteps)] = np.sum(1/polymer_charge_distances)
        #do distance-based clustering within the angle-based clusters
        final_cluster_indices = np.zeros(n_DA)
        num_clusters_found = 0
        for i in range(cl1.num_clusters):
            box = freud.box.Box(u.dimensions[0],u.dimensions[1],u.dimensions[2])
            system = freud.AABBQuery(box, cog_data[cl1.cluster_idx==i])
            cl = freud.cluster.Cluster()
            cl.compute(system, neighbors={"mode": 'ball', "r_max": 3.5})
            cluster_indices = np.copy(cl.cluster_idx)
            cluster_indices += num_clusters_found
            final_cluster_indices[cl1.cluster_idx==i] = cluster_indices
            num_clusters_found += cl.num_clusters
        final_num_clusters = len(np.unique(final_cluster_indices))

        min_cluster_size = 25
        #calculate number of clusters which have at least 'min_cluster_size' members
        n_threshold_clusters = 0
        threshold_cluster_indices = np.zeros(100)
        for i in range(final_num_clusters):
            if len(cog_data[final_cluster_indices==i]) >= min_cluster_size:
                threshold_cluster_indices[n_threshold_clusters] = i
                n_threshold_clusters += 1
        threshold_cluster_indices = threshold_cluster_indices[:n_threshold_clusters]


        if test and t == len(u.trajectory)-1:
            #display the results
            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
            for i in range(final_num_clusters):
                cluster_points = cog_data[final_cluster_indices==i]
                n_points = len(cluster_points)
                if n_points >= min_cluster_size:
                    ax.scatter3D(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2],label=f'Cluster {i}')
                    print(
                        f"There are {n_points} points in cluster {i}."
                    )
            ax.set_title("Clusters Identified", fontsize=20)
            ax.legend(bbox_to_anchor=(1.5, 1.05),fontsize=14)
            ax.tick_params(axis="both", which="both", labelsize=14, size=8)
            ax.set_xlabel('$X$')
            ax.set_ylabel('$Y$')
            ax.set_zlabel('$Z$')
            plt.show() 

        #CALCULATE CLUSTER PROPERTIES
        cluster_sizes = np.zeros(n_threshold_clusters)
        centrosymmetries = np.zeros(n_threshold_clusters)
        centrosymmetry_stds = np.zeros(n_threshold_clusters)
        interior_cluster_sizes = np.zeros(n_threshold_clusters) #contains information about interior sizes of threshold clusters to weight centrosym averages
        for i in range(n_threshold_clusters):
            cluster_sizes[i] = len(cog_data[final_cluster_indices==threshold_cluster_indices[i]])
            indices = np.where(final_cluster_indices==threshold_cluster_indices[i])[0]
            positions = cog_data[final_cluster_indices==threshold_cluster_indices[i]]
            box_dims = np.array([u.dimensions[0],u.dimensions[1],u.dimensions[2]])
            #compute the gyration tensor and the eigenvector with smallest eigenvalue
            positions = center(positions,box_dims) #center the cluster
            gyration_t = gyration_tensor(positions,box_dims) #calculate gyration tensor of the cluster
            evals,evecs = np.linalg.eig(gyration_t)
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:,idx]
            transition_mat = np.transpose(evecs) #matrix to transform positions to orthonormal coordinates
            new_positions = np.dot(positions, evecs)

            #calculate centrosymmetry parameters for the new positions
            renorm_positions = np.copy(new_positions)
            renorm_positions[:,0] = renorm_positions[:,0] / 3.3 #normalize by the face-to-face distance
            renorm_positions[:,1] = renorm_positions[:,1] / 7.5 #normalize by the edge-to-edge distance
            renorm_positions[:,2] = 0
            dist_mat = calc_dist_mat(renorm_positions)
            centro_syms = np.zeros(n_DA)
            edge_vals = np.zeros(n_DA)
            edge_ids = np.zeros(n_DA) #array noting if our points are edges (0 is not edge, 1 is edge)
            interior_ids = np.zeros(n_DA,dtype=int) #array noting if our points are on the interior or not (contains indices of all points in interior of a given cluster)
            n_interior=0 #number of interior points in a given cluster
            for pos in range(len(positions)):
                sorted_distmat = np.argsort(dist_mat[pos])
                closest_atoms=sorted_distmat[1:5]
                #TEST: Make plots of closest positions to certain DA molecules
#                 if pos % 50 == 0:
#                     plt.figure(figsize=(12,10))
#                     plt.scatter(renorm_positions[:,0]*3,renorm_positions[:,1]*3,s=5)
#                     plt.scatter(renorm_positions[pos,0]*3,renorm_positions[pos,1]*3,s=5,color='yellow')
#                     plt.scatter(renorm_positions[closest_atoms,0]*3,renorm_positions[closest_atoms,1]*3,s=5,color='red')
#                     plt.xlim(-40,40)
#                     plt.ylim(-40,40)
#                     plt.show()
                closest_vecs = renorm_positions[pos] - renorm_positions[closest_atoms]
                vec_sums = np.linalg.norm(closest_vecs[:, np.newaxis, :] + closest_vecs[np.newaxis, :, :],axis=2)
                #can we do some kind of check to make sure we aren't dealing with the edge of a cluster?

                # Compute the 8 nearest neighbor distances
                closest_neighbors=sorted_distmat[1:9]
                neighbor_distances=np.linalg.norm(renorm_positions[pos]-renorm_positions[closest_neighbors],axis=1)
                # Calculate the average distance to 8 neighbors for each particle
                avg_distance = np.mean(neighbor_distances)
                closest_vecs_real = new_positions[pos] - new_positions[closest_atoms]
                vec_sums_real = np.linalg.norm(closest_vecs_real[:, np.newaxis, :] + closest_vecs_real[np.newaxis, :, :],axis=2)
                centro_syms[indices[pos]] = np.sum(np.sort(vec_sums_real[np.triu_indices(vec_sums_real.shape[0], k=1)])[:2]**2)
                max_cs_val = 10
                edge_vals[indices[pos]] = avg_distance
                if avg_distance >= 0.4: #we ran some tests with histograms of this value and 0.42 seemed to be appropriate for the bimodal distribution of values
#                 if avg_distance >= 0.435: #we ran some tests with histograms of this value and 0.42 seemed to be appropriate for the bimodal distribution of values
                    edge_ids[indices[pos]] = 1
                else:
                    if centro_syms[indices[pos]] < max_cs_val:
                        interior_ids[n_interior] = indices[pos]
                        n_interior += 1
            interior_ids=interior_ids[:n_interior]
            interior_cluster_sizes[i] = len(interior_ids)
            #take the average of centrosymmetries for particles which are in the cluster and are not edges
#             print(t, i, len(centro_syms[np.intersect1d(interior_ids,indices)]))
            if interior_cluster_sizes[i] > 0:
#                 print(n_centrosymmetry_vals,n_centrosymmetry_vals+int(interior_cluster_sizes[i]))
                centrosymmetry_vals[n_centrosymmetry_vals:n_centrosymmetry_vals+int(interior_cluster_sizes[i])]=centro_syms[np.intersect1d(interior_ids,indices)]
                n_centrosymmetry_vals += int(interior_cluster_sizes[i])
                centrosymmetries[i] = np.mean(centro_syms[np.intersect1d(interior_ids,indices)])
                centrosymmetry_stds[i] = np.std(centro_syms[np.intersect1d(interior_ids,indices)])

            #make a printout of the large clusters
            if test and t == len(u.trajectory)-1:

                #this plots the renormalized cluster positions
#                 plt.figure(figsize=(12,10))
#                 plt.scatter(renorm_positions[:,0]*3,renorm_positions[:,1]*3,s=5)
#                 plt.xlim(-40,40)
#                 plt.ylim(-40,40)
#                 plt.show()

                #rotate cluster positions to align with coordinate axis
                rotated_cog_data = np.copy(cog_data)
                for j in range(len(positions)):
                    rotated_cog_data[indices[j]] = new_positions[j]
                #discriminate between positions which are free vs bound
                free_positions = rotated_cog_data[np.intersect1d(np.intersect1d(free_ids,indices),interior_ids)]
                free_centro_syms = centro_syms[np.intersect1d(np.intersect1d(free_ids,indices),interior_ids)]
                free_edge_vals = edge_vals[np.intersect1d(np.intersect1d(free_ids,indices),interior_ids)]
                bound_positions = rotated_cog_data[np.intersect1d(np.intersect1d(bound_ids,indices),interior_ids)]
                bound_centro_syms = centro_syms[np.intersect1d(np.intersect1d(bound_ids,indices),interior_ids)]
                bound_edge_vals = edge_vals[np.intersect1d(np.intersect1d(bound_ids,indices),interior_ids)]
                relevant_centro_syms = centro_syms[np.intersect1d(interior_ids,indices)]
                relevant_edge_vals = edge_vals[indices]
                edge_positions = rotated_cog_data[edge_ids==1]
                #new functionality (6/7): only show points which are in the cluster interior
                if len(relevant_centro_syms>0):
                    plt.figure()
    #                 plt.scatter(free_positions[:,0]*3,free_positions[:,1]*3,s=10,marker='o')
    #                 plt.scatter(bound_positions[:,0]*3,bound_positions[:,1]*3,s=10,marker='x')
    #                 plt.scatter(free_positions[:,0]*3,free_positions[:,1]*3,c=free_centro_syms,cmap='inferno',s=10,marker='o',vmin=0,vmax=0.12)#vmax=np.max(relevant_centro_syms))
    #                 plt.scatter(bound_positions[:,0]*3,bound_positions[:,1]*3,c=bound_centro_syms,cmap='inferno',s=10,marker='x',vmin=0,vmax=0.12)#vmax=np.max(relevant_centro_syms))
                    plt.scatter(free_positions[:,0]*3,free_positions[:,1]*3,c=free_centro_syms,cmap='viridis',s=40,marker='o',vmin=0,vmax=5.75)#vmax=np.max(relevant_centro_syms))
                    plt.scatter(bound_positions[:,0]*3,bound_positions[:,1]*3,c=bound_centro_syms,cmap='viridis',s=40,marker='x',vmin=0,vmax=5.75)#vmax=np.max(relevant_centro_syms))
    #                 plt.scatter(free_positions[:,0]*3,free_positions[:,1]*3,c=free_centro_syms,cmap='viridis',s=10,marker='o',vmin=0,vmax=0.25)
    #                 plt.scatter(bound_positions[:,0]*3,bound_positions[:,1]*3,c=bound_centro_syms,cmap='viridis',s=10,marker='x',vmin=0,vmax=0.25)
                    plt.colorbar(label='Centrosymmetry Parameter ($\AA$)')
                    #ax = plt.gca()
                    #ax.set_facecolor((210/256,210/256,210/256))
                    plt.xlabel('Position ($\AA$)')
                    plt.ylabel('Position ($\AA$)')
                    plt.savefig(f'C:\\Users\\nicho\\Documents\\Science\\ferroelectricity project\\figures\\heatmaps\\heatmap_phi_{phi}_chargefrac_{charge_frac}_run_{trajnr}_cluster_{i}.svg',format='svg',transparent=True)
                    plt.show()

                    #below plot is for edge detection purposes only
                    plt.figure()
                    plt.scatter(free_positions[:,0]*3,free_positions[:,1]*3,c='blue',marker='o',label='Internal Chromophores')
                    plt.scatter(bound_positions[:,0]*3,bound_positions[:,1]*3,c='blue',marker='o')
    #                 plt.scatter(free_positions[:,0]*3,free_positions[:,1]*3,c=free_centro_syms,cmap='viridis',s=10,marker='o',vmin=0,vmax=0.08)
    #                 plt.scatter(bound_positions[:,0]*3,bound_positions[:,1]*3,c=bound_centro_syms,cmap='viridis',s=10,marker='x',vmin=0,vmax=0.08)
                    plt.scatter(edge_positions[:,0]*3,edge_positions[:,1]*3,c='red',marker='o',label='Edge Chromophores')
#                     plt.colorbar(label='Centrosymmetry Parameter ($\AA$)')
                    plt.legend()
                    plt.xlabel('Position ($\AA$)')
                    plt.ylabel('Position ($\AA$)')
                    # plt.savefig(f'C:\\Users\\nicho\\Documents\\Science\\ferroelectricity project\\figures\\edgemap_phi_{phi}_chargefrac_{charge_frac}_run_{trajnr}_cluster_{i}.svg',format='svg',transparent=True)
                    plt.show()
                    #below histogram is for edge detection purposes only
                    plt.figure()
                    plt.hist(relevant_edge_vals,bins=50)
                    plt.minorticks_on()
                    plt.show()

        avg_num_clusters[t-(len(u.trajectory)-nsteps)] = n_threshold_clusters
        if (np.sum(interior_cluster_sizes)!=0):
            avg_assembled[t-(len(u.trajectory)-nsteps)] = np.sum(cluster_sizes)
            avg_cluster_size[t-(len(u.trajectory)-nsteps)] = np.mean(cluster_sizes)
            avg_centrosymmetry[t-(len(u.trajectory)-nsteps)] = np.average(centrosymmetries,weights=interior_cluster_sizes)
            avg_centrosymmetry_std[t-(len(u.trajectory)-nsteps)] = np.average(centrosymmetry_stds,weights=interior_cluster_sizes)
        else:
            avg_assembled[t-(len(u.trajectory)-nsteps)] = -1
            avg_cluster_size[t-(len(u.trajectory)-nsteps)] = -1
            avg_centrosymmetry[t-(len(u.trajectory)-nsteps)] = -1
            avg_centrosymmetry_std[t-(len(u.trajectory)-nsteps)] = -1

    centrosymmetry_vals=centrosymmetry_vals[:n_centrosymmetry_vals]
#         adr=fr'C:\Users\nicho\Documents\Science\ferroelectricity project\p_values_phi_{phi}.dat'
#         np.savetxt(adr,centrosymmetry_vals,fmt='%.6f')

    #show histogram of centrosymmetry values
    plt.figure()
    plt.hist(centrosymmetry_vals,bins='auto')
    plt.show()
    
    print(np.mean(avg_assembled), np.mean(avg_num_clusters), np.mean(avg_cluster_size[avg_cluster_size!=-1]), np.mean(avg_centrosymmetry[avg_centrosymmetry!=-1]), np.mean(avg_centrosymmetry_std[avg_centrosymmetry_std!=-1]), np.std(avg_centrosymmetry[avg_centrosymmetry!=-1]), n_centrosymmetry_vals)
    return centrosymmetry_vals

#THIS CALCULATES THE CENTROSYMMETRY PARAMETER BUT IN 3D AND NOT WEIGHTING THE DISTANCES OTHER THAN TO DETECT NEIGHBORS
def calc_sample_correlation(nsteps,phi,charge_frac,test,trajnr):
    """
    Function to estimate the number of trajectory frames between independent samples
    Inputs:
        nsteps: the number of trajectory steps to consider when taking averages
        phi: the fraction of free DA in the system
        charge_frac: the fraction of charge on the polymer
        test: if this is True then do some printouts, hide the printouts if false
        trajnr: the trajectory number to read
    """
    
    #GET BASIC INFORMATION AND LOAD THE DATA
    n_DA = 500 #hard-code the total number of DA chromophores
    index_data = np.linspace(0,n_DA-1,n_DA,dtype='int') #data showing indices of DA molecules
    n_polymer = int(np.round((1-float(phi))*n_DA/10)) #number of polymer chains
    trajfile=fr'C:\Users\nicho\Documents\Science\ferroelectricity project\scripts\phi_{phi}_{charge_frac}\traj_{trajnr}.lammpstrj'
    datafile=fr'C:\Users\nicho\Documents\Science\ferroelectricity project\scripts\lammps_inputs\Sys_{phi}_{charge_frac}.data'
    u = mda.Universe(datafile, trajfile, format='LAMMPSDUMP')
    
    corr_vals=np.zeros(nsteps)

    #LOOP THROUGH ALL TIMESTEPS REQUESTED
    for t in range(len(u.trajectory)-nsteps,len(u.trajectory)):
        
        if t % 10 == 1:
            print(t)
        u.trajectory[t] #go to the appropriate trajectory frame
        
        #get cog data for DA molecules
        cog_data=np.zeros((n_DA,3)) #array containing center of gravity (mass) data of the DA molecules
        for j in range(n_polymer,n_polymer+n_DA):
            selection=u.select_atoms(f'resid {j+1} and (type 3 or type 4)')
            cog_data[j-n_polymer]=compute_cog_periodic(selection.positions,u.dimensions[:3])
        cog_data = cog_data - np.mean(cog_data,axis=0)
            
        if t==len(u.trajectory)-nsteps:
            cog_data_0 = cog_data
        
        corr_vals[t-(len(u.trajectory)-nsteps)] = np.mean(np.sum(cog_data*cog_data_0,axis=1))
    
    return np.reshape(corr_vals,(corr_vals.shape[0],1))
    
#THIS CALCULATES THE INTEGRATION FRACTION
#integration fraction is defined as the fraction of segments which are 'integrated':
#which are connected to the same assembly at both of their chromophore ends
#also obtains the fraction of functionalized chromophores which are part of a large assembly
def calc_integration(nsteps,phi,charge_frac,test,trajnr):
    """
    Function to calculate the number of clusters as well as cluster centrosymmetry parameters
    Inputs:
        nsteps: the number of trajectory steps to consider when taking averages
        phi: the fraction of free DA in the system
        charge_frac: the fraction of charge on the polymer
        test: if this is True then do some printouts, hide the printouts if false
        trajnr: the trajectory number to read
    """
    
    #GET BASIC INFORMATION AND LOAD THE DATA
    n_DA = 500 #hard-code the total number of DA chromophores
    index_data = np.linspace(0,n_DA-1,n_DA,dtype='int') #data showing indices of DA molecules
    n_polymer = int(np.round((1-float(phi))*n_DA/10)) #number of polymer chains
    trajfile=fr'C:\Users\nicho\Documents\Science\ferroelectricity project\scripts\phi_{phi}_{charge_frac}\traj_{trajnr}.lammpstrj'
    datafile=fr'C:\Users\nicho\Documents\Science\ferroelectricity project\scripts\lammps_inputs\Sys_{phi}_{charge_frac}.data'
    u = mda.Universe(datafile, trajfile, format='LAMMPSDUMP')
    avg_num_clusters = np.zeros(nsteps)
    integration_fractions = np.zeros(nsteps)
    incorporation_fractions = np.zeros(nsteps)
    coassembly_fractions = np.zeros(nsteps)
    skip=5

    #LOOP THROUGH ALL TIMESTEPS REQUESTED
    for t in range(len(u.trajectory)-nsteps,len(u.trajectory),skip):
        
        #PERFORM CLUSTERING PROCESS
        print(t)
        u.trajectory[t] #go to the appropriate trajectory frame
        #get resids for free DAs
        DA_indices=np.linspace(n_polymer,n_polymer+n_DA-1,n_DA,dtype=int)
        free_DAs=u.select_atoms('type 6')
        free_ids=free_DAs.resids-1-n_polymer
        #get resids for DAs which are attached to the polymer
        bound_ids=np.setdiff1d(DA_indices,free_ids)
        #some stuff
        DA_vecs=np.zeros((n_DA,3))
        #get vectors from acceptor to donor group
        for j in range(n_polymer,n_polymer+n_DA):
            sel1=u.select_atoms(f'resid {j+1} and (type 3)')
            sel2=u.select_atoms(f'resid {j+1} and (type 4)')
            DA_vecs[j-n_polymer]=compute_vec_periodic(sel1.positions[0],sel2.positions[0],u.dimensions[:3])
        DA_vecs_raw = np.copy(DA_vecs) #save a copy with the vectors pointing their original directions - we will use this to get the polarization
        
        #get cog data for DA molecules
        cog_data=np.zeros((n_DA,3)) #array containing center of gravity data of the DA molecules
        for j in range(n_polymer,n_polymer+n_DA):
            selection=u.select_atoms(f'resid {j+1} and (type 3 or type 4)')
            cog_data[j-n_polymer]=compute_cog_periodic(selection.positions,u.dimensions[:3])
            
        DA_data=np.zeros((n_DA,3)) #contains the theta and phi values corresponding to each DA vector np.arctan2(DA_vecs[i,1],DA_vecs[i,0])+np.pi
        for i in range(n_DA):
            DA_data[i] = np.array([np.arctan2(DA_vecs_raw[i,1],DA_vecs_raw[i,0]), np.arccos(np.abs(DA_vecs_raw[i,2])/np.linalg.norm(DA_vecs_raw[i])),0])
        #do cluster analysis on the theta and phi values
        box1 = freud.box.Box(np.pi,np.pi,0)
        system1 = freud.AABBQuery(box1, DA_data)
        cl1 = freud.cluster.Cluster()
        cl1.compute(system1, neighbors={"mode": 'ball', "r_max": 0.2})
        if test and t == len(u.trajectory)-1:
            #print out a plot of the clusters as well as some information if test is selected
            fig = plt.figure(figsize = (20, 10))
            for cluster_id in range(cl1.num_clusters):
                cluster_system = freud.AABBQuery(system1.box, system1.points[cl1.cluster_keys[cluster_id]])
                if len(cl1.cluster_keys[cluster_id])>5:
                    plt.scatter(cluster_system.points[:,0], cluster_system.points[:,1],label=f'Cluster {cluster_id}')
                    print(
                        f"There are {len(cl1.cluster_keys[cluster_id])} points in cluster {cluster_id}."
                    )
                    intracluster_distances=calc_dist_mat_periodic(DA_data[cl1.cluster_idx==cluster_id],np.array([np.pi,np.pi,0]))
                    intracluster_pos_distances=calc_dist_mat_periodic(cog_data[cl1.cluster_idx==cluster_id],u.dimensions[:3])
                    print(len(intracluster_distances[(intracluster_distances!=0) & (intracluster_pos_distances<=3.5)]))
                    relevant_intracluster_distances=intracluster_distances[(intracluster_distances!=0) & (intracluster_pos_distances<=3.5)]
#                     for i in range(len(cog_data)):
#                         phi_theta_diffs = np.zeros(len(dist_mat[(dist_mat>0) & (dist_mat<=3.5)]))
#                         current_index=0
#                         loc_dists = dist_mat[i]
#                         relevant_data = DA_data[(loc_dists > 0) & (loc_dists <= 3.5)]
#                         phi_theta_diffs[current_index:current_index+len(relevant_data)] = np.linalg.norm(DA_data[i] - relevant_data, axis=1)
#                         current_index += len(relevant_data)
#                     print('50th percentile: ', np.percentile(relevant_intracluster_distances[relevant_intracluster_distances<0.5],50))
#                     print('90th percentile: ', np.percentile(relevant_intracluster_distances[relevant_intracluster_distances<0.5],90))
#                     print('95th percentile: ', np.percentile(relevant_intracluster_distances[relevant_intracluster_distances<0.5],95))
            plt.title("Clusters identified", fontsize=20)
            plt.xlim(-np.pi,np.pi)
            plt.ylim(0,np.pi)
            plt.legend()
            # ax.tick_params(axis="both", which="both", labelsize=14, size=8)
            plt.show()
            
        #do distance-based clustering within the angle-based clusters
        final_cluster_indices = np.zeros(n_DA) #an array of all DA chromophores and their corresponding cluster indices
        num_clusters_found = 0
        for i in range(cl1.num_clusters):
            box = freud.box.Box(u.dimensions[0],u.dimensions[1],u.dimensions[2])
            system = freud.AABBQuery(box, cog_data[cl1.cluster_idx==i])
            cl = freud.cluster.Cluster()
            cl.compute(system, neighbors={"mode": 'ball', "r_max": 3.5})
            cluster_indices = np.copy(cl.cluster_idx)
            cluster_indices += num_clusters_found
            final_cluster_indices[cl1.cluster_idx==i] = cluster_indices
            num_clusters_found += cl.num_clusters
        final_num_clusters = len(np.unique(final_cluster_indices))
        
        min_cluster_size = 25
        #calculate number of clusters which have at least 'min_cluster_size' members
        n_threshold_clusters = 0
        threshold_cluster_indices = np.zeros(100)
        for i in range(final_num_clusters):
            if len(cog_data[final_cluster_indices==i]) >= min_cluster_size:
                threshold_cluster_indices[n_threshold_clusters] = i
                n_threshold_clusters += 1
        threshold_cluster_indices = threshold_cluster_indices[:n_threshold_clusters]
                
        if test and t == len(u.trajectory)-1:
            #display the results
            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
            for i in range(final_num_clusters):
                cluster_points = cog_data[final_cluster_indices==i]
                n_points = len(cluster_points)
                if n_points >= min_cluster_size:
                    ax.scatter3D(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2],label=f'Cluster {i}')
                    print(
                        f"There are {n_points} points in cluster {i}."
                    )
            ax.set_title("Clusters Identified", fontsize=20)
            ax.legend(bbox_to_anchor=(1.5, 1.05),fontsize=14)
            ax.tick_params(axis="both", which="both", labelsize=14, size=8)
            ax.set_xlabel('$X$')
            ax.set_ylabel('$Y$')
            ax.set_zlabel('$Z$')
            plt.show() 
            
        #CALCULATE INTEGRATION PROPERTIES
        polymer_length = 100
        nseg = 9
        functionalization = 10 #percent functionalization
        n_func = polymer_length * functionalization / 100
        n_total_seg = nseg*n_polymer
        n_connected_seg = 0
        n_incorporated_chromophores = 0
        coassembly_vals = np.zeros(n_polymer) #0 is unassembled, 1 is assembled
        for i in range(n_polymer):
            da_cluster_ids = np.zeros(nseg+1)
            da_func_id = 0
            for j in range(polymer_length):
                if len(u.atoms[j+polymer_length*i].bonds)>2:
                    bound_chromophore_id = u.atoms[j+polymer_length*i].bonds[2].partner(u.atoms[j+polymer_length*i]).resid
                    da_cluster_ids[da_func_id] = final_cluster_indices[bound_chromophore_id-n_polymer-1]
                    if len(cog_data[final_cluster_indices==da_cluster_ids[da_func_id]])>min_cluster_size:
                        n_incorporated_chromophores += 1
                        if coassembly_vals[i]==0:
                            coassembly_vals[i] = 1
                    da_func_id += 1
#             print(f'cluster ids: {da_cluster_ids}')
            for j in range(nseg):
                if da_cluster_ids[j] == da_cluster_ids[j+1]:
                    n_connected_seg += 1
        integration_fractions[t-(len(u.trajectory)-nsteps)] = n_connected_seg / n_total_seg
        incorporation_fractions[t-(len(u.trajectory)-nsteps)] = n_incorporated_chromophores / (n_polymer*n_func)
        coassembly_fractions[t-(len(u.trajectory)-nsteps)] = np.mean(coassembly_vals)
    
#     times=np.linspace(0,nsteps-1,nsteps)
#     plt.plot(times,integration_fractions)
#     return np.mean(integration_fractions), np.std(integration_fractions), np.mean(incorporation_fractions), np.std(incorporation_fractions), np.mean(coassembly_fractions), np.std(coassembly_fractions)
    return incorporation_fractions, coassembly_fractions
    
#THIS CALCULATES THE INTEGRATION FRACTION
#integration fraction is defined as the fraction of segments which are 'integrated':
#which are connected to the same assembly at both of their chromophore ends
#also obtains the fraction of functionalized chromophores which are part of a large assembly
def calc_integration_alt(nsteps,phi,charge_frac,test,trajnr):
    """
    Function to calculate the number of clusters as well as cluster centrosymmetry parameters
    Inputs:
        nsteps: the number of trajectory steps to consider when taking averages
        phi: the fraction of free DA in the system
        charge_frac: the fraction of charge on the polymer
        test: if this is True then do some printouts, hide the printouts if false
        trajnr: the trajectory number to read
    """
    
    #GET BASIC INFORMATION AND LOAD THE DATA
    n_DA = 500 #hard-code the total number of DA chromophores
    index_data = np.linspace(0,n_DA-1,n_DA,dtype='int') #data showing indices of DA molecules
    n_polymer = int(np.round((1-float(phi))*n_DA/10)) #number of polymer chains
    skip=10
    trajfile=fr'C:\Users\nicho\Documents\Science\ferroelectricity project\scripts\phi_{phi}_{charge_frac}\traj_{trajnr}.lammpstrj'
    datafile=fr'C:\Users\nicho\Documents\Science\ferroelectricity project\scripts\lammps_inputs\Sys_{phi}_{charge_frac}.data'
    u = mda.Universe(datafile, trajfile, format='LAMMPSDUMP')
    print(len(u.trajectory))
    avg_num_clusters = np.zeros(nsteps//skip)
    integration_fractions = np.zeros(nsteps//skip)
    incorporation_fractions = np.zeros(nsteps//skip)
    coassembly_fractions = np.zeros(nsteps//skip)

    #LOOP THROUGH ALL TIMESTEPS REQUESTED
#     for t in range(len(u.trajectory)-nsteps,len(u.trajectory)):
    for t in range(0,nsteps,skip): #sample integration every 'skip' timesteps
        
        #PERFORM CLUSTERING PROCESS
#         if t % 10 == 1:
        print(t)
        u.trajectory[t] #go to the appropriate trajectory frame
        #get resids for free DAs
        DA_indices=np.linspace(n_polymer,n_polymer+n_DA-1,n_DA,dtype=int)
        free_DAs=u.select_atoms('type 6')
        free_ids=free_DAs.resids-1-n_polymer
        #get resids for DAs which are attached to the polymer
        bound_ids=np.setdiff1d(DA_indices,free_ids)
        #some stuff
        DA_vecs=np.zeros((n_DA,3))
        #get vectors from acceptor to donor group
        for j in range(n_polymer,n_polymer+n_DA):
            sel1=u.select_atoms(f'resid {j+1} and (type 3)')
            sel2=u.select_atoms(f'resid {j+1} and (type 4)')
            DA_vecs[j-n_polymer]=compute_vec_periodic(sel1.positions[0],sel2.positions[0],u.dimensions[:3])
        DA_vecs_raw = np.copy(DA_vecs) #save a copy with the vectors pointing their original directions - we will use this to get the polarization
        
        #get cog data for DA molecules
        cog_data=np.zeros((n_DA,3)) #array containing center of gravity data of the DA molecules
        for j in range(n_polymer,n_polymer+n_DA):
            selection=u.select_atoms(f'resid {j+1} and (type 3 or type 4)')
            cog_data[j-n_polymer]=compute_cog_periodic(selection.positions,u.dimensions[:3])
            
        DA_data=np.zeros((n_DA,3)) #contains the theta and phi values corresponding to each DA vector np.arctan2(DA_vecs[i,1],DA_vecs[i,0])+np.pi
        for i in range(n_DA):
            DA_data[i] = np.array([np.arctan2(DA_vecs_raw[i,1],DA_vecs_raw[i,0]), np.arccos(np.abs(DA_vecs_raw[i,2])/np.linalg.norm(DA_vecs_raw[i])),0])
        #do cluster analysis on the theta and phi values
        box1 = freud.box.Box(np.pi,np.pi,0)
        system1 = freud.AABBQuery(box1, DA_data)
        cl1 = freud.cluster.Cluster()
        cl1.compute(system1, neighbors={"mode": 'ball', "r_max": 0.2})
        if test and t == nsteps-1:
            #print out a plot of the clusters as well as some information if test is selected
            fig = plt.figure(figsize = (20, 10))
            for cluster_id in range(cl1.num_clusters):
                cluster_system = freud.AABBQuery(system1.box, system1.points[cl1.cluster_keys[cluster_id]])
                if len(cl1.cluster_keys[cluster_id])>5:
                    plt.scatter(cluster_system.points[:,0], cluster_system.points[:,1],label=f'Cluster {cluster_id}')
                    print(
                        f"There are {len(cl1.cluster_keys[cluster_id])} points in cluster {cluster_id}."
                    )
                    intracluster_distances=calc_dist_mat_periodic(DA_data[cl1.cluster_idx==cluster_id],np.array([np.pi,np.pi,0]))
                    intracluster_pos_distances=calc_dist_mat_periodic(cog_data[cl1.cluster_idx==cluster_id],u.dimensions[:3])
                    print(len(intracluster_distances[(intracluster_distances!=0) & (intracluster_pos_distances<=3.5)]))
                    relevant_intracluster_distances=intracluster_distances[(intracluster_distances!=0) & (intracluster_pos_distances<=3.5)]
#                     for i in range(len(cog_data)):
#                         phi_theta_diffs = np.zeros(len(dist_mat[(dist_mat>0) & (dist_mat<=3.5)]))
#                         current_index=0
#                         loc_dists = dist_mat[i]
#                         relevant_data = DA_data[(loc_dists > 0) & (loc_dists <= 3.5)]
#                         phi_theta_diffs[current_index:current_index+len(relevant_data)] = np.linalg.norm(DA_data[i] - relevant_data, axis=1)
#                         current_index += len(relevant_data)
#                     print('50th percentile: ', np.percentile(relevant_intracluster_distances[relevant_intracluster_distances<0.5],50))
#                     print('90th percentile: ', np.percentile(relevant_intracluster_distances[relevant_intracluster_distances<0.5],90))
#                     print('95th percentile: ', np.percentile(relevant_intracluster_distances[relevant_intracluster_distances<0.5],95))
            plt.title("Clusters identified", fontsize=20)
            plt.xlim(-np.pi,np.pi)
            plt.ylim(0,np.pi)
            plt.legend()
            # ax.tick_params(axis="both", which="both", labelsize=14, size=8)
            plt.show()
            
        #do distance-based clustering within the angle-based clusters
        final_cluster_indices = np.zeros(n_DA) #an array of all DA chromophores and their corresponding cluster indices
        num_clusters_found = 0
        for i in range(cl1.num_clusters):
            box = freud.box.Box(u.dimensions[0],u.dimensions[1],u.dimensions[2])
            system = freud.AABBQuery(box, cog_data[cl1.cluster_idx==i])
            cl = freud.cluster.Cluster()
            cl.compute(system, neighbors={"mode": 'ball', "r_max": 3.5})
            cluster_indices = np.copy(cl.cluster_idx)
            cluster_indices += num_clusters_found
            final_cluster_indices[cl1.cluster_idx==i] = cluster_indices
            num_clusters_found += cl.num_clusters
        final_num_clusters = len(np.unique(final_cluster_indices))
        
        min_cluster_size = 25
        #calculate number of clusters which have at least 'min_cluster_size' members
        n_threshold_clusters = 0
        threshold_cluster_indices = np.zeros(100)
        for i in range(final_num_clusters):
            if len(cog_data[final_cluster_indices==i]) >= min_cluster_size:
                threshold_cluster_indices[n_threshold_clusters] = i
                n_threshold_clusters += 1
        threshold_cluster_indices = threshold_cluster_indices[:n_threshold_clusters]
                
        if test and t == nsteps-1:
            #display the results
            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
            for i in range(final_num_clusters):
                cluster_points = cog_data[final_cluster_indices==i]
                n_points = len(cluster_points)
                if n_points >= min_cluster_size:
                    ax.scatter3D(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2],label=f'Cluster {i}')
                    print(
                        f"There are {n_points} points in cluster {i}."
                    )
            ax.set_title("Clusters Identified", fontsize=20)
            ax.legend(bbox_to_anchor=(1.5, 1.05),fontsize=14)
            ax.tick_params(axis="both", which="both", labelsize=14, size=8)
            ax.set_xlabel('$X$')
            ax.set_ylabel('$Y$')
            ax.set_zlabel('$Z$')
            plt.show() 
            
        #CALCULATE INTEGRATION PROPERTIES
        polymer_length = 100
        nseg = 9
        functionalization = 10 #percent functionalization
        n_func = polymer_length * functionalization / 100
        n_total_seg = nseg*n_polymer
        n_connected_seg = 0
        n_incorporated_chromophores = 0
        coassembly_vals = np.zeros(n_polymer) #0 is unassembled, 1 is assembled
        for i in range(n_polymer):
            da_cluster_ids = np.zeros(nseg+1)
            da_func_id = 0
            for j in range(polymer_length):
                if len(u.atoms[j+polymer_length*i].bonds)>2:
                    bound_chromophore_id = u.atoms[j+polymer_length*i].bonds[2].partner(u.atoms[j+polymer_length*i]).resid
                    da_cluster_ids[da_func_id] = final_cluster_indices[bound_chromophore_id-n_polymer-1]
                    if len(cog_data[final_cluster_indices==da_cluster_ids[da_func_id]])>min_cluster_size:
                        n_incorporated_chromophores += 1
                        if coassembly_vals[i]==0:
                            coassembly_vals[i] = 1
                    da_func_id += 1
#             print(f'cluster ids: {da_cluster_ids}')
            for j in range(nseg):
                if da_cluster_ids[j] == da_cluster_ids[j+1]:
                    n_connected_seg += 1
#         integration_fractions[t-(len(u.trajectory)-nsteps)] = n_connected_seg / n_total_seg
#         incorporation_fractions[t-(len(u.trajectory)-nsteps)] = n_incorporated_chromophores / (n_polymer*n_func)
#         coassembly_fractions[t-(len(u.trajectory)-nsteps)] = np.mean(coassembly_vals)
        integration_fractions[t//skip] = n_connected_seg / n_total_seg
        incorporation_fractions[t//skip] = n_incorporated_chromophores / (n_polymer*n_func)
        coassembly_fractions[t//skip] = np.mean(coassembly_vals)
    
#     times=np.linspace(0,nsteps-1,nsteps)
#     plt.plot(times,integration_fractions)
#     return np.mean(integration_fractions), np.std(integration_fractions), np.mean(incorporation_fractions), np.std(incorporation_fractions), np.mean(coassembly_fractions), np.std(coassembly_fractions)
    print(incorporation_fractions)
    return incorporation_fractions
    
#THIS CALCULATES THE CENTROSYMMETRY PARAMETER BUT IN 3D AND NOT WEIGHTING THE DISTANCES OTHER THAN TO DETECT NEIGHBORS
#updated on 6/16/25 to return the entire distribution of centrosymmetries found
#the alternate version takes the first nsteps of a trajectory rather than
#the last nsteps
def calc_centrosymmetry_props_3d_alt(nsteps,phi,charge_frac,test,trajnr):
    """
    Function to calculate the number of clusters as well as cluster centrosymmetry parameters
    Inputs:
        nsteps: the number of trajectory steps to consider when taking averages
        phi: the fraction of free DA in the system
        charge_frac: the fraction of charge on the polymer
        test: if this is True then do some printouts, hide the printouts if false
        trajnr: the trajectory number to read
    """
    
    #GET BASIC INFORMATION AND LOAD THE DATA
    n_DA = 500 #hard-code the total number of DA chromophores
    index_data = np.linspace(0,n_DA-1,n_DA,dtype='int') #data showing indices of DA molecules
    n_polymer = int(np.round((1-float(phi))*n_DA/10)) #number of polymer chains
    max_cs_val = 10 #maximum cs val to consider
    skip=5 #number of frames to skip
    trajfile=fr'C:\Users\nicho\Documents\Science\ferroelectricity project\scripts\phi_{phi}_{charge_frac}\traj_{trajnr}.lammpstrj'
    datafile=fr'C:\Users\nicho\Documents\Science\ferroelectricity project\scripts\lammps_inputs\Sys_{phi}_{charge_frac}.data'
    u = mda.Universe(datafile, trajfile, format='LAMMPSDUMP')
    centrosymmetry_vals=np.full((nsteps//skip,n_DA),-1.) #array to contain all of the different P values for each chromophore calculated in each timestep

    #LOOP THROUGH ALL TIMESTEPS REQUESTED
    for t in range(0,nsteps,skip): #perform analysis every 'skip' frames for now
        
        #PERFORM CLUSTERING PROCESS
        print(t)
        
        u.trajectory[t] #go to the appropriate trajectory frame
        #get resids for free DAs
        DA_indices=np.linspace(n_polymer,n_polymer+n_DA-1,n_DA,dtype=int)
        free_DAs=u.select_atoms('type 6')
        free_ids=free_DAs.resids-1-n_polymer
        #get resids for DAs which are attached to the polymer
        bound_ids=np.setdiff1d(DA_indices,free_ids)
        #some stuff
        DA_vecs=np.zeros((n_DA,3))
        #get vectors from acceptor to donor group
        for j in range(n_polymer,n_polymer+n_DA):
            sel1=u.select_atoms(f'resid {j+1} and (type 3)')
            sel2=u.select_atoms(f'resid {j+1} and (type 4)')
            DA_vecs[j-n_polymer]=compute_vec_periodic(sel1.positions[0],sel2.positions[0],u.dimensions[:3])
        DA_vecs_raw = np.copy(DA_vecs) #save a copy with the vectors pointing their original directions - we will use this to get the polarization

        #get cog data for DA molecules
        cog_data=np.zeros((n_DA,3)) #array containing center of gravity data of the DA molecules
        for j in range(n_polymer,n_polymer+n_DA):
            selection=u.select_atoms(f'resid {j+1} and (type 3 or type 4)')
            cog_data[j-n_polymer]=compute_cog_periodic(selection.positions,u.dimensions[:3])

        DA_data=np.zeros((n_DA,3)) #contains the theta and phi values corresponding to each DA vector np.arctan2(DA_vecs[i,1],DA_vecs[i,0])+np.pi
        for i in range(n_DA):
            DA_data[i] = np.array([np.arctan2(DA_vecs_raw[i,1],DA_vecs_raw[i,0]), np.arccos(np.abs(DA_vecs_raw[i,2])/np.linalg.norm(DA_vecs_raw[i])),0])
        #do cluster analysis on the theta and phi values
        box1 = freud.box.Box(np.pi,np.pi,0)
        system1 = freud.AABBQuery(box1, DA_data)
        cl1 = freud.cluster.Cluster()
        cl1.compute(system1, neighbors={"mode": 'ball', "r_max": 0.2})

        #do distance-based clustering within the angle-based clusters
        final_cluster_indices = np.zeros(n_DA)
        num_clusters_found = 0
        for i in range(cl1.num_clusters):
            box = freud.box.Box(u.dimensions[0],u.dimensions[1],u.dimensions[2])
            system = freud.AABBQuery(box, cog_data[cl1.cluster_idx==i])
            cl = freud.cluster.Cluster()
            cl.compute(system, neighbors={"mode": 'ball', "r_max": 3.5})
            cluster_indices = np.copy(cl.cluster_idx)
            cluster_indices += num_clusters_found
            final_cluster_indices[cl1.cluster_idx==i] = cluster_indices
            num_clusters_found += cl.num_clusters
        final_num_clusters = len(np.unique(final_cluster_indices))

        min_cluster_size = 25
        #calculate number of clusters which have at least 'min_cluster_size' members
        n_threshold_clusters = 0
        threshold_cluster_indices = np.zeros(100)
        for i in range(final_num_clusters):
            if len(cog_data[final_cluster_indices==i]) >= min_cluster_size:
                threshold_cluster_indices[n_threshold_clusters] = i
                n_threshold_clusters += 1
        threshold_cluster_indices = threshold_cluster_indices[:n_threshold_clusters]

        #CALCULATE CLUSTER PROPERTIES
        for i in range(n_threshold_clusters):
            indices = np.where(final_cluster_indices==threshold_cluster_indices[i])[0]
            positions = cog_data[final_cluster_indices==threshold_cluster_indices[i]]
            box_dims = np.array([u.dimensions[0],u.dimensions[1],u.dimensions[2]])
            #compute the gyration tensor and the eigenvector with smallest eigenvalue
            positions = center(positions,box_dims) #center the cluster
            gyration_t = gyration_tensor(positions,box_dims) #calculate gyration tensor of the cluster
            evals,evecs = np.linalg.eig(gyration_t)
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:,idx]
            transition_mat = np.transpose(evecs) #matrix to transform positions to orthonormal coordinates
            new_positions = np.dot(positions, evecs)

            #calculate centrosymmetry parameters for the new positions
            renorm_positions = np.copy(new_positions)
            renorm_positions[:,0] = renorm_positions[:,0] / 3.3 #normalize by the face-to-face distance
            renorm_positions[:,1] = renorm_positions[:,1] / 7.5 #normalize by the edge-to-edge distance
            renorm_positions[:,2] = 0
            dist_mat = calc_dist_mat(renorm_positions)
            centro_syms = np.zeros(n_DA)
            edge_vals = np.zeros(n_DA)
            edge_ids = np.zeros(n_DA) #array noting if our points are edges (0 is not edge, 1 is edge)
            interior_ids = np.zeros(n_DA,dtype=int) #array noting if our points are on the interior or not (contains indices of all points in interior of a given cluster)
            n_interior=0 #number of interior points in a given cluster
            for pos in range(len(positions)):
                sorted_distmat = np.argsort(dist_mat[pos])
                closest_atoms=sorted_distmat[1:5]
                # Compute the 8 nearest neighbor distances
                closest_neighbors=sorted_distmat[1:9]
                neighbor_distances=np.linalg.norm(renorm_positions[pos]-renorm_positions[closest_neighbors],axis=1)
                # Calculate the average distance to 8 neighbors for each particle
                avg_distance = np.mean(neighbor_distances)
                closest_vecs_real = new_positions[pos] - new_positions[closest_atoms]
                vec_sums_real = np.linalg.norm(closest_vecs_real[:, np.newaxis, :] + closest_vecs_real[np.newaxis, :, :],axis=2)
                if avg_distance < 0.4:
                    csval = np.sum(np.sort(vec_sums_real[np.triu_indices(vec_sums_real.shape[0], k=1)])[:2]**2)
                    if csval < max_cs_val:
                        centrosymmetry_vals[t//skip,indices[pos]] = csval

#     return centrosymmetry_vals[nsteps//(2*skip):,:]
    return centrosymmetry_vals
    
def calc_centrosymmetry_diffs(csvals1,csvals2):
    array_size=np.shape(csvals1)[0]
    n_DA=np.shape(csvals1)[1]
    diffs=np.zeros(n_DA*array_size)
    n_diffs=0
    for i in range(array_size):
        for j in range(n_DA):
            if csvals1[i,j]!=-1 and csvals2[i,j]!=-1:
                diffs[n_diffs]=csvals1[i,j]-csvals2[i,j]
                n_diffs += 1
    diffs=diffs[:n_diffs]
    return diffs