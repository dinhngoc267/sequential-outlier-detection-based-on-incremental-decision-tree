import numpy as np
import math 
import sys
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, show, imshow, pause
import matplotlib
import pylab
import pdb
import itertools

# assume x have 2 features: [x1,x2]
N = 1 # number of subspace
sum_points_left = np.zeros((1,2),dtype=np.float64)  # sum element-wise of x in subspace left
sum_points_right = np.zeros((1,2),dtype=np.float64) # sum element-wise of x in subspace right

num_point_subspace = np.zeros((1,2),dtype=np.int32) # save number of point in subspace left and right: left index 0 , right index 1

is_have_child = [0] # bool array indicate that whether subspace i th is leaf
hyperplane = {} # save tuple: (a,b) with a as numpy array(1,2), b as scalar <a,x> = b. Save a and b
parent_node_indices = [] # save the index of parent node. 
parent_node_indices.append([])
#idx_parents_of_node = [[0]]
direct_of_node = [0]# direct of node, 0 is root, 1 is left of parent, 2 is right of parent. 
centroids_of_childs = [(0,0)]
level_of_node = {}
level_of_node[0] = 0
max_level = 0
child_node_indices = {}
# generate random dataset
m = 1000  # number of examples
np.random.seed(10)
x1_normal = np.random.multivariate_normal([-1, 1], [[0.2, 0.], [0., 0.2]], int(m * 0.3))
x2_normal = np.random.multivariate_normal([1, -1], [[0.14, 0.2], [0.2, 0.4]], int(m * 0.3))
x3_normal = np.random.multivariate_normal([2, 2], [[0.4, -0.2], [-0.2, 0.14]], int(m * 0.3))

x_normal = np.vstack((x1_normal, x2_normal, x3_normal))
x_anomaly = np.random.multivariate_normal([1, 1], [[0.1, 0], [0., 0.1]], int(m * 0.1))

x = np.vstack((x_normal, x_anomaly))
#y = np.vstack((-np.ones((m * 0.9, 1)), np.ones((m * 0.1, 1))))  # -1 for normal, 1 for anomaly
# suffle dataset to feed one by one
T = m
mix_ids = np.random.permutation(T)
indices = np.arange(0,len(x))
np.random.shuffle(indices)
#level_of_node chuyển sang dict
#x, y = x[mix_ids], y[mix_ids]
#x = x[mix_ids]
dataset = np.array([[-1.3, 0.3], [-0.9, 0.9],[0.98,-1.4], [0.7,-1.2], [-0.5,0.3], [1.9,2.1],[2.1,2.1],[2.1,2.2],[0.97,0],[1.8,1.78],[1.3,2.1]])
x = x[indices]
x = x[11:]
#print(dataset.shape)
#print(x.shape)
dataset = np.append(dataset,x,axis=0)
#dataset = x
h_line = {}
k = 0
lines = []
print(dataset.shape)
pylab.ion()
p1  = None
p2 = None
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_ylim(-3, 3) 
ax.set_xlim(-2,4)
l = []
flag = {}
height = 0
draw_line = {}
# u = ax.scatter(x_normal[:,0], x_normal[:,1], marker='x', color='blue', label='Normal')
# v = ax.scatter(x_anomaly[:,0], x_anomaly[:,1], marker = 'x', color ='red', label='Anormal')
# ax.legend()


plt.grid(linestyle='--', alpha=0.5)
#plt.show()
#pause(1000)
for t, x in enumerate(dataset):

    ax.scatter(x[0], x[1], marker='x', color='black')
    if t < 12:
        plt.pause(1)
    elif t < 40:
        plt.pause(0.01)
    elif t < 150: 
        plt.pause(0.001)
    draw()
    is_belonged_to_nodes = [0]
    for n in range(0, N):
        n_left = num_point_subspace[n][0] 
        n_right = num_point_subspace[n][1]

        if n_left == 0:          
            n_left = 1
        if n_right == 0:
            n_right = 1

        centroid_left = sum_points_left[n]/n_left
        centroid_right = sum_points_right[n]/n_right

        dis_to_centroid_left = math.sqrt(np.dot((centroid_left-x).transpose(),(centroid_left-x)))
        dis_to_centroid_right = math.sqrt(np.dot((centroid_right-x).transpose(),(centroid_right-x)))

        if n == 0:             
            if dis_to_centroid_left <= dis_to_centroid_right:
                sum_points_left[n] = sum_points_left[n] + x
                num_point_subspace[n][0] += 1
            else:
                sum_points_right[n] = sum_points_right[n] +x
                num_point_subspace[n][1] += 1
                
        else:
            if np.array_equal(np.array(is_belonged_to_nodes),np.array(parent_node_indices[n])): # check x has to belongs to fathers of n. 
                
                nearest_parent_idx = parent_node_indices[n][-1] # get hyperplane of father
                coefficents_hyperplane = hyperplane[nearest_parent_idx][0]
                threshold = hyperplane[nearest_parent_idx][1]

                if (np.dot(coefficents_hyperplane.transpose(), x) - threshold >= 0 and direct_of_node[n] == 1) or (np.dot(coefficents_hyperplane.transpose(), x) - threshold < 0 and direct_of_node[n] == 2):
                    
                    is_belonged_to_nodes.append(n)

                    if dis_to_centroid_left <= dis_to_centroid_right:
                        sum_points_left[n] = sum_points_left[n] + x
                        num_point_subspace[n][0] += 1                        
                    else:
                        sum_points_right[n] = sum_points_right[n] +x
                        num_point_subspace[n][1] += 1
    
    # Plitting is performed
    if (t+1) in [5,11,40, 128, 256,512,800]:
        if(p1):
            p1.remove()
            p2.remove()
        draw()
        idx_max_ratio = 0
        max_ratio = sys.float_info.min
        #print('splitting times...')
        #print('number of nodes: ', N)
        
        centroid_left_of_chosen_node = None
        centroid_right_of_chosen_node = None


        # for item in l:
        #     ax.lines.remove(item[0])
        #     l.remove(item)
       
        for n in range(0,N):
            #if is_have_child[n] == 0:
                n_left = num_point_subspace[n][0] 
                n_right = num_point_subspace[n][1]

                if n_left == 0:
                    n_left = 1
                if n_right == 0:
                    n_right = 1

                centroid_left = sum_points_left[n]/n_left
                #print('centroid left of node {}: {}'.format(n, centroid_left))
                centroid_right = sum_points_right[n]/n_right
                #print('centroid right of node {}: {}'.format(n, centroid_right))

                dis = math.sqrt(np.dot((centroid_left-centroid_right).transpose(), (centroid_left-centroid_right)))
                ratio = dis / (2**level_of_node[n])

                #print('ratio of {} is: {}'.format(n, ratio))
                if ratio > max_ratio:
                    max_ratio = ratio
                    idx_max_ratio = n 
                    centroid_left_of_chosen_node = centroid_left
                    centroid_right_of_chosen_node = centroid_right
                

        p1 = plt.scatter(centroid_left_of_chosen_node[0],centroid_left_of_chosen_node[1], color = 'blue')
        p2 = plt.scatter(centroid_right_of_chosen_node[0],centroid_right_of_chosen_node[1], color = 'red')
        #plt.plot([centroid_left_of_chosen_node[0], centroid_right_of_chosen_node[0]], [centroid_left_of_chosen_node[1], centroid_right_of_chosen_node[1]], color='black')
        plt.pause(1)
        draw()
        
        D = centroid_left_of_chosen_node - centroid_right_of_chosen_node
        coefficents = D/math.sqrt(np.dot(D.transpose(), D))
        threshold = np.dot(coefficents.transpose(), (centroid_left_of_chosen_node+centroid_right_of_chosen_node)/2)
        x1 = None
        yhat = None
        if coefficents[1] != 0:
            point = np.array([])
            intersection = []
            # Find idx of parent node:
            parent_idxs = parent_node_indices[idx_max_ratio] 

            
            if len(parent_idxs) > 0: 

                for parent_idx in parent_idxs:

                    if parent_idx in hyperplane.keys():
                        parent_coefficent = hyperplane[parent_idx][0]
                        parent_threshold = hyperplane[parent_idx][1]      
                        x1 = np.arange(-2, 4, 0.00001)
                        
                        yparent = (parent_threshold - x1*parent_coefficent[0])/parent_coefficent[1]      
                        yhat = (threshold - x1*coefficents[0])/coefficents[1]

                        idx = np.argwhere(np.diff(np.sign(yhat - yparent))).flatten()

                        if len(idx) > 0: 
                            intersection.append([x1[idx], (threshold-x1[idx]*coefficents[0])/coefficents[1]])
                        #point = np.append(point, x1[idx])

                if (direct_of_node[idx_max_ratio] == 1 and len(parent_idxs) == 1) or (len(parent_idxs) >= 2 and parent_idxs[1]==1): 
                    if len(intersection) <= 2:
                        point = np.append(point, np.array([-2]))
                        point = np.append(point, intersection[0][0])
                        #point = np.array([-2,x1[idx]])
                        
                elif (direct_of_node[idx_max_ratio] == 2 and len(parent_idxs) == 1) or ((len(parent_idxs) >= 2 and parent_idxs[1]==2)) : 
                    if len(intersection) <= 2:
                        point = np.append(point, np.array([4]))
                        point = np.append(point, intersection[0][0])
                        #x = np.array([x1[idx], 4])

                if len(intersection) > 2:

                    for k,p in enumerate(intersection):
                        for i, parent_idx in enumerate(parent_idxs):
                            parent_coefficent = hyperplane[parent_idx][0]
                            parent_threshold = hyperplane[parent_idx][1]   

                            is_above = True

                            if i + 1 < len(parent_idxs):    
                                if direct_of_node[parent_idxs[i+1]] == 2:
                                    is_above = False
                            else: 
                                if direct_of_node[idx_max_ratio] == 2:
                                    is_above = False        
                            
                            m = p[0]*parent_coefficent[0] + p[1]*parent_coefficent[0] - parent_threshold 
                            #y_p = (parent_threshold - p*parent_coefficent[0])/parent_coefficent[1]
                            
                            if (m > 0 and is_above == False) or (m < 0 and is_above == True): 
                                intersection.remove(p)
                                k -= 1
                                break
                    
                    point = np.array([intersection[0][0], intersection[1][0]])
                    # min_dis = sys.float_info.max
                    # min_pair = None

                    # for pair in itertools.combinations(point,2):
                    #     dis_pair = abs(pair[0] - pair[1])

                    #     if dis_pair < min_dis:
                    #         min_dis = dis_pair
                    #         min_pair = pair
                    
                    # point = np.array([min_pair[0],min_pair[1]])

                yhat = (threshold - point*coefficents[0])/coefficents[1]

                if idx_max_ratio in draw_line.keys():
                    draw_line[idx_max_ratio][0].remove()
                draw_line[idx_max_ratio] = ax.plot(point, yhat,color='g')

            
            else:
                point = np.array([-2,4])
                yhat = (threshold - point*coefficents[0])/coefficents[1]
                
                if idx_max_ratio in draw_line.keys():
                    draw_line[idx_max_ratio][0].remove()
                draw_line[idx_max_ratio] = ax.plot(point, yhat, color='g')
            #pause(1000)
            plt.pause(1)
            draw()            
        # Create 2 new subspaces
        if is_have_child[idx_max_ratio] == 0: # if the plitted node is leaf
            
            print('choose node {} to split to node {},{}'.format(idx_max_ratio, N, N+1))

            hyperplane[idx_max_ratio] = (coefficents,threshold)
            is_have_child[idx_max_ratio] = 1
            
            # su lai dung dict
            child_node_indices[idx_max_ratio] = [N, N+1]
            N += 2
            # Add sum point array
            sum_points_left = np.append(sum_points_left,np.zeros((2,2),dtype=np.float64),axis=0)
            sum_points_right = np.append(sum_points_right,np.zeros((2,2),dtype=np.float64),axis=0)
            
            #print('asigned centroid of splitted node to centroids of child of its')
            #print('centroid left: ', centroid_left_of_chosen_node)
            #print('centroid right: ', centroid_right_of_chosen_node)

            # centroid of left will be centroid of left and right of child left.
            sum_points_left[-2] = copy.deepcopy(centroid_left_of_chosen_node)
            sum_points_right[-2] = copy.deepcopy(centroid_left_of_chosen_node)
            # centroid of right will be centroid of left and right of child right.  
            sum_points_left[-1] = copy.deepcopy(centroid_right_of_chosen_node)
            sum_points_right[-1] = copy.deepcopy(centroid_right_of_chosen_node)
            
            num_point_subspace = np.append(num_point_subspace, np.ones((2,2),dtype=np.int32),axis=0)
            
            tmp = np.zeros((1,2),dtype=np.float64).flatten()
  
            if np.array_equal(sum_points_left[-2], tmp):
                num_point_subspace[-2] = np.zeros((1,2))
            if np.array_equal(sum_points_left[-1], tmp):
                num_point_subspace[-1] = np.zeros((1,2))

            is_have_child.extend([0,0])


            parent_nodes = copy.deepcopy(parent_node_indices[idx_max_ratio])
            parent_nodes.append(idx_max_ratio)
            
            parent_node_indices.append(parent_nodes)
            parent_node_indices.append(parent_nodes)

            direct_of_node.extend([1,2])  
            
            level_of_node[N-1] = 0
            level_of_node[N-2] = 0
            
            # parent_idxs = parent_node_indices[N-1]
            # current_height = parent_idxs[0]
            
            
            for parent_idx in parent_node_indices[N-1]:
                level_of_node[parent_idx] += 1

            # size = len(level_of_node)
            # if idx_max_ratio == 0 and len(level_of_node) == 1: # If root node is splitted the first time
            #     level_of_node[1] = 1
            #     level_of_node[2] = 1 
            # else:
            #     level_of_node[size] = level_of_node[idx_max_ratio] + 1
            #     level_of_node[size + 1] = level_of_node[idx_max_ratio] + 1

        else:

            print('choose node {} to split again'.format(idx_max_ratio))

            hyperplane[idx_max_ratio] = (coefficents, threshold)      

            # centroid of left will be centroid of left and right of child left.
            sum_points_left[child_node_indices[idx_max_ratio][0]] = copy.deepcopy(centroid_left_of_chosen_node)
            sum_points_right[child_node_indices[idx_max_ratio][0]] = copy.deepcopy(centroid_left_of_chosen_node)
            # centroid of right will be centroid of left and right of child right.  
            sum_points_left[child_node_indices[idx_max_ratio][1]] = copy.deepcopy(centroid_right_of_chosen_node)
            sum_points_right[child_node_indices[idx_max_ratio][1]] = copy.deepcopy(centroid_right_of_chosen_node)
            
            num_point_subspace[child_node_indices[idx_max_ratio][0]] = np.ones((1,2))
            num_point_subspace[child_node_indices[idx_max_ratio][1]] = np.ones((1,2))
            tmp = np.zeros((1,2),dtype=np.float64).flatten()

            if np.array_equal(sum_points_left[child_node_indices[idx_max_ratio][0]], tmp):
                num_point_subspace[child_node_indices[idx_max_ratio][0]] = np.zeros((1,2))
            if np.array_equal(sum_points_left[child_node_indices[idx_max_ratio][1]], tmp):
                num_point_subspace[child_node_indices[idx_max_ratio][1]] = np.zeros((1,2))

            for parent_idx in parent_node_indices[idx_max_ratio]:
                level_of_node[parent_idx] += 1
            level_of_node[idx_max_ratio] += 1

        k += 1 # increase number of splitting times
    
    # Plot 
    #if t in [10,50,200,400,600,800]:
   
pause(1000)

    #for key,value in hyperplane.items():
# centroid_left = sum_points_left[0]/ num_point_subspace[0][0]
# centroid_right = sum_points_right[0]/ num_point_subspace[0][1]
# value = hyperplane[0]
# plt.scatter(centroid_left[0], centroid_left[1], color='r')
# plt.scatter(centroid_right[0], centroid_right[1], color='g')
# coefficents = value[0]
# threshold = value[1]
# if coefficents[1] != 0:
#     x1 = np.array([[-2],[4]])
#     yhat = (threshold - x1*coefficents[0])/coefficents[1]
# else:
#     yhat = threshold - x1
# plt.plot(x1,yhat)
# plt.show()
    # for idx,item in hyperplane.items(): 
    #     count+=1
    #     coefficents = item[0]
    #     threshold = item[1]

    #     x1 = np.append(x1, np.array([0]),axis=0)
    #     yhat = (threshold - x1*coefficents[0])/coefficents[1]
                
    #     plt.plot(x1,yhat)
    #     draw()  
    #     matplotlib.pyplot.pause(1)  

    # if len(lines) >= count:
    #     line = lines[idx]
    #     line = line.pop(0)

            
    #     draw()
    #     new_line = plt.plot(x1,yhat)
        
    #     matplotlib.pyplot.pause(1)   
    #     lines[idx] = new_line 
    #     line.remove()
    #     draw()
    # else:
    #     tmp = plt.plot(x1,yhat)
    #     lines.append(tmp)

    #     matplotlib.pyplot.pause(1)    




        


        



        
    

    
                            
            




