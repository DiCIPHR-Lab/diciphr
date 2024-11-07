import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt 
from sklearn.metrics import calinski_harabasz_score 

def distance_mat(dat):
    N = dat.shape[0]
    dist = np.ndarray(shape=(N,N))
    dist.fill(0)
    for ix in range(0,N):
        x = dat[ix,]
        if ix >0:
            for iy in range(0,ix):
                y = dat[iy,]
                dist[ix,iy] = np.nanmean((x - y)**2)
                dist[iy,ix] = dist[ix,iy]
    return dist
    
def hierarchical_clustering(dataframe, method='ward'):
    subjects = dataframe.index
    v = StandardScaler().fit_transform(dataframe.values)
    X = distance_mat(v)
    Xc = squareform(X)
    dist_df = pd.DataFrame(X, index=subjects, columns=subjects)
    Z = linkage(Xc, method=method)  # try single 
    cluster_df = pd.DataFrame()
    for i in range(1,len(subjects)+1):        
        clusters = fcluster(Z, i, criterion='maxclust')
        cluster_df[i] = clusters
        c = calinski_harabasz_score(v, clusters)
        print(d, c)
        # Define clusters 
        cohort.loc[subjects,'Cluster'] = ['C{0}'.format(int(i)) for i in cluster_df.loc[subjects,clid]]
     #   cohort.to_csv('cohort_{}.csv'.format(clid))

