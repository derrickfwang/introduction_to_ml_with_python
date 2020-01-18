#########################################################################################
#############   Sample Code in the Book: Introduction to ML with Python     #############
#########################################################################################

# Part 3: unsupervised learning #

plt.rcParams['image.cmap'] = "gray"  # set color of images to black and white. Or else 
                                     # images will have weird color

# 1. Scaling
# StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
scaler.transform(X_train)

# 2. Dimensionality reduction, feature extraction, manifold learning
# 2.1 PCA
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
ax = axes.ravel()  # ravel: get axes to a list

# 先scaler,再PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.transform(X_scaled)
X_pca[:, 0], X_pca[:, 1]

pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)


print("{0:25} {1:3}".format(name, count), end='   ')  # format first 25 space for variable 0

# 每打印三个换行
if (i + 1) % 3 == 0:
        print()
		
		
# 2.2 NMF: Non-Negative Matrix Factorization
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)


# 2.3 t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
# use fit_transform instead of fit, as TSNE has no transform method
digits_tsne = tsne.fit_transform(digits.data)

for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
			 


# 3. Clustering

# 3.1 KMeans
from sklearn.cluster import KMeans
# build the clustering model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
 
# 3.2 hierarchy clustering

from scipy.cluster.hierarchy import dendrogram, ward
linkage_array = ward(X)
dendrogram(linkage_array)

# 3.3 DBSCAN
from sklearn.cluster import DBSCAN

# 3.4 evaluating clustering algorithms: Adjusted Rand index
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score


























