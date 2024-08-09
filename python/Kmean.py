from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist #thu vien tinh khoang cach giua cac diem
np.random.seed(11)

means = [[2,2], [8,3], [3,6]] # cac toa do cua center ban dau dung de tao ra du lieu random 

cov = [[1,0], [0,1]]
N = 500 # so luong phan tu xung quan moi center
# tao cac diem xung quanh cac center
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)


#gop 3 cum
X = np.concatenate((X0, X1, X2), axis = 0)
K = 3
# tao mang gom N x cac so
label_array = np.asarray([0]*N + [1]*N + [2]*N).T

# hien thi du lieu
def kmeans_display(X, label):
	K = np.amax(label)+1
	X0 = X[label == 0, :]
	X1 = X[label == 1, :]
	X2 = X[label == 2, :]
#ve cac diem 
	plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
	plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
	plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

	plt.axis('equal')
	plt.plot()
	plt.show()
# ham in cac diem
kmeans_display(X, label_array)


#K-means clustering
def kmeans_init_centers(X, k):
	return X[np.random.choice(X.shape[0], k, replace = False)] # lay ra k diem du lieu bat ki trong cac phan tu cua X

#gan diem du lieu vao cum gan nhat
def kmeans_assign_label(X, centers):
	#tinh khoang cach tu diem trong x toi cac tam cum trong centers
	D = cdist(X, centers)
	#tim chi so nho nhat trong tung hang cua ma tran khoang cach D
	return np.argmin(D, axis = 1)

#cap nhan cac cum center
def kmeans_update_centers(X, labels, K):
	#khoi tao mang 0
	centers = np.zeros((K, X.shape[1]))
	for k in range(K):
		#chon cac diem du lieu thuoc cum k
		Xk = X[labels == k, :]
		#tinh trung binh diem trong cum de cap nhat tam
		centers[k, :] = np.mean(Xk, axis = 0)
	return centers
# kiem tra xem tam cum co con thay doi hay khong (mac dinh)
def has_converged(centers, new_centers):
	 return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))
# trien khai thuat toan
def kmeans(X,K):
	centers = [kmeans_init_centers(X, K)]
	labels = []
	it = 0
	while True:
		labels.append(kmeans_assign_label(X, centers[-1]))
		new_centers = kmeans_update_centers(X, labels[-1], K)
		if has_converged(centers[-1], new_centers):
			break
		centers.append(new_centers)
		it+=1
	return (centers, labels, it)
# hien thi
(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])
