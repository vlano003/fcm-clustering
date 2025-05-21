import numpy as np
import matplotlib.pyplot as plt

# Генерація даних
def generate_data(n=80, noise=0.05, rng=42):
    np.random.seed(rng)
    centers = np.array([[0,0],[1,1],[0,1]])
    pts, c = [], len(centers)
    per = n//c
    for m in centers:
        pts.append(m + noise*(np.random.rand(per,2)-0.5))
    X = np.vstack(pts)
    return X

# Ініціалізація U
def init_U(N,c):
    U = np.random.rand(N,c)
    return U/U.sum(axis=1,keepdims=True)

# Оновлення центрів та U

def update_centroids(X,U,m):
    um = U**m
    return (um.T@X) / um.sum(axis=0)[:,None]

def update_U(X,v,m):
    dist = np.linalg.norm(X[:,None]-v[None],axis=2)
    dist = np.fmax(dist,1e-10)
    exp = 2/(m-1)
    inv = dist**(-exp)
    return inv / inv.sum(axis=1,keepdims=True)

# FCM

def fuzzy_c_means(X,c=3,m=2,eps=1e-5,max_iter=100):
    N = X.shape[0]
    U = init_U(N,c)
    for _ in range(max_iter):
        v = update_centroids(X,U,m)
        U_new = update_U(X,v,m)
        if np.linalg.norm(U_new-U) < eps:
            break
        U = U_new
    return v, U

# Запуск і візуалізація
if __name__=='__main__':
    X = generate_data()
    v,U = fuzzy_c_means(X)
    labels = np.argmax(U,axis=1)
    for j in range(v.shape[0]):
        plt.scatter(X[labels==j,0],X[labels==j,1],label=f'C{j+1}')
    plt.scatter(v[:,0],v[:,1],marker='x',s=100)
    plt.legend(); plt.show()