import io
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, send_file

# 1. Генерація даних

def generate_data(n=80, noise=0.05, rng=42):
    np.random.seed(rng)
    centers = np.array([[0,0],[1,1],[0,1]])
    X = np.vstack([c + noise*(np.random.rand(n//3,2)-0.5) for c in centers])
    return X

# 2. Ініціалізація матриці нечітких належностей

def init_U(N, c):
    U = np.random.rand(N, c)
    return U / U.sum(axis=1, keepdims=True)

# 3. Оновлення центрів

def update_centroids(X, U, m):
    um = U ** m
    return (um.T @ X) / um.sum(axis=0)[:, None]

# 4. Оновлення належностей

def update_U(X, V, m):
    D = np.linalg.norm(X[:, None] - V[None], axis=2)
    D = np.maximum(D, 1e-8)
    inv = D ** (-2/(m-1))
    return inv / inv.sum(axis=1, keepdims=True)

# 5. Алгоритм Fuzzy C-Means

def fuzzy_c_means(X, c=3, m=2.0, eps=1e-5, max_iter=100):
    N = X.shape[0]
    U = init_U(N, c)
    for _ in range(max_iter):
        V = update_centroids(X, U, m)
        U_new = update_U(X, V, m)
        if np.linalg.norm(U_new - U) < eps:
            break
        U = U_new
    return V, U

# 6. Flask веб-інтерфейс

app = Flask(__name__)

@app.route('/')
def index():
    # Генеруємо дані та запускаємо FCM
    X = generate_data()
    V, U = fuzzy_c_means(X)
    labels = np.argmax(U, axis=1)

    # Створюємо графік
    fig, ax = plt.subplots()
    for j in range(V.shape[0]):
        ax.scatter(X[labels == j, 0], X[labels == j, 1], label=f'Cluster {j+1}')
    ax.scatter(V[:, 0], V[:, 1], marker='x', s=100, color='k', label='Centroids')
    ax.set_title('Fuzzy C-Means Clustering')
    ax.set_xlabel('X1'); ax.set_ylabel('X2')
    ax.legend()

    # Зберігаємо у буфер та відправляємо PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    # Запускаємо сервер на 8080
    app.run(host='0.0.0.0', port=8080)