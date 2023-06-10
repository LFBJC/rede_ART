# REDE ART - IMPLEMENTADA COMO UMA FORMA DE FIXAR O CONHECIMENTO
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# INICIALIZAÇÃO DOS PARÂMETROS
a, b, c, d = 10, 10, 0.1, 0.9 # VALORES REAIS FINITOS
e = 1e-7 # VALOR REAL PEQUENO PARA EVITAR DIVISÃO POR 0
theta = 0.126 # LIMIAR PARA ATIVAÇÃO
lr = 0.8 # LEARNING RATE
l = 0.8 # LIMIAR DE VIGILÂNCIA
n_epochs = 30000
n_iter = 300
# A1, B1, A2, B2
entradas = [
    np.array([
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 1]
    ]),
    np.array([
        [1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 0],
    ]),
    np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0]
    ]),
    np.array([
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 0],
    ]),
]
# for entrada in entradas:
#     plt.imshow(entrada)
#     plt.show()
M = -2*np.random.rand(2, 9, 7) + 1
T = -2*np.random.rand(2, 9, 7) + 1


def unitary(input):
    return input/(e+np.linalg.norm(input.flatten()))


for epoch in tqdm(range(n_epochs)):
    for entrada in entradas:
        # ATUALIZE AS ATIVAÇÕES NAS UNIDADES DAS CAMADAS DE ENTRADA (DENOMINADAS F1)
        u, p, q = np.zeros_like(entrada), np.zeros_like(entrada), np.zeros_like(entrada)
        w = entrada
        x = unitary(w)
        v = np.clip(x, 0, np.max(x))
        # ATUALIZE AS ATIVAÇÕES OUTRA VEZ
        u = unitary(v)
        p = u
        q = unitary(p)
        w = entrada + a*u
        x = unitary(w)
        v = np.clip(x, 0, np.max(x)) + b*np.clip(q, 0, np.max(q))
        activation = np.sum(np.sum(np.multiply(M,p), axis=2), axis=1)
        y = 1/(1+np.exp(-activation))
        reset = True
        while(reset):
            winner = np.argmax(y)
            u = unitary(v)
            p = u + d*T[winner]
            r = unitary(u + c*p)
            reset = np.linalg.norm(r.flatten()) < l - e
            if not reset:
                w = entrada + a * u
                x = unitary(w)
                q = unitary(p)
                v = np.clip(x, 0, np.max(x)) + b * np.clip(q, 0, np.max(q))
        for step in range(n_iter):
            T[winner] = lr*d*u + (1 + lr*d*(d-1))*T[winner]
            M[winner] = lr * d * u + (1 + lr * d * (d - 1)) * M[winner]
        u = unitary(v)
        w = entrada + a * u
        p = u + d * T[winner]
        x = unitary(w)
        q = unitary(p)
        v = np.clip(x, 0, np.max(x)) + b * np.clip(q, 0, np.max(q))
plt.imshow(T[0])
plt.show()
plt.imshow(T[1])
plt.show()
