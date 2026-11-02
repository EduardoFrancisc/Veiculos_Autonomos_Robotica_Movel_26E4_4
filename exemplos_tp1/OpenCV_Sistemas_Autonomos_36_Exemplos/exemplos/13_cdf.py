"""Exemplo 13 — Interpretar a distribuição acumulada

COMENTÁRIO GERAL
Normaliza o histograma e soma as frequências progressivamente. A CDF mostra a fração de pixels com intensidade menor ou igual a cada nível.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/13_cdf.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/13_cdf/.

O QUE OBSERVAR
A CDF final deve valer 1. Ela ajuda a explicar o mapeamento da equalização.
"""
import numpy as np
import matplotlib.pyplot as plt
from apoio import (iniciar, carregar_imagem, cinza, finalizar_figura)

a,out=iniciar(__file__)
y=cinza(carregar_imagem(a.imagem));h=np.bincount(y.ravel(),minlength=256)/y.size
cdf=h.cumsum();print('CDF em 64,128,255:',cdf[[64,128,255]].tolist())
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
axs[0].plot(h,color='black');axs[0].set(xlabel='Intensidade',ylabel='Frequência relativa')
axs[1].plot(cdf,color='black');axs[1].set(xlabel='Intensidade',ylabel='CDF',ylim=(0,1.02))
for ax in axs: ax.grid(alpha=.2)
finalizar_figura(fig,out/'cdf.png',a)
