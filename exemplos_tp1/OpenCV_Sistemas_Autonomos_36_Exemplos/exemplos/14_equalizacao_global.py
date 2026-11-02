"""Exemplo 14 — Comparar equalização global e original

COMENTÁRIO GERAL
Usa equalizeHist sobre a imagem uint8 de um canal. O mapeamento é calculado considerando todos os pixels, o que pode ser inadequado sob sombras locais.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/14_equalizacao_global.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/14_equalizacao_global/.

O QUE OBSERVAR
Inspecione separadamente regiões claras e escuras; contraste visual não comprova melhor detecção.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from apoio import (iniciar, carregar_imagem, cinza, finalizar_figura, painel)

a,out=iniciar(__file__)
y=cinza(carregar_imagem(a.imagem));eq=cv2.equalizeHist(y)
painel([y,eq],['Original em cinza','Equalização global'],out/'global.png',a)
fig,ax=plt.subplots(figsize=(8,4))
for im,label,ls in [(y,'Original','-'),(eq,'Global','--')]:
 ax.plot(np.bincount(im.ravel(),minlength=256)/im.size,color='black',label=label,linestyle=ls)
ax.set(xlabel='Intensidade',ylabel='Frequência relativa');ax.legend()
finalizar_figura(fig,out/'histogramas_global.png',a)
