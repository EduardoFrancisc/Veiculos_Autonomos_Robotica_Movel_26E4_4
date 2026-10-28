"""Exemplo 32 — Relacionar contraste e bordas da pista

COMENTÁRIO GERAL
Aplica Canny antes e após CLAHE e uma máscara trapezoidal de região de interesse. As bordas são uma etapa ilustrativa para localização de faixas, sem medir acurácia.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/32_bordas_faixas.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/32_bordas_faixas/.

O QUE OBSERVAR
Um aumento na quantidade de bordas pode representar faixas ou ruído. Não equivale a uma melhora demonstrada na detecção.
"""
import cv2
import numpy as np
from apoio import (iniciar, carregar_imagem, cinza, painel, RAIZ)

a,out=iniciar(__file__)
# Para este exemplo de faixas, a entrada padrão é um frame do vídeo de pista.
# Uma imagem fornecida explicitamente com --imagem tem prioridade.
caminho=RAIZ/'dados/pista_udacity.png' if a.imagem==RAIZ/'dados/frontal_kitti.png' else a.imagem
y=cinza(cv2.resize(carregar_imagem(caminho),(640,480)))
eq=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(y)
mask=np.zeros_like(y);pontos=np.array([[(30,479),(260,270),(380,270),(610,479)]],np.int32)
cv2.fillPoly(mask,pontos,255)
e0=cv2.bitwise_and(cv2.Canny(y,70,140),mask)
e1=cv2.bitwise_and(cv2.Canny(eq,70,140),mask)
print('Pixels de borda na ROI antes/depois:',int((e0>0).sum()),int((e1>0).sum()))
painel([y,eq,e0,e1],['Cinza','CLAHE','Bordas na ROI: original','Bordas na ROI: CLAHE'],out/'bordas.png',a,colunas=2)
