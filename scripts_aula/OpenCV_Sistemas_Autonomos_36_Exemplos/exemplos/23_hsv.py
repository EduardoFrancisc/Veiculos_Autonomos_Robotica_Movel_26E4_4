"""Exemplo 23 — Interpretar H, S e V

COMENTÁRIO GERAL
Converte a imagem BGR uint8 para HSV e mostra H, S e V como planos. H é reescalado apenas para visualização; o processamento conserva a codificação original.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/23_hsv.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/23_hsv/.

O QUE OBSERVAR
Na conversão HSV padrão de 8 bits, H usa 0–179 e S e V usam 0–255. V não é luminância física.
"""
import cv2
import numpy as np
from apoio import (iniciar, carregar_imagem, painel)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem);hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h,s,v=cv2.split(hsv)
print('Faixas observadas H/S/V:',[(int(c.min()),int(c.max())) for c in [h,s,v]])
hv=np.rint(h.astype(np.float32)*255/179).astype(np.uint8)
painel([hv,s,v],['H reescalado só para exibição','S: saturação','V: valor'],out/'hsv.png',a)
