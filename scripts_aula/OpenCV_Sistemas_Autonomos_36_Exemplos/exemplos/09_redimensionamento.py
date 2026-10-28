"""Exemplo 09 — Redimensionar e preservar proporção

COMENTÁRIO GERAL
Aplica o resize exigido, 640×480, e compara com um letterbox que mantém a proporção da imagem. O item B usa o resize direto; letterbox é uma alternativa discutida.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/09_redimensionamento.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/09_redimensionamento/.

O QUE OBSERVAR
Observe a deformação de carros e compare com as margens do letterbox.
"""
import cv2
import numpy as np
from apoio import (iniciar, carregar_imagem, painel)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem);h,w=img.shape[:2]
direta=cv2.resize(img,(640,480),interpolation=cv2.INTER_AREA)
escala=min(640/w,480/h);nw,nh=round(w*escala),round(h*escala)
menor=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
letter=np.zeros((480,640,3),np.uint8);x,y=(640-nw)//2,(480-nh)//2
letter[y:y+nh,x:x+nw]=menor
print('Original:',(w,h),'Resize direto:',direta.shape,'Escala uniforme:',escala)
painel([direta,letter],['Resize direto 640 × 480','Proporção preservada'],out/'resize.png',a)
