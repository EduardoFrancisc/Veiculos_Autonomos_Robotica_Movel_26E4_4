"""Exemplo 19 — Simular uma sombra na pista

COMENTÁRIO GERAL
Escurece uma faixa da imagem com multiplicação em float e compara equalização global e CLAHE. A sombra é uma transformação artificial controlada de uma fotografia real.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/19_iluminacao_sombra.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/19_iluminacao_sombra/.

O QUE OBSERVAR
Não interprete esta simulação como uma medição de câmera sob iluminação real.
"""
import cv2
import numpy as np
from apoio import (iniciar, carregar_imagem, cinza, painel)

a,out=iniciar(__file__)
y=cinza(carregar_imagem(a.imagem));h,w=y.shape
fator=np.ones_like(y,dtype=np.float32);fator[:,w//3:2*w//3]=.30
sombra=np.clip(y.astype(np.float32)*fator,0,255).astype(np.uint8)
glob=cv2.equalizeHist(sombra)
eq=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(sombra)
painel([y,sombra,glob,eq],['Foto original','Sombra artificial × 0,30','Equalização global','CLAHE'],out/'sombra.png',a,colunas=2)
print('Simulação de iluminação; nenhuma informação perdida por saturação é recuperada.')
