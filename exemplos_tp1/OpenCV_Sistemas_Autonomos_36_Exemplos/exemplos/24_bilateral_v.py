"""Exemplo 24 — Filtrar V preservando transições

COMENTÁRIO GERAL
Redimensiona para 640×480, converte para HSV, extrai V e aplica bilateral com d=5 e sigmas 50. Mostra a diferença absoluta amplificada entre V e o filtrado.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/24_bilateral_v.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/24_bilateral_v/.

O QUE OBSERVAR
O bilateral combina proximidade espacial e semelhança de intensidade; parâmetros excessivos podem remover detalhes.
"""
import cv2
import numpy as np
from apoio import (iniciar, carregar_imagem, painel)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem)
menor=cv2.resize(img,(640,480),interpolation=cv2.INTER_AREA)
hsv=cv2.cvtColor(menor,cv2.COLOR_BGR2HSV);v=hsv[:,:,2]
filtrado=cv2.bilateralFilter(v,a.d,a.sigma_cor,a.sigma_espaco)
dif=cv2.absdiff(v,filtrado);ampliada=np.clip(dif.astype(np.int16)*8,0,255).astype(np.uint8)
print('Diferença absoluta média:',float(dif.mean()))
painel([v,filtrado,ampliada],['V original','V após bilateral','Diferença × 8'],out/'bilateral.png',a)
