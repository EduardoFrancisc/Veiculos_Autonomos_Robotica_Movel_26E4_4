"""Exemplo 03 — Entender pixel, linha, coluna e recorte

COMENTÁRIO GERAL
Lê um pixel central e extrai uma região da metade inferior. O acesso NumPy usa [linha, coluna], enquanto o tamanho em resize usa (largura, altura).

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/03_pixels_coordenadas.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/03_pixels_coordenadas/.

O QUE OBSERVAR
Localize visualmente o recorte; a faixa inferior costuma concentrar o pavimento.
"""
import cv2
from apoio import (iniciar, carregar_imagem, painel)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem);h,w=img.shape[:2]
print('Centro [y,x] e valores BGR:',(h//2,w//2),img[h//2,w//2].tolist())
roi=img[h//2:h,w//4:3*w//4].copy()
marcada=img.copy()
cv2.rectangle(marcada,(w//4,h//2),(3*w//4-1,h-1),(255,255,255),2)
painel([marcada,roi],['Região escolhida','Recorte da pista'],out/'recorte.png',a)
