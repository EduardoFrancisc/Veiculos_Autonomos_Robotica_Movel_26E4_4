"""Exemplo 05 — Separar os canais B, G e R

COMENTÁRIO GERAL
Usa cv2.split para obter três planos de intensidade. Cada plano tem uma matriz H×W e não é, sozinho, uma fotografia colorida.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/05_canais_bgr.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/05_canais_bgr/.

O QUE OBSERVAR
Compare a resposta da vegetação, do céu e das marcações nos três canais.
"""
import cv2
from apoio import (iniciar, carregar_imagem, painel)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem);b,g,r=cv2.split(img)
print('Médias B, G e R:',float(b.mean()),float(g.mean()),float(r.mean()))
painel([b,g,r],['B: intensidade de azul','G: intensidade de verde','R: intensidade de vermelho'],out/'canais.png',a)
