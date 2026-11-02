"""Exemplo 16 — Equalizar o canal L de Lab

COMENTÁRIO GERAL
Converte BGR para Lab, modifica apenas L por CLAHE, preserva a e b e converte de volta. Os histogramas comparam o mesmo L antes e após o algoritmo.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/16_clahe_colorido.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/16_clahe_colorido/.

O QUE OBSERVAR
Não equalize separadamente B, G e R: isso pode alterar a relação entre cores.
"""
import cv2
from apoio import (iniciar, carregar_imagem, salvar_imagem, painel, histograma_comparado)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB);l,aa,bb=cv2.split(lab)
leq=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
eq=cv2.cvtColor(cv2.merge([leq,aa,bb]),cv2.COLOR_LAB2BGR)
painel([img,eq],['Original colorida','CLAHE apenas no canal L'],out/'clahe_lab.png',a)
histograma_comparado(l,leq,out/'histogramas_l.png',a,canal='L codificado em uint8')
salvar_imagem(out/'colorida_equalizada.png',eq)
