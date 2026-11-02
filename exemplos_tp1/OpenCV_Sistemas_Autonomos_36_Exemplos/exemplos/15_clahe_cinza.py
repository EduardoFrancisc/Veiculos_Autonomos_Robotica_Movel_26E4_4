"""Exemplo 15 — Aplicar CLAHE à imagem em cinza

COMENTÁRIO GERAL
Cria CLAHE com limite de contraste 2 e grade de 8×8 regiões. Compara imagem e histogramas antes e depois usando o mesmo canal e 256 níveis.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/15_clahe_cinza.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/15_clahe_cinza/.

O QUE OBSERVAR
tileGridSize=(8,8) significa oito regiões em cada eixo, e não regiões de oito pixels.
"""
import cv2
from apoio import (iniciar, carregar_imagem, cinza, salvar_imagem, painel, histograma_comparado)

a,out=iniciar(__file__)
y=cinza(carregar_imagem(a.imagem))
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
eq=clahe.apply(y)
# Variações de iluminação mudam o contraste entre faixas e asfalto e entre
# obstáculos e fundo. CLAHE pode recuperar contraste local, mas amplificar
# ruído; a influência na percepção deve ser verificada, não presumida.
painel([y,eq],['Original em cinza','CLAHE: clip 2 / grade 8 × 8'],out/'clahe.png',a)
histograma_comparado(y,eq,out/'histogramas.png',a)
salvar_imagem(out/'equalizada.png',eq)
