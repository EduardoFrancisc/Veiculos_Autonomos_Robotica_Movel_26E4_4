"""Exemplo 18 — Investigar a grade do CLAHE

COMENTÁRIO GERAL
Mantém clipLimit=2 e varia a quantidade de regiões: 2×2, 4×4, 8×8 e 16×16. Para uma imagem de largura W, o tamanho aproximado de cada região é W/nx.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/18_clahe_grade.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/18_clahe_grade/.

O QUE OBSERVAR
Uma grade mais densa reage mais localmente e pode reforçar ruído ou textura.
"""
import cv2
from apoio import (iniciar, carregar_imagem, cinza, painel)

a,out=iniciar(__file__)
y=cinza(carregar_imagem(a.imagem));imagens=[y];titulos=['Original'];h,w=y.shape
for n in [2,4,8,16]:
 eq=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(n,n)).apply(y)
 imagens.append(eq);titulos.append(f'Grade {n} × {n}')
 print(f'{n} × {n}: região aproximadamente {w/n:.1f} × {h/n:.1f} pixels')
painel(imagens,titulos,out/'varrer_grade.png',a)
