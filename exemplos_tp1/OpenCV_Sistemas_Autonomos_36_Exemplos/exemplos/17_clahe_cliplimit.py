"""Exemplo 17 — Investigar o limite de contraste do CLAHE

COMENTÁRIO GERAL
Mantém a grade fixa e varia apenas clipLimit entre 1, 2, 4 e 8. Isso ajuda a perceber contraste e amplificação de pequenas variações sem misturar dois parâmetros.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/17_clahe_cliplimit.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/17_clahe_cliplimit/.

O QUE OBSERVAR
Mais contraste não é automaticamente melhor. Observe texturas e possíveis artefatos.
"""
import cv2
from apoio import (iniciar, carregar_imagem, cinza, painel)

a,out=iniciar(__file__)
y=cinza(carregar_imagem(a.imagem));imagens=[y];titulos=['Original']
for clip in [1.,2.,4.,8.]:
 eq=cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8)).apply(y)
 imagens.append(eq);titulos.append(f'clipLimit = {clip:g}')
 print('clipLimit',clip,'média',float(eq.mean()),'desvio',float(eq.std()))
painel(imagens,titulos,out/'varrer_clip.png',a)
