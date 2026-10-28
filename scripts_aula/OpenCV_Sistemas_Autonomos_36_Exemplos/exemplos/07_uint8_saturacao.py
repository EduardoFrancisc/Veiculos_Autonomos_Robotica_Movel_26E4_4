"""Exemplo 07 — Evitar estouro na aritmética uint8

COMENTÁRIO GERAL
Soma 60 à intensidade usando NumPy uint8, que pode retornar ao início da faixa, e compara com saturação explícita. A conversão para int16 permite somar antes do clip.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/07_uint8_saturacao.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/07_uint8_saturacao/.

O QUE OBSERVAR
Verifique 240+60: o resultado esperado para aumentar brilho com saturação é 255.
"""
import numpy as np
from apoio import (iniciar, carregar_imagem, painel)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem)
incorreta=img+np.uint8(60)
correta=np.clip(img.astype(np.int16)+60,0,255).astype(np.uint8)
print('240+60 em uint8:',(np.array([240],np.uint8)+np.uint8(60)).tolist())
print('240+60 com clip:',np.clip(np.array([240],np.int16)+60,0,255).tolist())
painel([img,incorreta,correta],['Original','Soma com retorno em uint8','Soma com saturação'],out/'uint8.png',a)
