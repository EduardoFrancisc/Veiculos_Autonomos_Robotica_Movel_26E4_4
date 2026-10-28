"""Exemplo 08 — Normalizar e voltar a uint8

COMENTÁRIO GERAL
Converte para float32, divide por 255 e reconstrói uint8 com arredondamento. Mostra a mudança no espaço de memória e no intervalo numérico.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/08_normalizacao.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/08_normalizacao/.

O QUE OBSERVAR
A matriz float32 ocupa quatro vezes os bytes de uint8 para o mesmo número de elementos.
"""
import numpy as np
from apoio import (iniciar, carregar_imagem, salvar_json)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem)
f=img.astype(np.float32)/255.0
volta=np.rint(np.clip(f,0,1)*255).astype(np.uint8)
print('uint8 bytes:',img.nbytes,'float32 bytes:',f.nbytes)
print('Intervalo normalizado:',float(f.min()),float(f.max()))
print('Diferença máxima de reconstrução:',int(np.abs(img.astype(int)-volta.astype(int)).max()))
salvar_json(out/'normalizacao.json',{'uint8_bytes':img.nbytes,'float32_bytes':f.nbytes})
