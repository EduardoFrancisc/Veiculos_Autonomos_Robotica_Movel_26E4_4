"""Exemplo 12 — Construir um histograma de 256 níveis

COMENTÁRIO GERAL
Conta quantos pixels possuem cada intensidade com numpy.bincount e confirma que a soma das contagens equivale ao número de pixels.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/12_histograma.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/12_histograma/.

O QUE OBSERVAR
Um histograma informa distribuição tonal, mas não a localização dos pixels na pista.
"""
import numpy as np
import matplotlib.pyplot as plt
from apoio import (iniciar, carregar_imagem, cinza, salvar_csv, finalizar_figura)

a,out=iniciar(__file__)
y=cinza(carregar_imagem(a.imagem));h=np.bincount(y.ravel(),minlength=256)
print('Pixels:',y.size,'Soma do histograma:',int(h.sum()))
assert h.sum()==y.size
fig,ax=plt.subplots(figsize=(8,4));ax.plot(h,color='black')
ax.set(xlabel='Intensidade',ylabel='Quantidade de pixels',xlim=(0,255));ax.grid(alpha=.2)
finalizar_figura(fig,out/'histograma.png',a)
salvar_csv(out/'histograma.csv',[{'intensidade':i,'contagem':int(v)} for i,v in enumerate(h)])
