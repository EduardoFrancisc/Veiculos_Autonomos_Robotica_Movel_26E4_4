"""Exemplo 25 — Medir o custo do diâmetro do bilateral

COMENTÁRIO GERAL
Varia somente d entre 3, 5 e 9, realiza cinco aquecimentos por configuração e mede vinte aplicações ao mesmo canal V. É um microbenchmark do filtro, sem leitura de vídeo.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/25_custo_bilateral.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/25_custo_bilateral/.

O QUE OBSERVAR
Compare mediana e p95. Os tempos pertencem ao computador em que o código foi executado.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from apoio import (iniciar, carregar_imagem, salvar_json, salvar_csv, finalizar_figura, ambiente)

from time import perf_counter_ns
a,out=iniciar(__file__)
img=cv2.resize(carregar_imagem(a.imagem),(640,480),interpolation=cv2.INTER_AREA)
v=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,2];linhas=[]
for d in [3,5,9]:
 for _ in range(5): cv2.bilateralFilter(v,d,50,50)
 tempos=[]
 for _ in range(20):
  t0=perf_counter_ns();cv2.bilateralFilter(v,d,50,50);tempos.append((perf_counter_ns()-t0)/1e6)
 r={'d':d,'mediana_ms':float(np.median(tempos)),'p95_ms':float(np.percentile(tempos,95)),'media_ms':float(np.mean(tempos))}
 linhas.append(r);print(r)
salvar_csv(out/'custo_d.csv',linhas);salvar_json(out/'ambiente.json',ambiente())
fig,ax=plt.subplots(figsize=(7,4));ax.bar([str(r['d']) for r in linhas],[r['mediana_ms'] for r in linhas],color='0.55')
ax.set(xlabel='Diâmetro d',ylabel='Mediana do tempo (ms)');finalizar_figura(fig,out/'custo_d.png',a)
