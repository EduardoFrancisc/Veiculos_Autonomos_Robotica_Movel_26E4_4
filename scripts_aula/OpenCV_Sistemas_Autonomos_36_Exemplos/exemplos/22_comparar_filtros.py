"""Exemplo 22 — Comparar Gaussian, mediana e bilateral

COMENTÁRIO GERAL
Aplica três filtros ao mesmo ruído artificial e calcula MSE em relação à referência. O painel permite examinar a troca entre suavização e preservação de detalhes.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/22_comparar_filtros.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/22_comparar_filtros/.

O QUE OBSERVAR
O menor MSE não garante a melhor localização de uma faixa; inspecione bordas e textura.
"""
import cv2
import numpy as np
from apoio import (iniciar, carregar_imagem, cinza, salvar_csv, painel)

a,out=iniciar(__file__)
y=cinza(carregar_imagem(a.imagem));rng=np.random.default_rng(a.seed)
suja=np.clip(y.astype(np.float32)+rng.normal(0,12,y.shape),0,255).astype(np.uint8)
resultados={'Gaussian 5 × 5':cv2.GaussianBlur(suja,(5,5),0),
            'Mediana 5 × 5':cv2.medianBlur(suja,5),
            'Bilateral d=5':cv2.bilateralFilter(suja,5,50,50)}
linhas=[]
for nome,im in resultados.items():
 mse=float(np.mean((y.astype(np.float32)-im.astype(np.float32))**2))
 linhas.append({'filtro':nome,'mse_referencia':mse});print(nome,'MSE',mse)
salvar_csv(out/'filtros.csv',linhas)
painel([y,suja,*resultados.values()],['Referência','Ruído',*resultados.keys()],out/'filtros.png',a)
