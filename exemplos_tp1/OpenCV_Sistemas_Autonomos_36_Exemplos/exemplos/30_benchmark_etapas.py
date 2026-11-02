"""Exemplo 30 — Encontrar a etapa que consome mais tempo

COMENTÁRIO GERAL
Executa o mesmo pipeline sem gravar vídeo e sem imprimir cada frame, calcula médias por etapa e apresenta latência por frame e o prazo da meta de FPS.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/30_benchmark_etapas.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/30_benchmark_etapas/.

O QUE OBSERVAR
A medição sem gravação é comparável apenas com outra execução sob o mesmo protocolo.
"""
import numpy as np
import matplotlib.pyplot as plt
from apoio import (iniciar, finalizar_figura, rodar_stream)

a,out=iniciar(__file__)
linhas,resumo=rodar_stream(a,out,gravar=False,verbose=False)
etapas=['resize_ms','hsv_ms','extracao_v_ms','bilateral_ms']
medias=[np.mean([float(r[k]) for r in linhas]) for k in etapas]
fig,axs=plt.subplots(1,2,figsize=(11,4))
axs[0].bar(['Resize','HSV','Extrair V','Bilateral'],medias,color='0.55');axs[0].set_ylabel('Média (ms)')
axs[1].plot([r['frame'] for r in linhas],[r['loop_ms'] for r in linhas],color='black',label='Loop')
axs[1].axhline(1000/a.fps_alvo,color='0.4',linestyle='--',label=f'Prazo: {a.fps_alvo:g} FPS')
axs[1].set(xlabel='Frame',ylabel='Tempo (ms)');axs[1].legend()
finalizar_figura(fig,out/'etapas_prazo.png',a)
