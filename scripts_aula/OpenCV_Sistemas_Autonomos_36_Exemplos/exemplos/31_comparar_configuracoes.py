"""Exemplo 31 — Comparar resolução e diâmetro

COMENTÁRIO GERAL
Carrega os mesmos frames em memória e mede nove configurações, combinando três resoluções e três diâmetros. Evita misturar leitura e gravação com o custo do pipeline.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/31_comparar_configuracoes.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/31_comparar_configuracoes/.

O QUE OBSERVAR
320×240 e 960×720 são extensões de estudo; a resolução exigida no item B continua 640×480.
"""
import numpy as np
import matplotlib.pyplot as plt
from apoio import (iniciar, abrir_video, salvar_json, salvar_csv, finalizar_figura, pipeline, ambiente)

a,out=iniciar(__file__)
cap=abrir_video(a.video);frames=[]
try:
 for _ in range(min(a.quadros,45)):
  ok,frame=cap.read()
  if not ok: break
  frames.append(frame)
finally:
 cap.release()
if len(frames)<2: raise ValueError('São necessários pelo menos dois frames.')
warm=min(a.aquecimento,len(frames)-1);linhas=[]
for tamanho in [(320,240),(640,480),(960,720)]:
 for d in [3,5,9]:
  tempos=[]
  for i,frame in enumerate(frames):
   _,_,_,t=pipeline(frame,tamanho,d,50,50)
   if i>=warm: tempos.append(t['proc_ms'])
  r={'largura':tamanho[0],'altura':tamanho[1],'d':d,'n':len(tempos),
     'media_ms':float(np.mean(tempos)),'p95_ms':float(np.percentile(tempos,95)),
     'fps_processamento':float(1000*len(tempos)/sum(tempos))}
  linhas.append(r);print(r)
salvar_csv(out/'configuracoes.csv',linhas);salvar_json(out/'ambiente.json',ambiente())
fig,ax=plt.subplots(figsize=(9,4))
for d,ls in [(3,'-'),(5,'--'),(9,':')]:
 rs=[r for r in linhas if r['d']==d]
 ax.plot([r['largura']*r['altura']/1e6 for r in rs],[r['media_ms'] for r in rs],label=f'd={d}',linestyle=ls,color='black',marker='o')
ax.set(xlabel='Milhões de pixels',ylabel='Média do pipeline (ms)');ax.legend();ax.grid(alpha=.2)
finalizar_figura(fig,out/'configuracoes.png',a)
