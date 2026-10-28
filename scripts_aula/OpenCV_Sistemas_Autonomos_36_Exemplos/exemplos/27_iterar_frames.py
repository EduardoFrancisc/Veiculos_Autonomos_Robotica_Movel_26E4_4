"""Exemplo 27 — Ler frames sequencialmente

COMENTÁRIO GERAL
Percorre o vídeo até o limite ou o fim, calcula a média de V por frame e desenha a evolução temporal. Este exemplo introduz o loop antes de acrescentar o filtro.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/27_iterar_frames.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/27_iterar_frames/.

O QUE OBSERVAR
Mudanças de brilho também podem vir de uma mudança da cena. Não atribua toda variação à exposição.
"""
import cv2
import matplotlib.pyplot as plt
from apoio import (iniciar, abrir_video, salvar_csv, finalizar_figura)

a,out=iniciar(__file__)
cap=abrir_video(a.video);linhas=[]
try:
 for i in range(a.quadros):
  ok,frame=cap.read()
  if not ok: break
  v=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,2]
  r={'frame':i,'media_v':float(v.mean())};linhas.append(r);print(r)
finally:
 cap.release()
salvar_csv(out/'brilho_video.csv',linhas)
fig,ax=plt.subplots(figsize=(8,4));ax.plot([r['frame'] for r in linhas],[r['media_v'] for r in linhas],color='black')
ax.set(xlabel='Frame',ylabel='Média V (0–255)');ax.grid(alpha=.2)
finalizar_figura(fig,out/'brilho_video.png',a)
