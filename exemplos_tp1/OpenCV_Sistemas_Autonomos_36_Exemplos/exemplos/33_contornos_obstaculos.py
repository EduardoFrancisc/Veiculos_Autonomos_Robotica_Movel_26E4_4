"""Exemplo 33 — Examinar contornos sem atribuir classes

COMENTÁRIO GERAL
Filtra V, extrai bordas e desenha retângulos de contornos grandes na metade inferior. O exemplo mostra como ruído e sombras alteram candidatos geométricos.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/33_contornos_obstaculos.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/33_contornos_obstaculos/.

O QUE OBSERVAR
Um retângulo não comprova um obstáculo. Um detector real precisa de modelo ou critérios e validação com anotações.
"""
import cv2
from apoio import (iniciar, carregar_imagem, painel, pipeline)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem);menor,v,f,_=pipeline(img)
bordas=cv2.Canny(f,80,160);bordas[:240]=0
contornos,_=cv2.findContours(bordas,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
view=menor.copy();n=0
for c in contornos:
 x,y,w,h=cv2.boundingRect(c)
 if w*h>=600:
  cv2.rectangle(view,(x,y),(x+w,y+h),(255,255,255),1);n+=1
print('Regiões geométricas candidatas:',n,'(não são obstáculos confirmados)')
painel([menor,bordas,view],['Frame','Bordas na metade inferior','Retângulos de contornos'],out/'contornos.png',a)
