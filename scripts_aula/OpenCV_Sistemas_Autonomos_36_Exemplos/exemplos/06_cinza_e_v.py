"""Exemplo 06 — Distinguir cinza e canal V

COMENTÁRIO GERAL
Produz cinza por BGR2GRAY e extrai V após BGR2HSV. Cinza é uma combinação ponderada de canais; V acompanha o maior componente RGB. Não são a mesma medida.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/06_cinza_e_v.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/06_cinza_e_v/.

O QUE OBSERVAR
Um objeto colorido pode ter V alto e cinza menor. Declare a definição de brilho antes de comparar regiões.
"""
import cv2
from apoio import (iniciar, carregar_imagem, cinza, painel)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem);y=cinza(img)
v=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,2]
print('Média cinza:',float(y.mean()),'Média V:',float(v.mean()))
painel([img,y,v],['Original','Cinza ponderado','V = valor do HSV'],out/'cinza_v.png',a)
