"""Exemplo 04 — Exibir as cores corretamente

COMENTÁRIO GERAL
Compara a exibição incorreta de BGR diretamente no Matplotlib com a conversão BGR→RGB. O arquivo original continua BGR nas operações do OpenCV.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/04_bgr_rgb.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/04_bgr_rgb/.

O QUE OBSERVAR
Veja como azul e vermelho trocam de posição quando a ordem dos canais é ignorada.
"""
import cv2
import matplotlib.pyplot as plt
from apoio import (iniciar, carregar_imagem, finalizar_figura)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem)
fig,axs=plt.subplots(1,2,figsize=(11,4))
axs[0].imshow(img);axs[0].set_title('BGR tratado como RGB: cores incorretas')
axs[1].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));axs[1].set_title('Conversão BGR → RGB')
for ax in axs: ax.axis('off')
finalizar_figura(fig,out/'cores.png',a)
print('BGR no OpenCV; RGB no Matplotlib.')
