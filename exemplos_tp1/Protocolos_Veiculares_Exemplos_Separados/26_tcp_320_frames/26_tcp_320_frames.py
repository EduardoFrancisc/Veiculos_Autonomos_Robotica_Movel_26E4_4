"""Exemplo 26 — Medir TCP com pelo menos 300 frames

COMENTÁRIO GERAL
Envia 320 frames por padrão e mede a mesma latência da aplicação. Identifica IDs não decodificados sem afirmar que isso mede perdas de segmentos TCP.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 26_tcp_320_frames.py --sem-gui
Colab: abrir 26_tcp_320_frames.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Retransmissões TCP não são contadas por esse código. Para estudá-las, seria necessária instrumentação da pilha ou captura de tráfego.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=320,ciclos=20)
    if a.frames<300:raise ValueError('Este ensaio exige ao menos 300 frames.')
    rodar_stream('tcp',out,parametros_stream(a))
