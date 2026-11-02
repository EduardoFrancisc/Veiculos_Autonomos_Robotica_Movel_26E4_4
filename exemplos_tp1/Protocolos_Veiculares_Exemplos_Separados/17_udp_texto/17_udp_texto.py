"""Exemplo 17 — Enviar um datagrama UDP de texto

COMENTÁRIO GERAL
Cria sockets UDP reais de loopback, usa uma porta livre e envia uma mensagem textual. O receptor recebe um datagrama completo com recvfrom.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 17_udp_texto.py --sem-gui
Colab: abrir 17_udp_texto.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Um sendto representa um datagrama, mas pode falhar ou o datagrama pode não chegar. UDP não garante entrega nem ordem.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as rx,      socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as tx:
        rx.bind(('127.0.0.1',0));rx.settimeout(1)
        tx.sendto(b'velocidade=45.30',rx.getsockname())
        data,addr=rx.recvfrom(1024)
        print('Recebido:',data.decode(),'origem:',addr)
