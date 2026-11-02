"""Exemplo 18 — Conectar cliente e servidor TCP

COMENTÁRIO GERAL
Um servidor em thread aceita uma conexão de loopback e envia texto. O cliente usa recv_exact para obter a quantidade conhecida de bytes, independentemente das leituras parciais.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 18_tcp_texto.py --sem-gui
Colab: abrir 18_tcp_texto.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
sendall entrega bytes ao mecanismo TCP local; o retorno não significa que a aplicação remota já decodificou os dados.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    payload=b'temperatura=82.75'
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as srv:
        srv.bind(('127.0.0.1',0));srv.listen(1)
        def enviar():
            conn,_=srv.accept()
            with conn:conn.sendall(payload)
        th=threading.Thread(target=enviar);th.start()
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as cli:
            cli.settimeout(2);cli.connect(srv.getsockname())
            print('Recebido:',recv_exact(cli,len(payload)).decode())
        th.join()
