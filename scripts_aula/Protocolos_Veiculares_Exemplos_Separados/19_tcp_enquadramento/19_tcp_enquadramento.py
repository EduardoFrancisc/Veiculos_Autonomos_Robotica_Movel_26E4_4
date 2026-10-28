"""Exemplo 19 — Enquadrar mensagens no fluxo TCP

COMENTÁRIO GERAL
Envia três textos prefixados por comprimento de 4 bytes em network byte order. O servidor envia pedaços pequenos para evidenciar que a aplicação precisa remontar cabeçalho e payload.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 19_tcp_enquadramento.py --sem-gui
Colab: abrir 19_tcp_enquadramento.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
TCP não preserva fronteiras entre chamadas sendall. Um recv pode conter parte de uma mensagem ou bytes de várias mensagens.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    mensagens=['velocidade=45.30','rpm=1600','temperatura=82.75']
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as srv:
        srv.bind(('127.0.0.1',0));srv.listen(1)
        def enviar():
            conn,_=srv.accept()
            with conn:
                for texto in mensagens:
                    data=texto.encode();wire=struct.pack('!I',len(data))+data
                    for i in range(0,len(wire),3):conn.sendall(wire[i:i+3])
        th=threading.Thread(target=enviar);th.start()
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as cli:
            cli.settimeout(2);cli.connect(srv.getsockname())
            for _ in mensagens:
                n=struct.unpack('!I',recv_exact(cli,4))[0]
                print('Mensagem completa:',recv_exact(cli,n).decode())
        th.join()
