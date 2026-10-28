"""Exemplo 21 — Fragmentar e remontar JPEG por índice

COMENTÁRIO GERAL
Divide um JPEG em datagramas de aplicação de até 1200 bytes, embaralha os fragmentos em memória e reconstrói o JPEG pelos índices. Confere comprimento e CRC32.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 21_fragmentar_udp.py --sem-gui
Colab: abrir 21_fragmentar_udp.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Este exemplo ensina o formato em memória; a transmissão por sockets começa no exemplo 23. Os 1200 bytes incluem o cabeçalho da aplicação, mas não IP/UDP.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    frame=quadro_inicial(a);_,buf=cv2.imencode('.jpg',frame)
    jpeg=buf.tobytes();pacotes=fragmentar(jpeg,7,123,100,a.datagrama)
    random.Random(42).shuffle(pacotes)
    partes={}
    for packet in pacotes:
        meta,body=ler_datagrama(packet);partes[meta[1]]=body
    remontado=b''.join(partes[i] for i in range(len(partes)))
    assert remontado==jpeg
    print('JPEG:',len(jpeg),'bytes; fragmentos:',len(pacotes),
          'maior datagrama:',max(map(len,pacotes)),'cabeçalho:',UDP_H.size)
    salvar_json(out/'fragmentacao.json',dict(jpeg_bytes=len(jpeg),fragmentos=len(pacotes),
                 datagrama_total_max=max(map(len,pacotes)),crc32=zlib.crc32(remontado)))
