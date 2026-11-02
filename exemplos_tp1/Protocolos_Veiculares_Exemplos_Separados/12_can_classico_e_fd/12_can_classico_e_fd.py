"""Exemplo 12 — Comparar CAN clássico e CAN FD

COMENTÁRIO GERAL
Constrói uma mensagem clássica de 8 bytes e uma FD de 64 bytes, envia ambas no VirtualBus e imprime flags e tamanhos. A comparação de bitrate é conceitual, não um benchmark físico.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 12_can_classico_e_fd.py --sem-gui
Colab: abrir 12_can_classico_e_fd.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
CAN FD pode usar IDs de 11 ou 29 bits. FD não significa necessariamente ID estendido, e BRS não configura a velocidade elétrica do VirtualBus.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    # CAN 2.0A: 11 bits; Classical CAN: até 8 bytes e até 1 Mbit/s.
    # CAN FD: 11/29 bits, até 64 bytes; BRS permite fase de dados mais rápida.
    # Ex.: nominal 500 kbit/s, dados 2 Mbit/s; depende do hardware/rede.
    channel='fd-'+uuid4().hex
    with can.Bus(interface='virtual',channel=channel) as tx,      can.Bus(interface='virtual',channel=channel) as rx:
        for fd,tamanho in [(False,8),(True,64)]:
            msg=can.Message(arbitration_id=0x100,is_extended_id=False,is_fd=fd,
                            bitrate_switch=fd,data=bytes(range(tamanho)),check=True)
            tx.send(msg);m=rx.recv(1)
            print('FD=',m.is_fd,'BRS=',m.bitrate_switch,'payload=',len(m.data),'bytes')
