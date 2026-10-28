"""Exemplo 05 — Enviar a primeira mensagem CAN virtual

COMENTÁRIO GERAL
Abre dois objetos Bus no mesmo canal virtual e no mesmo processo, envia uma mensagem e a recebe com timeout. O contexto with encerra os nós ao final.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 05_can_virtual.py --sem-gui
Colab: abrir 05_can_virtual.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Dois terminais com interface virtual não se enxergam automaticamente: os objetos precisam estar no mesmo processo ou usar uma ponte explícita.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    channel='primeiro-'+uuid4().hex
    with can.Bus(interface='virtual',channel=channel) as tx,      can.Bus(interface='virtual',channel=channel) as rx:
        msg=can.Message(arbitration_id=0x100,is_extended_id=False,
                        data=codificar(0x100,45.3,0,0),check=True)
        tx.send(msg)
        recebido=rx.recv(timeout=1)
        if recebido is None: raise RuntimeError('Nenhuma mensagem recebida.')
        print('Recebido:',decodificar(recebido),'timestamp=',recebido.timestamp)
