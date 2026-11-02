"""Exemplo 15 — Detectar ausência e duplicação por contador

COMENTÁRIO GERAL
Envia uma sequência pelo VirtualBus, retirando dois contadores e duplicando um para criar um caso controlado. Calcula IDs ausentes e duplicados após a decodificação.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 15_contador_e_ausencias.py --sem-gui
Colab: abrir 15_contador_e_ausencias.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
As ausências são introduzidas no publicador para estudo; não são erros de bit medidos em CAN físico.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    seqs=[0,1,2,4,5,5,6,8,9]
    channel='falhas-'+uuid4().hex;recebidos=[]
    with can.Bus(interface='virtual',channel=channel) as tx,      can.Bus(interface='virtual',channel=channel) as rx:
        for s in seqs:
            tx.send(can.Message(arbitration_id=0x100,is_extended_id=False,
                                data=codificar(0x100,40,s,s*100),check=True))
            recebidos.append(decodificar(rx.recv(1))[3])
    ausentes=sorted(set(range(10))-set(recebidos))
    duplicados=sorted({s for s in recebidos if recebidos.count(s)>1})
    print('Ausências artificiais:',ausentes,'duplicados:',duplicados)
    salvar_json(out/'contadores.json',dict(ausentes=ausentes,duplicados=duplicados))
