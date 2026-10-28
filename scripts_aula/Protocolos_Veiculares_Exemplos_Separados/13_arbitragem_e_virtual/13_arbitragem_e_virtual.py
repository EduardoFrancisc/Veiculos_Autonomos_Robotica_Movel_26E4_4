"""Exemplo 13 — Distinguir arbitragem física e ordem da fila

COMENTÁRIO GERAL
Envia IDs em uma ordem definida e observa a ordem do VirtualBus. Mostra ao lado uma ordenação conceitual por prioridade, válida como ideia para disputa simultânea entre frames padrão equivalentes.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 13_arbitragem_e_virtual.py --sem-gui
Colab: abrir 13_arbitragem_e_virtual.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
O modelo de prioridade é ilustrativo. O VirtualBus mantém ordem de envio e não executa arbitragem bit a bit.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    ids=[0x300,0x100,0x200]
    channel='ordem-'+uuid4().hex
    with can.Bus(interface='virtual',channel=channel) as tx,      can.Bus(interface='virtual',channel=channel) as rx:
        for i in ids:tx.send(can.Message(arbitration_id=i,is_extended_id=False,data=b'AB',check=True))
        observado=[rx.recv(1).arbitration_id for _ in ids]
    print('Ordem observada no VirtualBus:',[hex(i) for i in observado])
    print('Modelo didático de disputa simultânea:',[hex(i) for i in sorted(ids)])
    salvar_json(out/'ordem.json',dict(virtual=observado,modelo_prioridade=sorted(ids)))
