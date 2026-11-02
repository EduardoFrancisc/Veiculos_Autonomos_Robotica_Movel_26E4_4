"""Exemplo 14 — Item A completo com dois processos

COMENTÁRIO GERAL
Cria processos distintos publicador e assinante. Cada processo usa dois nós de interface virtual; gateways e multiprocessing.Queue formam a ponte entre processos. O assinante filtra, decodifica, imprime e grava CSV. O programa plota as séries.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 14_item_a_processos.py --sem-gui
Colab: abrir 14_item_a_processos.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
A ponte IPC é necessária porque VirtualBus é intraprocesso. O arquivo processos.json comprova PIDs distintos e documenta a ponte usada.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=50)
    # CAN 2.0A: IDs de 11 bits, até 8 bytes; Classical CAN até 1 Mbit/s.
    # CAN FD: IDs de 11 ou 29 bits, até 64 bytes e fase de dados mais rápida com BRS.
    # Não comparar estes tempos de software com latência de transceptor físico.
    can_processos(out,ciclos=a.ciclos,hz=a.hz)
