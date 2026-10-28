"""Exemplo 06 — Observar dois assinantes

COMENTÁRIO GERAL
Cria um publicador e dois receptores no mesmo canal e confirma que ambos recebem a mensagem. Demonstra difusão e a independência das filas de recepção.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 06_multiplos_receptores.py --sem-gui
Colab: abrir 06_multiplos_receptores.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
O VirtualBus entrega cópias ordenadas às filas; essa observação não testa carga elétrica, ACK ou arbitragem de um barramento real.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    channel='assinantes-'+uuid4().hex
    with can.Bus(interface='virtual',channel=channel) as tx,      can.Bus(interface='virtual',channel=channel) as r1,      can.Bus(interface='virtual',channel=channel) as r2:
        tx.send(can.Message(arbitration_id=0x101,is_extended_id=False,
                            data=codificar(0x101,1800,1,100),check=True))
        for nome,rx in [('A',r1),('B',r2)]:
            m=rx.recv(1)
            print('Assinante',nome,decodificar(m))
