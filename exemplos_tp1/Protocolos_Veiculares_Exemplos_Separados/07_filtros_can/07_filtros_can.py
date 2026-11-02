"""Exemplo 07 — Filtrar por ID padrão

COMENTÁRIO GERAL
Envia três IDs e configura o receptor para aceitar apenas 0x100. Usa máscara 0x7FF para comparar todos os 11 bits e extended=False para exigir formato padrão.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 07_filtros_can.py --sem-gui
Colab: abrir 07_filtros_can.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
O filtro reduz mensagens entregues à aplicação. Ele não reduz o tráfego que já circulou pelo meio físico.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    channel='filtro-'+uuid4().hex
    filtro=[{'can_id':0x100,'can_mask':0x7FF,'extended':False}]
    with can.Bus(interface='virtual',channel=channel) as tx,      can.Bus(interface='virtual',channel=channel,can_filters=filtro) as rx:
        for i,valor in sinais_simulados(0).items():
            tx.send(can.Message(arbitration_id=i,is_extended_id=False,
                                data=codificar(i,valor,0,0),check=True))
        msg=rx.recv(1);print('Aceito:',decodificar(msg))
        print('Outra mensagem disponível:',rx.recv(.05))
