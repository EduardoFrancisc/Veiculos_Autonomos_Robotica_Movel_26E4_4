"""Exemplo 02 — Representar IDs e tamanho de payload

COMENTÁRIO GERAL
Converte os três IDs exigidos entre hexadecimal, decimal e binário. Usa can.Message com validação para distinguir identificador, tamanho de dados e flag de formato padrão.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 02_ids_e_payload.py --sem-gui
Colab: abrir 02_ids_e_payload.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
0x100 não é um endereço de nó. É um identificador de mensagem; a associação com um sinal é definida pelo projeto.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    for i in IDS:
        print(f'ID=0x{i:03X} decimal={i} binário={i:011b}')
    msg=can.Message(arbitration_id=0x100,is_extended_id=False,data=bytes(8),check=True)
    print('Mensagem padrão:', msg, 'payload:', len(msg.data), 'bytes')
    try:
        can.Message(arbitration_id=0x800,is_extended_id=False,data=b'',check=True)
    except ValueError as e:
        print('ID de 12 bits rejeitado no formato padrão:', e)
