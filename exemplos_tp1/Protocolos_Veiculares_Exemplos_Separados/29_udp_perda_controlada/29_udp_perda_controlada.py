"""Exemplo 29 — Introduzir perda de datagramas de forma controlada

COMENTÁRIO GERAL
Usa seed 42 e probabilidade de 0,003 para retirar datagramas no publicador antes de sendto. Mede os frames que o cliente efetivamente consegue decodificar e separa perdas artificiais de perdas observadas nos datagramas enviados.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 29_udp_perda_controlada.py --sem-gui
Colab: abrir 29_udp_perda_controlada.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Não chame esses descartes de falha de uma Ethernet física. A tabela registra planejados, enviados, retirados artificialmente e recebidos únicos.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=320,ciclos=20)
    if a.frames<300:raise ValueError('Mantenha ao menos 300 frames neste experimento.')
    r=rodar_stream('udp',out,parametros_stream(a,drop_prob=.003,seed=42))
    print('Drops artificiais:',r['drops_artificiais'],
          'perda observada nos enviados (%):',r['perda_datagramas_observada_pct'],
          'perda de frames (%):',r['perda_frames_pct'])
