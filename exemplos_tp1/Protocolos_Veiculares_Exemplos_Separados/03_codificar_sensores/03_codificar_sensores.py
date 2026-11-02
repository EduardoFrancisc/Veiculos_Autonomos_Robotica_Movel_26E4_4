"""Exemplo 03 — Codificar velocidade, RPM e temperatura

COMENTÁRIO GERAL
Transforma números físicos em payloads de 8 bytes com struct.pack. Os dois primeiros bytes guardam o valor; os dois seguintes, o contador; os quatro últimos, o tempo simulado em milissegundos.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 03_codificar_sensores.py --sem-gui
Colab: abrir 03_codificar_sensores.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Velocidade e temperatura usam escala 0,01; RPM usa escala 1. Os formatos iniciados por < usam little-endian.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    rows=[]
    for i,valor in [(0x100,45.3),(0x101,1600),(0x102,82.75)]:
        data=codificar(i,valor,seq=7,t_ms=700)
        print(f'0x{i:03X}:', data.hex(' '), 'bytes=',len(data))
        rows.append(dict(id_hex=f'0x{i:03X}',valor=valor,payload_hex=data.hex(' ')))
    salvar_csv(out/'payloads.csv',rows)
