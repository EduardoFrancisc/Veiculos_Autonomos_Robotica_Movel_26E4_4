"""Exemplo 24 — Receber o primeiro feed TCP

COMENTÁRIO GERAL
Repete o streaming com TCP, cabeçalho de comprimento, recv_exact e checagem do JPEG. Cliente e servidor são processos distintos, com as mesmas condições visuais usadas no UDP.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 24_camera_tcp_curto.py --sem-gui
Colab: abrir 24_camera_tcp_curto.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Confiabilidade e ordem pertencem ao TCP; a aplicação ainda precisa enquadrar mensagens e verificar encerramento.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    rodar_stream('tcp',out,parametros_stream(a))
