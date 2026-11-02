"""Exemplo 08 — Publicar periodicamente a 10 Hz por sinal

COMENTÁRIO GERAL
Um publicador em thread envia os três sensores em cada ciclo. O instante de cada ciclo é calculado a partir de uma origem monotônica, reduzindo o acúmulo de atraso de sleeps sucessivos.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 08_publicacao_10hz.py --sem-gui
Colab: abrir 08_publicacao_10hz.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
10 Hz por ID significa 30 mensagens por segundo no conjunto. Python e um SO comum podem introduzir atraso de escalonamento.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=10)
    can_threads(out,ciclos=a.ciclos,hz=a.hz)
