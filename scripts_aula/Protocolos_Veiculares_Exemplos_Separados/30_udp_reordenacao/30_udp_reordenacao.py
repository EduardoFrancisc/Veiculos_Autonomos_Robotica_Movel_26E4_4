"""Exemplo 30 — Enviar datagramas fora de ordem e duplicados

COMENTÁRIO GERAL
Embaralha a ordem de envio dos fragmentos de cada JPEG e duplica um fragmento por frame usando sockets reais. O cliente usa índices e deduplicação para reconstruir o JPEG.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 30_udp_reordenacao.py --sem-gui
Colab: abrir 30_udp_reordenacao.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
A reordenação é introduzida no emissor. O experimento verifica a lógica da aplicação, sem atribuir esse comportamento à rede física.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    rodar_stream('udp',out,parametros_stream(a,reorder=True,duplicates=True))
