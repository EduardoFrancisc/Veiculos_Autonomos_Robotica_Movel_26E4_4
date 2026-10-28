"""Exemplo 01 — Conferir bibliotecas e relógios

COMENTÁRIO GERAL
Identifica CPU, sistema, versões e relógio monotônico antes de medir. Registra o ambiente em JSON; os experimentos de sockets usam o mesmo host e não utilizam GPU.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 01_ambiente.py --sem-gui
Colab: abrir 01_ambiente.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Confirme o interpretador Python. Nanosegundos são a unidade retornada, não uma garantia de precisão de um nanosegundo.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    print(json.dumps(ambiente(), ensure_ascii=False, indent=2))
    print('Clock monotônico:', time.get_clock_info('monotonic'))
    print('Resolução do clock (ns):', time.get_clock_info('monotonic').resolution*1e9)
