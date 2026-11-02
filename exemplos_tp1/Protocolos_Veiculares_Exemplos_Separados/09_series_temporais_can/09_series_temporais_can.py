"""Exemplo 09 — Registrar CSV e plotar três séries

COMENTÁRIO GERAL
Executa um ensaio completo com nós virtuais em threads, salva cada mensagem com timestamp e valor físico e desenha três séries temporais separadas.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 09_series_temporais_can.py --sem-gui
Colab: abrir 09_series_temporais_can.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Os valores são sensores simulados. Os timestamps e a execução do software são medidos de verdade.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=30)
    res=can_threads(out,ciclos=a.ciclos,hz=a.hz)
    print('CSV:',out/'mensagens_can.csv')
    print('Gráfico:',out/'series_can.png')
