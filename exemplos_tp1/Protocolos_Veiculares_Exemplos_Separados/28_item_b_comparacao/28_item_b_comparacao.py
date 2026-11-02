"""Exemplo 28 — Item B completo: UDP versus TCP

COMENTÁRIO GERAL
Executa 320 frames para UDP e 320 para TCP nas mesmas condições, salva dois feeds recebidos, CSVs, histogramas sobrepostos com os mesmos bins, tabela de métricas e relatório fundamentado.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 28_item_b_comparacao.py --sem-gui
Colab: abrir 28_item_b_comparacao.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
O relatório separa conclusão sobre esta execução e escolha arquitetural para uma câmera real. Use --log-cada 1 se quiser imprimir todos os frames.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=320,ciclos=20)
    if a.frames<300:raise ValueError('A comparação exige ao menos 300 frames por protocolo.')
    comparar(out,parametros_stream(a),ordem=('udp','tcp'))
