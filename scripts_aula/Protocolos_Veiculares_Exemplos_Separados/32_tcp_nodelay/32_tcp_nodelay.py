"""Exemplo 32 — Comparar TCP_NODELAY no mesmo cenário

COMENTÁRIO GERAL
Executa dois ensaios TCP curtos, um com e outro sem TCP_NODELAY. Salva a comparação medida. O efeito pode ser pequeno para JPEGs maiores que um segmento.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 32_tcp_nodelay.py --sem-gui
Colab: abrir 32_tcp_nodelay.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Não conclua que TCP_NODELAY sempre melhora o resultado. Escalonamento e ordem das execuções podem superar a diferença observada.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=80,ciclos=20)
    rows=[]
    for flag in [False,True]:
        r=rodar_stream('tcp',out/f'nodelay_{int(flag)}',parametros_stream(a,tcp_nodelay=flag))
        rows.append(dict(tcp_nodelay=flag,media_ms=r['media_latencia_ms'],jitter_ms=r['jitter_std_ms']))
    salvar_csv(out/'nodelay.csv',rows);print(rows)
