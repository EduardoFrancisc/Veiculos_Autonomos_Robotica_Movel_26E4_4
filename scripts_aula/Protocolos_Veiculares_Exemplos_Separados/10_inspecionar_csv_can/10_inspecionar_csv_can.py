"""Exemplo 10 — Inspecionar o registro produzido

COMENTÁRIO GERAL
Gera sua própria captura CAN e relê o CSV, sem depender do exemplo 09. Verifica a quantidade por ID e a progressão dos contadores.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 10_inspecionar_csv_can.py --sem-gui
Colab: abrir 10_inspecionar_csv_can.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
A contagem esperada é ciclos × 3. Conferir sequências ajuda a detectar ausência e duplicação na aplicação.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=15)
    can_threads(out,ciclos=a.ciclos,hz=a.hz,log=False)
    rows=ler_csv(out/'mensagens_can.csv')
    for i in IDS:
        rs=[r for r in rows if r['id_hex']==f'0x{i:03X}']
        seq=[int(r['contador']) for r in rs]
        print(f'0x{i:03X}: {len(rs)} mensagens; contadores:',seq)
        assert seq==list(range(a.ciclos))
