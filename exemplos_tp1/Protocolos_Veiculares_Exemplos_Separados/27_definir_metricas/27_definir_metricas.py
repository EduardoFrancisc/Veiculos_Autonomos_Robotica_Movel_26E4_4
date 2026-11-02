"""Exemplo 27 — Calcular média, jitter e percentis a partir do CSV

COMENTÁRIO GERAL
Executa um novo ensaio UDP curto e recalcula as métricas lendo os timestamps brutos. Compara média, desvio padrão e diferença absoluta entre latências consecutivas.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 27_definir_metricas.py --sem-gui
Colab: abrir 27_definir_metricas.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Jitter pode ter outras definições. Aqui, o indicador principal é std(ddof=0); a variação entre frames é um indicador adicional com nome distinto.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=80,ciclos=20)
    rodar_stream('udp',out,parametros_stream(a))
    rows=ler_csv(out/'recebidos.csv')
    lat=np.array([(int(r['decode_fim_ns'])-int(r['envio_ns']))/1e6 for r in rows])
    print('Média:',lat.mean(),'ms; jitter std:',lat.std(ddof=0),'ms')
    print('p95:',np.percentile(lat,95),'ms; p99:',np.percentile(lat,99),'ms')
    print('Variação absoluta consecutiva média:',np.abs(np.diff(lat)).mean(),'ms')
    salvar_json(out/'metricas_recalculadas.json',dict(media_ms=float(lat.mean()),
                 jitter_std_ms=float(lat.std()),p95_ms=float(np.percentile(lat,95))))
