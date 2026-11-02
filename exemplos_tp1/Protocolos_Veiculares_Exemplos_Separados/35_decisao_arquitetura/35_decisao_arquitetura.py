"""Exemplo 35 — Montar evidências para uma decisão de arquitetura

COMENTÁRIO GERAL
Gera suas próprias medições CAN e TCP/UDP curtas e registra uma matriz de decisão. A escolha diferencia sinais de controle, câmera para percepção e transferência íntegra de arquivos.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 35_decisao_arquitetura.py --sem-gui
Colab: abrir 35_decisao_arquitetura.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Latência média e ausência de perdas em loopback não demonstram determinismo. Vincule cada escolha ao volume, ao prazo e à tolerância a lacunas.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    can_threads(out/'can',ciclos=10,hz=a.hz,log=False)
    comparar(out/'stream',parametros_stream(a))
    rows=[dict(dado='Velocidade/RPM/temperatura',meio='CAN/CAN FD',transporte='Protocolo do projeto',criterio='Pequeno payload; prioridade e prazo'),
          dict(dado='Câmera para percepção',meio='Ethernet automotiva',transporte='UDP com política de ausência',criterio='Volume e atualidade dos frames'),
          dict(dado='Arquivo/registro íntegro',meio='Ethernet',transporte='TCP',criterio='Entrega ordenada de todos os bytes')]
    salvar_csv(out/'decisao_arquitetura.csv',rows);print(rows)
