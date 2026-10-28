"""Exemplo 22 — Identificar fragmentos ausentes e duplicados

COMENTÁRIO GERAL
Constrói fragmentos, retira um e duplica outro em uma lista didática. Usa conjuntos para identificar ausência e duplicação antes de tentar decodificar um JPEG incompleto.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 22_ausencias_e_duplicatas_udp.py --sem-gui
Colab: abrir 22_ausencias_e_duplicatas_udp.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
A ausência é artificial. No receptor real, um frame incompleto expira por timeout e é contabilizado como perda de frame.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    _,buf=cv2.imencode('.jpg',quadro_inicial(a))
    packets=fragmentar(buf.tobytes(),0,1,0,a.datagrama)
    simulados=packets[1:]+[packets[-1]]
    indices=[ler_datagrama(p)[0][1] for p in simulados]
    faltantes=sorted(set(range(len(packets)))-set(indices))
    duplicados=sorted({i for i in indices if indices.count(i)>1})
    print('Ausente:',faltantes,'duplicado:',duplicados,'JPEG não será decodificado.')
    salvar_json(out/'remontagem_incompleta.json',dict(ausentes=faltantes,duplicados=duplicados))
