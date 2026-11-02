"""Exemplo 34 — Entender offset entre relógios de dois hosts

COMENTÁRIO GERAL
Desenha um exemplo numérico de como o offset do relógio se soma à latência unidirecional. O guia de dois terminais permite rodar os serviços reais em uma máquina ou em uma LAN.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 34_relogios_e_lan.py --sem-gui
Colab: abrir 34_relogios_e_lan.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
O gráfico é ilustrativo. Clock monotônico de dois hosts distintos não tem origem comum. Use epoch com sincronização validada ou outro método de medição.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    offsets=np.array([-10,-3,0,3,10],dtype=float);latencia_real=2.
    medida=latencia_real+offsets
    rows=[dict(offset_ilustrativo_ms=o,latencia_real_ilustrativa_ms=latencia_real,
               latencia_aparente_ms=m) for o,m in zip(offsets,medida)]
    salvar_csv(out/'modelo_offset.csv',rows)
    fig,ax=plt.subplots(figsize=(9,4));ax.plot(offsets,medida,'ko-')
    ax.axhline(latencia_real,color='0.5',linestyle='--',label='Latência ilustrativa real: 2 ms')
    ax.set(xlabel='Offset do receptor (ms)',ylabel='Latência aparente (ms)');ax.legend()
    salvar_fig(fig,out/'offset_relogio.png');print(rows)
    print('Consulte documentacao/GUIA_DOIS_TERMINAIS.md para os serviços TCP/UDP.')
