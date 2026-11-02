"""Exemplo 11 — Medir período observado no CAN virtual

COMENTÁRIO GERAL
Coleta mensagens e calcula diferenças entre timestamps monotônicos de recepção, média e desvio padrão. Plota os intervalos em torno da meta de 100 ms.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 11_periodo_e_jitter_can.py --sem-gui
Colab: abrir 11_periodo_e_jitter_can.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
O jitter medido pertence ao software do ensaio. O VirtualBus não reproduz a serialização dos bits no fio.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=30)
    can_threads(out,ciclos=a.ciclos,hz=a.hz,log=False)
    rows=[r for r in ler_csv(out/'mensagens_can.csv') if r['id_hex']=='0x100']
    intervalos=np.diff([int(r['rx_mono_ns']) for r in rows])/1e6
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(np.arange(1,len(rows)),intervalos,'k.-')
    ax.axhline(1000/a.hz,color='0.5',linestyle='--',label='Período-alvo')
    ax.set(xlabel='Intervalo',ylabel='Recepção → recepção (ms)');ax.legend()
    salvar_fig(fig,out/'intervalos_can.png')
    print('Média (ms):',intervalos.mean(),'jitter std (ms):',intervalos.std())
