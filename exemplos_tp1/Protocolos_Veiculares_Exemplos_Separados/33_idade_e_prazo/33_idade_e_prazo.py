"""Exemplo 33 — Comparar idade do frame com o prazo

COMENTÁRIO GERAL
Executa dois streams curtos e plota envio→decode e captura→decode. Traça o prazo ilustrativo de 33,33 ms e conta frames recebidos acima dele.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 33_idade_e_prazo.py --sem-gui
Colab: abrir 33_idade_e_prazo.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
A média de latência não descreve sozinha a atualidade da imagem. O timestamp de captura ocorre antes da leitura do vídeo; não mede exposição de uma câmera física.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    comparar(out,parametros_stream(a))
    fig,axs=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    for ax,p in zip(axs,['udp','tcp']):
        rows=ler_csv(out/p/'recebidos.csv');ids=[int(r['frame_id']) for r in rows]
        ax.plot(ids,[float(r['latencia_ms']) for r in rows],color='black',label='Envio → decode')
        ax.plot(ids,[float(r['idade_frame_ms']) for r in rows],color='0.5',linestyle='--',label='Captura → decode')
        ax.axhline(1000/30,color='0.4',linestyle=':',label='Prazo ilustrativo')
        ax.set(title=p.upper(),xlabel='Frame');ax.legend(fontsize=9)
    axs[0].set_ylabel('Tempo (ms)');salvar_fig(fig,out/'idade_prazo.png')
