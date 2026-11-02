"""Exemplo 16 — Estimar volume de dados de uma câmera

COMENTÁRIO GERAL
Lê um frame do vídeo, mede o tamanho real do JPEG e compara com o tamanho da matriz BGR sem compressão. Calcula taxa de payload para 30 FPS, antes de cabeçalhos e retransmissões.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 16_volume_dados.py --sem-gui
Colab: abrir 16_volume_dados.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
CAN e Ethernet não pertencem ao mesmo nível que TCP e UDP. Compare meio/enlace e transporte em decisões separadas.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    frame=quadro_inicial(a)
    ok,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,a.qualidade])
    rows=[]
    for nome,bytes_ in [('BGR sem compressão',frame.nbytes),('JPEG',buf.nbytes)]:
        taxa=bytes_*8*a.fps/1e6
        print(nome,bytes_,'bytes/frame;',taxa,'Mbit/s de payload')
        rows.append(dict(representacao=nome,bytes_frame=bytes_,fps=a.fps,payload_mbit_s=taxa))
    salvar_csv(out/'volume_dados.csv',rows)
