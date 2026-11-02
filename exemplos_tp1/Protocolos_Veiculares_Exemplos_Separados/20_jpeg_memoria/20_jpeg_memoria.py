"""Exemplo 20 — Codificar e decodificar JPEG em memória

COMENTÁRIO GERAL
Lê um frame real, aplica imencode e imdecode para três qualidades e compara bytes e erro quadrático médio em relação à entrada. Salva as imagens e uma tabela.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 20_jpeg_memoria.py --sem-gui
Colab: abrir 20_jpeg_memoria.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
JPEG é compressão com perdas: diminuir bytes pode alterar bordas e texturas, afetando a percepção.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    frame=quadro_inicial(a);rows=[]
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,q in zip(axs,[95,70,35]):
        ok,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,q])
        decoded=cv2.imdecode(buf,cv2.IMREAD_COLOR)
        mse=float(np.mean((decoded.astype(float)-frame.astype(float))**2))
        rows.append(dict(qualidade=q,jpeg_bytes=buf.nbytes,mse=mse))
        ax.imshow(cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB));ax.set_title(f'Q={q}; {buf.nbytes} bytes');ax.axis('off')
        cv2.imwrite(str(out/f'jpeg_q{q}.png'),decoded)
    salvar_csv(out/'jpeg.csv',rows);salvar_fig(fig,out/'qualidades_jpeg.png')
    print(rows)
