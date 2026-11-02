"""Exemplo 31 — Relacionar tamanho de datagrama e chance de remontagem

COMENTÁRIO GERAL
Mede quantos fragmentos o JPEG de entrada exige para três tamanhos de datagrama. Plota um modelo matemático de sucesso do frame sob perdas independentes por fragmento.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 31_tamanho_datagrama.py --sem-gui
Colab: abrir 31_tamanho_datagrama.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
O gráfico é um modelo ilustrativo, não um benchmark. A independência das perdas pode falhar em redes com perdas em rajadas.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    _,buf=cv2.imencode('.jpg',quadro_inicial(a),[cv2.IMWRITE_JPEG_QUALITY,a.qualidade])
    p=np.linspace(0,.03,100);fig,ax=plt.subplots(figsize=(9,4.5));rows=[]
    for tamanho,ls in [(512,':'),(1200,'-'),(1400,'--')]:
        n=len(fragmentar(buf.tobytes(),0,1,0,tamanho))
        ax.plot(p*100,100*(1-p)**n,label=f'{tamanho} bytes; {n} fragmentos',color='black',linestyle=ls)
        rows.append(dict(datagrama_bytes=tamanho,jpeg_bytes=buf.nbytes,fragmentos=n))
    ax.set(xlabel='Probabilidade de perda por fragmento (%)',ylabel='Sucesso do frame no modelo (%)')
    ax.legend();salvar_fig(fig,out/'modelo_fragmentacao.png');salvar_csv(out/'fragmentos.csv',rows)
    print('Modelo ilustrativo:',rows)
