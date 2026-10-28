"""Exemplo 04 — Decodificar e conferir a escala

COMENTÁRIO GERAL
Reconstrói mensagens e aplica struct.unpack e a escala definida para cada ID. Inclui temperatura negativa para mostrar por que esse sinal usa inteiro com sinal.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 04_decodificar_sensores.py --sem-gui
Colab: abrir 04_decodificar_sensores.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
A mesma sequência de bytes pode significar valores diferentes quando o formato, o endian ou a escala estão errados.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=60,ciclos=20)
    for i,valor in [(0x100,48.25),(0x101,2350),(0x102,-5.5)]:
        msg=can.Message(arbitration_id=i,is_extended_id=False,
                        data=codificar(i,valor,3,300),check=True)
        nome,dec,unidade,seq,t_ms=decodificar(msg)
        print(nome,dec,unidade,'contador=',seq,'tempo=',t_ms)
        assert abs(dec-valor) <= .01
