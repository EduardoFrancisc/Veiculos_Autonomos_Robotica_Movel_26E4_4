"""Exemplo 36 — Projeto integrador com os itens A e B

COMENTÁRIO GERAL
Executa o CAN com dois processos e os dois streams de 320 frames por protocolo, em ordem TCP→UDP. Reúne os resultados em um relatório para o aluno interpretar. Não depende de saídas de nenhum exercício anterior.

O QUE INSTALAR
Python 3.10 ou superior. Na pasta do pacote, executar:
python -m pip install -r requirements.txt
Bibliotecas: python-can, OpenCV, NumPy e Matplotlib. No Colab, usar a célula
de instalação do notebook (requirements-colab.txt / OpenCV headless).
socket, time, struct, multiprocessing e outras bibliotecas padrão não exigem pip.
Os códigos usam CPU. Mantenha os arquivos que estão nesta pasta do exemplo.

COMO EXECUTAR
VS Code: python 36_projeto_integrador.py --sem-gui
Colab: abrir 36_projeto_integrador.ipynb, enviar o ZIP dos exemplos separados e executar as células.
Não é necessário executar um exemplo anterior; entradas e saídas são próprias.
Streaming: --mostrar abre a janela no desktop; --log-cada 1 imprime cada frame.
Ao testar pelo item B, manter ao menos 300 frames por protocolo.

O QUE OBSERVAR
Compare a execução em ordem invertida com o exemplo 28. Revise ambiente, definições e limitações antes de transformar o texto em uma conclusão da turma.
"""
from apoio import *
from can_laboratorio import *
from rede_laboratorio import *
import json, random, socket, struct, threading, time, zlib
from uuid import uuid4

if __name__=='__main__':
    a,out=iniciar(__file__,frames=320,ciclos=50)
    if a.frames<300:raise ValueError('Projeto integrador: ao menos 300 frames por protocolo.')
    res_can=can_processos(out/'can',ciclos=a.ciclos,hz=a.hz)
    res_stream=comparar(out/'stream',parametros_stream(a),ordem=('tcp','udp'))
    salvar_json(out/'resultado_integrador.json',dict(can=res_can,stream=res_stream))
    texto='# Relatório integrador — protocolos veiculares\n\n'
    texto+=f"CAN: {res_can['total_mensagens']} mensagens, {a.hz} Hz por ID, dois processos e ponte IPC.\n\n"
    texto+='Evidências do item A: can/mensagens_can.csv, can/series_can.png, can/processos.json.\n\n'
    texto+=(out/'stream/relatorio_streaming.md').read_text(encoding='utf-8')
    (out/'relatorio_integrador.md').write_text(texto,encoding='utf-8')
