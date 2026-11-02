"""Exemplo 26 — Abrir o vídeo e ler o primeiro frame

COMENTÁRIO GERAL
Verifica a abertura de VideoCapture, lê FPS nominal, número de frames e dimensões e exibe o primeiro frame. A captura é liberada mesmo em caso de erro.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/26_abrir_video.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/26_abrir_video/.

O QUE OBSERVAR
O FPS informado pelo arquivo é diferente da capacidade de processamento do algoritmo.
"""
import cv2
from apoio import (iniciar, abrir_video, salvar_json, painel)

a,out=iniciar(__file__)
cap=abrir_video(a.video)
try:
 m={'fps_nominal':cap.get(cv2.CAP_PROP_FPS),'frames_declarados':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    'largura':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'altura':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
 print(m);salvar_json(out/'video.json',m)
 ok,frame=cap.read()
 if not ok: raise ValueError('Primeiro frame não decodificado.')
 painel([frame],['Primeiro frame do vídeo real'],out/'primeiro_frame.png',a)
finally:
 cap.release()
