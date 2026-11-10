"""
Exemplo 27/40 — Medir tempo de inferencia YOLO

O QUE FAZ:
Realiza aquecimento e cronometra 10 inferencias numa imagem.

COMO FUNCIONA, PASSO A PASSO:
perf_counter e sincronizacao CUDA quando disponivel.
O codigo pode ser aberto e executado individualmente. Suas funcoes estao
neste mesmo arquivo, sem importacao de um modulo de apoio do curso.
O terminal apresenta dados concretos e as imagens/relatorios ficam na pasta
resultados/27_latencia_yolo/ quando a operacao estiver concluida.

IMPORTANCIA PARA SISTEMAS AUTONOMOS:
Excluir download e inicializacao do tempo de inferencia.
O exercicio permite discutir a diferenca entre observacao visual, estimativa
por processamento de imagem e medida fisica validada. Nem toda classe COCO
representa uma placa, e dados de pixels nao devem receber unidade fisica sem
calibracao da camera e da geometria da pista.

DEPENDENCIAS: consulte requirements.txt; para cenas reais simuladas, e necessario
MetaDrive com renderizacao grafica funcional. Em Google Colab, configurar EGL ou
Xvfb; use_render=True e apropriado para desktop com janela.
EXERCICIO AO ALUNO: modifique os parametros, repita com a mesma semente,
compare o resultado e descreva a mudanca visual e temporal observada.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "dados"
RESULTADOS = ROOT / "resultados" / Path(__file__).stem
RESULTADOS.mkdir(parents=True, exist_ok=True)

def imagem_demo():
    """Fotografia externa fornecida com o pacote; NAO e frame do MetaDrive."""
    caminho = DADOS / 'frontal_kitti.png'
    frame = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f'Imagem nao encontrada: {caminho}')
    return cv2.resize(frame, (640, 360))


def salvar(nome, imagem):
    caminho = RESULTADOS / nome
    if not cv2.imwrite(str(caminho), imagem):
        raise OSError(f'Falha ao gravar {caminho}')
    print('Imagem salva:', caminho)
    return caminho


def salvar_csv(nome, registros):
    caminho=RESULTADOS/nome
    if not registros:
        raise ValueError('Nao ha registros a salvar; o teste falhou antes do primeiro step.')
    colunas=list(registros[0].keys())
    with caminho.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=colunas);w.writeheader();w.writerows(registros)
    print('CSV:', caminho, '| registros:', len(registros))
    return caminho


def escrever_texto(nome, texto):
    p=RESULTADOS/nome;p.write_text(texto,encoding='utf-8')
    print('Relatorio:',p)

CLASSES_TRAFEGO={'person','bicycle','car','motorcycle','bus','truck','traffic light','stop sign'}

def carregar_yolo():
    """Carrega YOLOv8n treinado em COCO; primeiro uso pode baixar pesos (internet)."""
    try:from ultralytics import YOLO
    except ImportError as e:raise RuntimeError('Execute pip install ultralytics') from e
    return YOLO('yolov8n.pt')


def detectar_yolo(modelo, frame, imgsz=640, conf=0.25, memoria=None):
    """Inferencia YOLO, classes COCO de interesse e caixas rotuladas.

    COCO contem 'stop sign' e 'traffic light', mas nao todas as placas de transito.
    Deteccoes vazias sao validas. Em imagens MetaDrive ha domain gap visual.
    """
    resultados=modelo.predict(source=frame,imgsz=imgsz,conf=conf,verbose=False)
    det=resultados[0]
    saida=frame.copy(); contagem=Counter()
    if memoria is not None: memoria.clear()
    if det.boxes is not None:
        boxes=det.boxes.xyxy.cpu().numpy()
        cls=det.boxes.cls.cpu().numpy().astype(int)
        probs=det.boxes.conf.cpu().numpy()
        for coords,id_,p in zip(boxes,cls,probs):
            nome=str(det.names[int(id_)])
            if nome not in CLASSES_TRAFEGO:continue
            contagem[nome]+=1
            x1,y1,x2,y2=map(int,coords)
            if memoria is not None:memoria.append((x1,y1,x2,y2,nome,float(p)))
            cv2.rectangle(saida,(x1,y1),(x2,y2),(30,220,80),2)
            cv2.putText(saida,f'{nome} {float(p):.2f}',(x1,max(20,y1-7)),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,(30,220,80),2)
    return saida,dict(contagem)

def args_comuns():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else 'Exemplo independente')
    parser.add_argument('--render',action='store_true',help='Usar janela desktop. No Colab, omitir.')
    parser.add_argument('--steps',type=int,default=None,help='Substitui o numero de steps didatico')
    parser.add_argument('--imgsz',type=int,default=640,help='Resolucao de inferencia YOLO')
    parser.add_argument('--skip',type=int,default=1,help='YOLO a cada N frames')
    parser.add_argument('--offline',action='store_true',help='Somente imagem de amostra; NAO e benchmark do MetaDrive')
    return parser.parse_args()


if __name__ == "__main__":
    a=args_comuns()
    modelo=carregar_yolo();frame=imagem_demo()

    for _ in range(2):detectar_yolo(modelo,frame)
    t=[]
    for i in range(10):
        tic=time.perf_counter();detectar_yolo(modelo,frame)
        try:
            import torch
            if torch.cuda.is_available():torch.cuda.synchronize()
        except ImportError:pass
        t.append(1000*(time.perf_counter()-tic))
    print('Latencia YOLO (ms) media:',round(np.mean(t),2),'p95:',round(np.percentile(t,95),2))
