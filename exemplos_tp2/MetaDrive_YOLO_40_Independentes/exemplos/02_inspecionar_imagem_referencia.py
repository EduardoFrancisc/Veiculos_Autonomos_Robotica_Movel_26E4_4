"""
Exemplo 02/40 — Inspecionar uma fotografia veicular

O QUE FAZ:
Abre a imagem KITTI e exibe largura, altura, canais e amostras de pixels.

COMO FUNCIONA, PASSO A PASSO:
Usa imread, shape e imwrite para inspecao controlada.
O codigo pode ser aberto e executado individualmente. Suas funcoes estao
neste mesmo arquivo, sem importacao de um modulo de apoio do curso.
O terminal apresenta dados concretos e as imagens/relatorios ficam na pasta
resultados/02_inspecionar_imagem_referencia/ quando a operacao estiver concluida.

IMPORTANCIA PARA SISTEMAS AUTONOMOS:
Uma imagem KITTI nao deve ser apresentada como captura do MetaDrive.
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
    im=imagem_demo();print('shape:',im.shape,'tipo:',im.dtype,'pixel central BGR:',im[im.shape[0]//2,im.shape[1]//2]);salvar('amostra_kitti.png',im)
