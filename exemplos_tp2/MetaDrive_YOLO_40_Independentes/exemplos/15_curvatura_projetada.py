"""
Exemplo 15/40 — Curvatura na geometria da imagem

O QUE FAZ:
Calcula um raio aparente em pixels a partir dos polinomios.

COMO FUNCIONA, PASSO A PASSO:
Usa derivadas x primeira e segunda para raio geometrico.
O codigo pode ser aberto e executado individualmente. Suas funcoes estao
neste mesmo arquivo, sem importacao de um modulo de apoio do curso.
O terminal apresenta dados concretos e as imagens/relatorios ficam na pasta
resultados/15_curvatura_projetada/ quando a operacao estiver concluida.

IMPORTANCIA PARA SISTEMAS AUTONOMOS:
Raio no plano da imagem nao e curvatura fisica da estrada.
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

def detectar_faixas(frame):
    """Adaptacao DIDATICA do detector classico do TP1 para resolucoes variaveis.

    Pipeline: redimensionamento -> HLS + Sobel -> ROI -> Canny -> HoughLinesP
    -> separacao esquerda/direita -> ajuste quadratico -> desvio em pixels.
    A homografia e as escalas em metros do TP1 nao sao reutilizadas: requerem
    calibracao especifica da camera MetaDrive. Na ausencia de duas faixas
    validas, devolve None em vez de inventar desvios e curvaturas.
    """
    h,w=frame.shape[:2]
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
    l,s=hsv[:,:,1],hsv[:,:,2]
    branco=(l>=165)&(s<=125)
    amarelo=(s>=90)&(l>=65)
    y=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(y,(5,5),0)
    gx=np.abs(cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3))
    lim=np.percentile(gx,85)
    grad=(gx>max(16.,lim)).astype(np.uint8)*255
    mask=((branco|amarelo).astype(np.uint8)*255)
    mask=cv2.bitwise_or(mask,grad)
    roi=np.zeros_like(mask)
    pol=np.array([[(0,h-1),(int(.43*w),int(.60*h)),(int(.57*w),int(.60*h)),(w-1,h-1)]],np.int32)
    cv2.fillPoly(roi,pol,255)
    mask=cv2.bitwise_and(mask,roi)
    bordas=cv2.Canny(mask,60,150)
    linhas=cv2.HoughLinesP(bordas,1,np.pi/180,35,minLineLength=24,maxLineGap=30)
    annotated=frame.copy()
    cv2.polylines(annotated,pol,True,(100,100,100),1)
    pontos={'esquerda': [], 'direita': []}
    if linhas is not None:
        for x1,y1,x2,y2 in linhas.reshape(-1,4):
            dy=y2-y1;dx=x2-x1
            if abs(dy)<12:continue
            m=dx/float(dy)
            if abs(m)<.10 or abs(m)>2.0:continue
            ym=(y1+y2)/2; xm=(x1+x2)/2
            lado='esquerda' if (xm < w/2 and m < 0) else ('direita' if (xm > w/2 and m > 0) else None)
            if lado:
                pontos[lado].extend([(y1,x1),(y2,x2)])
                cv2.line(annotated,(x1,y1),(x2,y2),(20,210,255),2)
    ajustes={}
    for lado, dados in pontos.items():
        if len(dados)>=6:
            ar=np.array(dados,dtype=float)
            ys,xs=ar[:,0],ar[:,1]
            if np.ptp(ys) >= 20:
                ajuste=np.polyfit(ys,xs,2)
                ajustes[lado]=ajuste
                yy=np.arange(int(.61*h),h,2)
                xx=np.clip(np.polyval(ajuste,yy),0,w-1).astype(int)
                pts=np.column_stack([xx,yy]).astype(np.int32)
                cv2.polylines(annotated,[pts],False,(0,255,30),3)
    desvio_px=None; curvatura_px=None
    if len(ajustes)==2:
        c1=np.polyval(ajustes['esquerda'],h-1)
        c2=np.polyval(ajustes['direita'],h-1)
        if (c2-c1) > .15*w:
            centro=(c1+c2)/2
            desvio_px=float(w/2-centro)
            y0=float(.85*h)
            radios=[]
            for fit in ajustes.values():
                a,b,_=fit
                if abs(a)>1e-8:
                    radios.append((1+(2*a*y0+b)**2)**1.5/(2*abs(a)))
            curvatura_px=float(np.mean(radios)) if radios else None
            cv2.line(annotated,(int(centro),h-1),(int(w/2),h-1),(255,255,255),3)
    status=(f'Desvio: {desvio_px:+.1f} px' if desvio_px is not None else 'Faixas insuficientes')
    cv2.rectangle(annotated,(0,0),(w,36),(15,15,15),-1)
    cv2.putText(annotated,status,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(230,230,230),2)
    return {'imagem':annotated,'mascara':mask,'bordas':bordas,
            'desvio_px':desvio_px,'curvatura_raio_px':curvatura_px,
            'status_valido':desvio_px is not None}

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
    frame=imagem_demo();cinza=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    res=detectar_faixas(frame)
    salvar('resultado_15.png',res['imagem'])
    print('Desvio em pixels:',res['desvio_px'],'raio projetado em pixels:',res['curvatura_raio_px'])
    print('Leitura critica: NENHUMA conversao para metros sem calibracao.')
