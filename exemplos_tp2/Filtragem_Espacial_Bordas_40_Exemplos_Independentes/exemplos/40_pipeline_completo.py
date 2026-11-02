"""
EXEMPLO 40 - Pipeline final: operadores, cenários, tempo, F1 e ângulos

O QUE ESTE CÓDIGO FAZ
Integra Sobel, LoG, duas configurações Canny e DoG e mede desempenho nas três condições.

COMO O CÓDIGO FUNCIONA
Calcula pixels ativados, tempo mediano, precisão/recall/F1 tolerantes a 2 pixels e respostas
direcionais.

POR QUE É IMPORTANTE NA DIREÇÃO AUTÔNOMA
Permite justificar uma decisão experimental limitada ao dataset controlado sem afirmar
superioridade universal.

COMO EXPLICAR AOS ALUNOS
Este exemplo representa uma etapa pequena de um pipeline de percepção. Observe a
imagem inicial, os valores numéricos apresentados no terminal e as imagens geradas.
Compare alterações de parâmetros mantendo as demais condições constantes.

ATIVIDADE PRÁTICA
Repita em imagens reais anotadas antes de recomendar aplicação veicular.

COMO EXECUTAR
python exemplos/40_pipeline_completo.py

SAÍDA
Uma ou mais figuras ou métricas serão salvas automaticamente em saidas/.
O programa pode ser executado diretamente: as funções estão neste próprio arquivo.
"""

import argparse
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Cada exemplo contém suas próprias funções; NÃO utiliza apoio.py.
# A pasta dados guarda apenas imagens de entrada. É permitido executar
# os exemplos em qualquer ordem, diretamente ou no Google Colab.
BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados"
RESULTADOS = BASE / "saidas" / Path(__file__).stem
RESULTADOS.mkdir(parents=True, exist_ok=True)


def ler(nome="pista_bev.png", cinza=False):
    """Abre uma imagem com verificação explícita de existência."""
    p = DADOS / nome
    imagem = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE if cinza else cv2.IMREAD_COLOR)
    if imagem is None:
        raise FileNotFoundError(f"Imagem ausente: {p}. Execute o código com o ZIP extraído.")
    return imagem


def cinza(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def sombra(img):
    """Perturbação controlada: sombra artificial em faixa diagonal da estrada."""
    a = img.astype(np.float32).copy()
    h, w = img.shape[:2]
    m = np.zeros((h,w),np.uint8)
    pts = np.array([[0,int(h*.16)],[w,int(h*.36)],[w,int(h*.59)],[0,int(h*.39)]],np.int32)
    cv2.fillConvexPoly(m,pts,255)
    a[m>0]*=.38
    a += np.random.default_rng(41).normal(0,2.5,a.shape)
    return np.clip(a,0,255).astype(np.uint8)


def molhado(img):
    """CENÁRIO SINTÉTICO, não um teste com asfalto molhado real.
    Refrações/reflexos e variação tonal aproximam certos efeitos visuais.
    """
    h,w=img.shape[:2]
    a=(img.astype(np.float32)*.68).astype(np.uint8)
    overlay=np.zeros_like(a)
    cv2.ellipse(overlay,(int(w*.50),int(h*.73)),(int(w*.39),int(h*.1)),0,0,360,(175,181,189),-1)
    cv2.line(overlay,(int(w*.27),int(h*.57)),(int(w*.76),int(h*.89)),(255,235,226),6)
    cv2.line(overlay,(int(w*.47),int(h*.62)),(int(w*.66),int(h*.97)),(225,224,240),4)
    a=cv2.addWeighted(a,.83,overlay,.17,0)
    noise=np.random.default_rng(27).normal(0,6,a.shape)
    return np.clip(a.astype(np.float32)+noise,0,255).astype(np.uint8)


def selecionar_cena(nome="referencia"):
    img=ler()
    if nome == "sombra": return sombra(img)
    if nome == "molhado": return molhado(img)
    return img


def normalize8(x):
    x=np.asarray(x,dtype=np.float64)
    maxv=float(np.max(np.abs(x)))
    return np.uint8(np.clip((np.abs(x)/max(maxv,1e-9))*255,0,255))


def mostrar(paineis,titulos,nome="painel.png",colunas=3):
    """Salva painel, sem depender de janelas gráficas (compatível com Colab)."""
    n=len(paineis); linhas=(n+colunas-1)//colunas
    fig,ax=plt.subplots(linhas,colunas,figsize=(5*colunas,3.45*linhas),squeeze=False)
    for i, (im,t) in enumerate(zip(paineis,titulos)):
        a=ax.flat[i]
        if im.ndim==2: a.imshow(im,cmap="gray",vmin=0,vmax=255)
        else: a.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        a.set_title(t,fontsize=11);a.axis("off")
    for a in ax.flat[n:]:a.axis("off")
    fig.tight_layout()
    p=RESULTADOS/nome
    fig.savefig(p,dpi=120,bbox_inches="tight");plt.close(fig)
    print("Figura gerada:",p)
    return p


def operadores(img):
    """Operadores com parâmetros fixos para uma comparação reproduzível.
    Sobel, LoG e DoG não são naturalmente binários; adotamos limiares
    explícitos para possibilitar a contagem de pixels. Os limiares não
    são universalmente ótimos; devem ser calibrados em outras cenas.
    """
    g=cinza(img)
    sx=cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3)
    sy=cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)
    mag=np.hypot(sx,sy)
    log=cv2.Laplacian(cv2.GaussianBlur(g,(5,5),1.2),cv2.CV_64F,ksize=3)
    ga=cv2.GaussianBlur(g,(0,0),1.0).astype(np.float32)
    gb=cv2.GaussianBlur(g,(0,0),2.2).astype(np.float32)
    dog=np.abs(ga-gb)
    return {
        "Sobel":(np.uint8(mag>=95)*255,mag),
        "LoG":(np.uint8(np.abs(log)>=35)*255,np.abs(log)),
        "Canny 50/110":(cv2.Canny(g,50,110),None),
        "Canny 100/200":(cv2.Canny(g,100,200),None),
        "DoG":(np.uint8(dog>=7.5)*255,dog),
    }


def avaliacao(pred,gt,tolerancia=2):
    """Precisão/recall com tolerância geométrica: cada predição é aceita
    quando fica a até 'tolerancia' px da borda verdadeira, e vice-versa.
    Essa aproximação é didática e NÃO substitui avaliação em dados reais.
    """
    p=pred>0;g=gt>0
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tolerancia+1,2*tolerancia+1))
    gd=cv2.dilate(g.astype(np.uint8),k)>0
    pd=cv2.dilate(p.astype(np.uint8),k)>0
    precisao=float(np.sum(p & gd)/max(int(p.sum()),1))
    revocacao=float(np.sum(g & pd)/max(int(g.sum()),1))
    f1=2*precisao*revocacao/max(precisao+revocacao,1e-12)
    return precisao,revocacao,f1


def banco_direcional(img):
    """Respostas de gradiente (direção normal à borda), NÃO direção
    longitudinal da faixa. Portanto faixa vertical tipicamente gera
    resposta maior perto de 0 grau (gradiente horizontal).
    """
    g=cinza(img).astype(np.float64)
    kx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float64)
    ky=kx.T
    resp={}
    for graus in (0,45,90,135):
        th=np.deg2rad(graus)
        k=np.cos(th)*kx+np.sin(th)*ky
        resp[graus]=np.abs(cv2.filter2D(g,cv2.CV_64F,k,borderType=cv2.BORDER_REFLECT_101))
    return resp


def tabela(rows,columns):
    print(" | ".join(columns))
    print("-"*88)
    for row in rows:print(" | ".join(str(row.get(c,"")) for c in columns))
    with open(RESULTADOS/"metricas.csv","w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=columns);wr.writeheader();wr.writerows(rows)


# ---------------- EXPERIMENTO DESTE EXEMPLO ----------------
gt=ler("bordas_gt.png",cinza=True);ref=ler();rows=[];paineis=[];tit=[]
for nome,i in [("normal",ref),("sombra",sombra(ref)),("reflexos",molhado(ref))]:
 op=operadores(i);paineis.append(i);tit.append(nome)
 for operador,(edge,_) in op.items():
  # Cronometria inclui o cálculo integral do conjunto como custo aproximado por operador isolado? NÃO.
  # Reexecuta função equivalente individual para medir etapa corretamente.
  g=cinza(i)
  fns={"Sobel":lambda:np.hypot(cv2.Sobel(g,cv2.CV_64F,1,0),cv2.Sobel(g,cv2.CV_64F,0,1)),"LoG":lambda:cv2.Laplacian(cv2.GaussianBlur(g,(5,5),1.2),cv2.CV_64F),"Canny 50/110":lambda:cv2.Canny(g,50,110),"Canny 100/200":lambda:cv2.Canny(g,100,200),"DoG":lambda:cv2.absdiff(cv2.GaussianBlur(g,(0,0),1.0),cv2.GaussianBlur(g,(0,0),2.2))}
  fn=fns[operador];fn();tempos=[]
  for _ in range(6):t=time.perf_counter();fn();tempos.append((time.perf_counter()-t)*1000)
  pr,re,f1=avaliacao(edge,gt,tolerancia=2)
  rows.append(dict(cenario=nome,operador=operador,pixels_ativos=int(np.count_nonzero(edge)),mediana_ms=round(float(np.median(tempos)),3),precisao=round(pr,3),recall=round(re,3),f1=round(f1,3)))
  paineis.append(edge);tit.append(operador)
tabela(rows,["cenario","operador","pixels_ativos","mediana_ms","precisao","recall","f1"])
mostrar(paineis,tit,"comparacao_3_cenarios.png",colunas=6)
# Critério explícito de melhor resultado neste experimento: F1 no cenário degradado.
for c in ("sombra","reflexos"):
 x=max((r for r in rows if r["cenario"]==c),key=lambda r:r["f1"])
 print(f"MELHOR F1 em {c}: {x['operador']} (F1={x['f1']})")
r=banco_direcional(ref);g=ler("bordas_gt.png",cinza=True)>0
angles={k:round(float(v[g].mean()),3) for k,v in r.items()}
print("Resposta direcional média nas bordas da faixa:",angles)
print("Ângulo de maior resposta:",max(angles,key=angles.get))
with open(RESULTADOS/"conclusao.txt","w",encoding="utf8") as f:
 for c in ("sombra","reflexos"):
  x=max((r for r in rows if r["cenario"]==c),key=lambda r:r["f1"])
  f.write(f"Cenário sintético {c}: maior F1 = {x['operador']} ({x['f1']}); não é recomendação universal.\n")
 f.write(f"Direção de GRADIENTE mais forte nas bordas de faixa: {max(angles,key=angles.get)} graus.\n")
 f.write("A comparação exige imagens reais anotadas, múltiplas cenas, mesma ROI, métricas de qualidade e perfil de latência.\n")

