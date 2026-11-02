"""
EXEMPLO 40 - Pipeline Completo

OBJETIVO DA AULA
Executar detecção geométrica de sinais, linhas e keypoints e gerar feed de fluxo LK.

COMO FUNCIONA, ETAPA POR ETAPA
Roda três configurações Hough sobre cinco cenas, compara Harris/Shi em N=200, traça HoughLinesP, grava MP4 Lucas–Kanade e exporta métricas. O programa é inteiramente autônomo: importa OpenCV e
NumPy, identifica a pasta onde se encontra, lê os arquivos da pasta dados,
executa as operações descritas abaixo e salva os resultados em sua própria
pasta sob saidas/. Nenhuma função é importada de outro arquivo Python.

POR QUE ISSO IMPORTA NA PERCEPÇÃO VEICULAR
É a integração didática final; o escore circular é apenas heurístico e velocidade física requer calibração. A câmera entrega intensidades de pixels; nenhuma dessas
rotinas, isoladamente, compreende semanticamente a cena ou garante
funcionamento sob qualquer iluminação, chuva ou oclusão.

COMO EXECUTAR
    python exemplos/40_pipeline_completo.py

EXPERIMENTO PARA OS ALUNOS
Altere um parâmetro de cada vez, compare visualmente as imagens produzidas
e registre as medidas impressas no terminal. Compare também pelo menos duas
cenas. Não confunda número de detecções com acerto sem ter anotações de
referência validadas. Os arquivos são salvos automaticamente; não precisa
abrir janelas GUI, sendo adequado ao Google Colab.

CUIDADOS E LIMITAÇÕES
As fotografias de fundo são reais, mas os sinais circulares são sobrepostos
artificialmente para estudo. O vídeo de fluxo óptico é uma simulação criada
com transformação dos frames de uma cena real. O indicador de apoio do anel
é heurístico, NÃO uma confiança probabilística fornecida pelo OpenCV. A
velocidade estimada em pixels/frame NÃO representa, sem calibração e
profundidade, a velocidade real de um obstáculo em metros por segundo.
"""

import cv2
import numpy as np
import time
import csv
from pathlib import Path

# Cada programa contém integralmente as funções que usa: sem arquivo apoio.py.
BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / 'dados'
SAIDA = BASE / 'saidas' / Path(__file__).stem
SAIDA.mkdir(parents=True, exist_ok=True)

def carregar(nome='rua_01.jpg'):
    imagem = cv2.imread(str(DADOS / nome), cv2.IMREAD_COLOR)
    if imagem is None:
        raise FileNotFoundError(f'Imagem ausente: {DADOS / nome}')
    return imagem

def salvar(nome, imagem):
    destino = SAIDA / nome
    if not cv2.imwrite(str(destino), imagem):
        raise IOError(f'Não foi possível salvar {destino}')
    print('Imagem salva:', destino)
    return destino

def painel(imagens, nomes, arquivo='painel.jpg', colunas=3):
    # Redimensiona cada célula e cria o mosaico de forma legível sem GUI.
    largura, altura = 500, 300
    blocos = []
    for img, nome in zip(imagens, nomes):
        if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)
        cv2.rectangle(vis, (0, 0), (largura, 32), (14, 25, 41), -1)
        cv2.putText(vis, nome[:54], (12, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    .63, (245, 245, 245), 2, cv2.LINE_AA)
        blocos.append(vis)
    while len(blocos) % colunas:
        blocos.append(np.full((altura, largura, 3), 240, np.uint8))
    linhas = [np.hstack(blocos[i:i + colunas]) for i in range(0, len(blocos), colunas)]
    return salvar(arquivo, np.vstack(linhas))

def preparar_circulos(img):
    # O HoughCircles opera em 8 bits e costuma responder melhor após suavização.
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cinza, (7, 7), 1.5)

def confianca_heuristica(gray, x, y, r):
    # NÃO é uma probabilidade da HoughCircles. Calcula a fração do anel
    # com bordas Canny próximas, gerando somente um escore heurístico [0,1].
    e = cv2.Canny(gray, 80, 160)
    dilatada = cv2.dilate(e, np.ones((3,3),np.uint8), iterations=1)
    angulos = np.linspace(0, 2*np.pi, 96, endpoint=False)
    xs = np.clip(np.rint(x+r*np.cos(angulos)).astype(int),0,e.shape[1]-1)
    ys = np.clip(np.rint(y+r*np.sin(angulos)).astype(int),0,e.shape[0]-1)
    return float(np.mean(dilatada[ys, xs] > 0))

def detectar_circulos(imagem, dp=1.2, minDist=35, param1=135, param2=32,
                      minRadius=12, maxRadius=80, somente_vermelhos=False):
    gray = preparar_circulos(imagem)
    t0 = time.perf_counter()
    raw = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                           param1=param1, param2=param2,
                           minRadius=minRadius, maxRadius=maxRadius)
    ms = (time.perf_counter()-t0)*1000
    achados = [] if raw is None else np.rint(raw[0]).astype(int).tolist()
    if somente_vermelhos and achados:
        # Filtro geométrico + cromático para exemplos de placas vermelhas.
        # Não usar este filtro para placas azuis ou sinais sem borda vermelha.
        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv,(0,65,70),(13,255,255)) | cv2.inRange(hsv,(168,65,70),(179,255,255))
        def apoio_vermelho(x,y,r):
            aa=np.linspace(0,2*np.pi,120,endpoint=False)
            valores=[]
            for frac in [.80,.87,.94,1.01,1.08]:
                xs=np.clip(np.rint(x+frac*r*np.cos(aa)).astype(int),0,mascara.shape[1]-1)
                ys=np.clip(np.rint(y+frac*r*np.sin(aa)).astype(int),0,mascara.shape[0]-1)
                valores.append(float(np.mean(mascara[ys,xs]>0)))
            return max(valores)
        candidatos=sorted([(apoio_vermelho(x,y,r),(x,y,r)) for x,y,r in achados],reverse=True)
        retidos=[]
        for score,(x,y,r) in candidatos:
            if score<.16: continue
            if any((x-u)**2+(y-v)**2 < max(15,int(min(r,rr)*.75))**2 for u,v,rr in retidos):continue
            retidos.append((x,y,r))
        achados=retidos
    return gray, achados, ms

def desenhar_circulos(img, circulos, gray=None):
    vis = img.copy()
    for j,(x,y,r) in enumerate(circulos,1):
        cv2.circle(vis,(x,y),r,(0,240,0),2,cv2.LINE_AA)
        cv2.circle(vis,(x,y),3,(0,255,255),-1,cv2.LINE_AA)
        conf = confianca_heuristica(gray if gray is not None else preparar_circulos(img),x,y,r)
        cv2.putText(vis,f'{j} s={conf:.2f}',(max(2,x-r),max(20,y-r-8)),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(0,250,0),2,cv2.LINE_AA)
    return vis

def desenhar_linhas(img, low=60, high=150):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),low,high)
    roi=np.zeros_like(canny)
    h,w=canny.shape
    pol=np.array([[(0,h),(w,h),(int(.68*w),int(.60*h)),(int(.32*w),int(.60*h))]],np.int32)
    cv2.fillPoly(roi,pol,255)
    canny=cv2.bitwise_and(canny,roi)
    linhas=cv2.HoughLinesP(canny,1,np.pi/180,threshold=28,
                           minLineLength=35,maxLineGap=45)
    vis=img.copy()
    n=0
    if linhas is not None:
        for x1,y1,x2,y2 in linhas[:,0]:
            if abs(y2-y1)<7: continue
            cv2.line(vis,(x1,y1),(x2,y2),(0,220,240),2,cv2.LINE_AA)
            n+=1
    return vis,canny,n

def detectar_harris(img,max_pontos=200):
    gray=np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    t0=time.perf_counter()
    r=cv2.cornerHarris(gray,blockSize=2,ksize=3,k=.04)
    r=cv2.dilate(r,None)
    # Ordenação decrescente e supressão espacial explícita.
    limite=float(r.max())*.01
    coords=np.argwhere(r>limite)
    coords=sorted(coords,key=lambda yx:float(r[yx[0],yx[1]]),reverse=True)
    pontos=[]
    for y,x in coords:
        if all((x-a)**2+(y-b)**2 >= 7**2 for a,b in pontos):
            pontos.append((int(x),int(y)))
            if len(pontos)>=max_pontos:break
    ms=(time.perf_counter()-t0)*1000
    return pontos,r,ms

def detectar_shi(img,max_pontos=200, quality=.01, distance=7):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    t0=time.perf_counter()
    pontos=cv2.goodFeaturesToTrack(gray,maxCorners=max_pontos,qualityLevel=quality,
                                  minDistance=distance,blockSize=7)
    ms=(time.perf_counter()-t0)*1000
    return (np.empty((0,1,2),np.float32) if pontos is None else pontos.astype(np.float32)),ms

def desenhar_pontos(img,pontos,cor=(20,240,230)):
    vis=img.copy()
    pts=np.asarray(pontos).reshape(-1,2)
    for x,y in pts:cv2.circle(vis,(round(float(x)),round(float(y))),3,cor,-1,cv2.LINE_AA)
    return vis

def ler_frames(qtd=2):
    cap=cv2.VideoCapture(str(DADOS/'trafego_simulado.mp4'))
    if not cap.isOpened():raise FileNotFoundError('Vídeo não encontrado')
    imagens=[]
    for _ in range(qtd):
        ok,fr=cap.read()
        if not ok:break
        imagens.append(fr)
    cap.release()
    if len(imagens)<2:raise RuntimeError('Vídeo insuficiente')
    return imagens

def fluxo_lk(a,b,pontos=None,win=(21,21),niveis=3):
    ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    if pontos is None:
        pontos=cv2.goodFeaturesToTrack(ga,maxCorners=200,
                  qualityLevel=.01,minDistance=8,blockSize=7)
    if pontos is None or not len(pontos):return np.empty((0,2)),np.empty((0,2)),0
    t=time.perf_counter()
    p1,status,erro=cv2.calcOpticalFlowPyrLK(ga,gb,pontos,None,
              winSize=win,maxLevel=niveis,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    ms=(time.perf_counter()-t)*1000
    if p1 is None:return np.empty((0,2)),np.empty((0,2)),ms
    mask=status.reshape(-1)==1
    return pontos.reshape(-1,2)[mask],p1.reshape(-1,2)[mask],ms

def vetores(a,p0,p1):
    vis=a.copy()
    for (x,y),(u,v) in zip(p0,p1):
        cv2.arrowedLine(vis,(round(x),round(y)),(round(u),round(v)),(0,255,255),1,cv2.LINE_AA,tipLength=.36)
        cv2.circle(vis,(round(x),round(y)),2,(20,20,220),-1,cv2.LINE_AA)
    return vis

# ===== PROCEDIMENTO PRINCIPAL DO EXEMPLO =====
configs=[('sensivel',1.,25,90,23),('intermediario',1.2,35,135,29),('restrito',1.5,60,165,40)]
rows=[];pan=[];names=[]
for i in range(1,6):
 img=carregar(f'rua_{i:02d}.jpg');h,r,ht=detectar_harris(img,200);s,st=detectar_shi(img,200)
 vis,edges,linhas=desenhar_linhas(img)
 for nome,dp,md,p1,p2 in configs:
  g,c,t=detectar_circulos(img,dp=dp,minDist=md,param1=p1,param2=p2,somente_vermelhos=True)
  escores=[confianca_heuristica(g,x,y,rr) for x,y,rr in c]
  rows.append([i,nome,len(c),linhas,len(h),len(s),round(t,3),round(ht,3),round(st,3),round(float(np.mean(escores)) if escores else 0,3)])
  print(f'Cena {i}; {nome}: C={len(c)}; HoughLinesP={linhas}; Harris={len(h)}; Shi={len(s)}; tempo Hough={t:.2f}ms')
  for x,y,rr in c:print(f'  centro=({x},{y}); raio={rr}; apoio={confianca_heuristica(g,x,y,rr):.2f}')
  if nome=='intermediario':pan.append(desenhar_circulos(vis,c,g));names.append(f'Cena {i}: C={len(c)}; L={linhas}')
with (SAIDA/'resultados.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['cena','config','circulos','linhas','harris','shi','hough_ms','harris_ms','shi_ms','apoio_medio_heuristico']);w.writerows(rows)
painel(pan,names,'cinco_cenas.jpg',colunas=2)
fs=ler_frames(20);video=SAIDA/'lucas_kanade.mp4'
writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),12,(960,540))
reg=[]
for k in range(len(fs)-1):
 p0,p1,ms=fluxo_lk(fs[k],fs[k+1]);frame=vetores(fs[k+1],p0,p1);writer.write(frame)
 med=float(np.median(np.linalg.norm(p1-p0,axis=1))) if len(p1) else 0.
 reg.append([k+1,len(p1),round(ms,3),round(med,3)])
writer.release()
with (SAIDA/'optical_flow.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['frame','rastreios','lk_ms','deslocamento_mediano_px']);w.writerows(reg)
print('Vídeo:',video,'| frames:',len(reg),'| FIM')
