"""
EXEMPLO 40/40 — PIPELINE COMPLETO E INDEPENDENTE

O QUE ESTE CODIGO FAZ: Este programa foi elaborado para reunir mosaico panorâmico e segmentação BEV em um programa autônomo. Cada exercicio
e independente e pode ser executado diretamente, sem apoio.py ou importacao
de outros exemplos. As imagens de entrada ficam na pasta dados.

COMO FUNCIONA: A demonstracao usa ORB, Lowe, RANSAC, warpPerspective, threshold, opening, closing, kernels e componentes. O programa apresenta os parametros,
imprime medidas quantitativas quando sao relevantes e salva resultados na
pasta saidas/40_pipeline_completo/, criada automaticamente.

POR QUE E IMPORTANTE: demonstra as etapas exigidas na atividade sem dependências entre scripts. Na visao veicular, a correspondencia de pontos
auxilia o alinhamento de imagens, enquanto o BEV e a morfologia fornecem
representacoes uteis para encontrar marcacoes de pista.

APLICACAO E CUIDADOS: Este material utiliza duas janelas sobrepostas de uma
mesma fotografia, simulando vistas de cameras. Nao e uma calibracao real
de surround view 360 graus. Para multiplas cameras fisicas sao necessarias
calibracao intrinseca/extrinseca, compensacao de distorcao e tratamento da
paralaxe. A segmentacao BEV trabalha inicialmente com cena controlada.

EXPERIMENTE NA AULA: Troque bev_marcacoes por bev_tp1 e recalibre filtros.

PASSO A PASSO DA EXECUCAO: 1. Detecte ORB nas duas imagens e filtre correspondencias com BFMatcher e Lowe. 2. Estime a homografia com RANSAC, conte inliers e calcule o erro de reprojecao. 3. Projete as duas vistas para um canvas e salve o mosaico. 4. Abra a BEV, aplique threshold adaptativo e opening mais closing. 5. Repita a morfologia com kernels retangular, eliptico e cruz, salvando o painel. 6. Use connectedComponentsWithStats para filtrar e medir as marcacoes. 7. Apresente as saídas e discuta os limites de calibracao e do modelo geometrico.

COMO EXECUTAR: python exemplos/40_pipeline_completo.py
DEPENDENCIAS: opencv-python, numpy; nenhuma dependencia de outro .py.
"""

from pathlib import Path
import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
SAIDA = RAIZ / "saidas" / Path(__file__).stem
SAIDA.mkdir(parents=True, exist_ok=True)

def ler(nome):
    caminho = DADOS / nome
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        raise FileNotFoundError(f"Imagem ausente: {caminho}")
    return imagem

def painel(pares, destino="resultado.png", largura=560, altura=310):
    """Cada tupla contem (legenda, imagem); salva um grid com ate 3 colunas."""
    quadros = []
    for legenda, imagem in pares:
        if len(imagem.shape) == 2:
            imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (largura, 43), (22, 31, 46), -1)
        cv2.putText(img, legenda[:43], (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.66, (255, 255, 255), 2, cv2.LINE_AA)
        quadros.append(img)
    ncol = min(3, len(quadros))
    while len(quadros) % ncol:
        quadros.append(np.full_like(quadros[0], (235, 235, 235)))
    linhas = [np.hstack(quadros[i:i+ncol]) for i in range(0, len(quadros), ncol)]
    resultado = np.vstack(linhas)
    cv2.imwrite(str(SAIDA / destino), resultado)
    print("Imagem salva:", SAIDA / destino)
    return resultado


def orb_pares(nfeatures=2500):
    a, b = ler("visao_01.png"), ler("visao_02.png")
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=12)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        raise RuntimeError("Nao foram detectados descritores ORB. Use cenas com textura.")
    return a, b, k1, d1, k2, d2

def matches_lowe(d1, d2, ratio=0.80):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pares = bf.knnMatch(d2, d1, k=2)  # referencia: imagem 02 --> 01
    bons = [m for par in pares if len(par) == 2
            for m, n in [par] if m.distance < ratio * n.distance]
    return bons

def estimar_homografia(ratio=0.80, limiar=3.0):
    a, b, k1, d1, k2, d2 = orb_pares()
    bons = matches_lowe(d1, d2, ratio)
    if len(bons) < 4:
        raise RuntimeError(f"Somente {len(bons)} matches. Precisa de pelo menos 4.")
    src = np.float32([k2[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in bons]).reshape(-1, 1, 2)
    H, mascara = cv2.findHomography(src, dst, cv2.RANSAC, limiar)
    if H is None or mascara is None:
        raise RuntimeError("Homografia nao encontrada; verifique sobreposicao e matches.")
    inliers = mascara.ravel().astype(bool)
    projetados = cv2.perspectiveTransform(src, H)
    erros = np.linalg.norm((projetados - dst).reshape(-1, 2), axis=1)
    erro_medio = float(np.mean(erros[inliers])) if inliers.any() else float("nan")
    return a, b, H, inliers, erro_medio, bons, k1, k2, erros, src, dst

def construir_mosaico(blending=True):
    a, b, H, inliers, erro, bons, k1, k2, erros, src, dst = estimar_homografia()
    h1,w1=a.shape[:2];h2,w2=b.shape[:2]
    c1=np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    c2=np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    p2=cv2.perspectiveTransform(c2,H).reshape(-1,2)
    todos=np.vstack([c1.reshape(-1,2),p2])
    xmin,ymin=np.floor(todos.min(axis=0)).astype(int)
    xmax,ymax=np.ceil(todos.max(axis=0)).astype(int)
    T=np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]],np.float64)
    tam=(int(xmax-xmin),int(ymax-ymin))
    im1=cv2.warpPerspective(a,T,tam)
    im2=cv2.warpPerspective(b,T@H,tam)
    m1=cv2.warpPerspective(np.full((h1,w1),255,np.uint8),T,tam)>0
    m2=cv2.warpPerspective(np.full((h2,w2),255,np.uint8),T@H,tam)>0
    mosaico=im1.copy(); apenas2=m2&~m1; mosaico[apenas2]=im2[apenas2]
    if blending:
        comum=m1&m2
        mosaico[comum]=np.clip(0.5*im1[comum].astype(float)
                               +0.5*im2[comum].astype(float),0,255).astype(np.uint8)
    return a,b,H,inliers,erro,mosaico,im1,im2,m1,m2,T

def binarizar(img):
    cinza=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    return cv2.adaptiveThreshold(cinza,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,31,-12)

def morfologia(mask, forma=cv2.MORPH_RECT, tamanho=(5,7)):
    kernel=cv2.getStructuringElement(forma,tamanho)
    abertura=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    fechamento=cv2.morphologyEx(abertura,cv2.MORPH_CLOSE,kernel)
    return abertura, fechamento, kernel

def selecionar_componentes(img, mascara, area_min=100, razao_min=1.5):
    n,rotulos,stats,centros=cv2.connectedComponentsWithStats(mascara,8)
    saida=img.copy(); aceitos=[]
    for i in range(1,n):  # indice 0 corresponde ao fundo
        x,y,w,h,area=map(int,stats[i])
        razao=h/max(w,1)
        if area>=area_min and razao>=razao_min:
            aceitos.append((x,y,w,h,area,razao))
            cv2.rectangle(saida,(x,y),(x+w,y+h),(0,255,0),2)
    if aceitos:
        print('Marcacoes aprovadas:',len(aceitos),
              'largura media:',round(float(np.mean([v[2] for v in aceitos])),1),
              'altura media:',round(float(np.mean([v[3] for v in aceitos])),1),
              'area media:',round(float(np.mean([v[4] for v in aceitos])),1))
    else:
        print('Nenhuma marcacao atende aos filtros de area e aspect ratio.')
    return saida, aceitos, stats

# Integracao em um unico programa autonomo. Nenhum outro exemplo precisa ter sido executado.
a,b,H,inliers,erro,mos,im1,im2,m1,m2,T=construir_mosaico(blending=True)
cv2.imwrite(str(SAIDA/"mosaico.png"),mos)
print("MOSAICO: matches inliers:",int(inliers.sum()),"de",len(inliers),
      "erro medio dos inliers (px):",round(erro,3))
print("Matriz de homografia H (vista 02 para 01):\n",H)
painel([("Primeira vista",a),("Segunda vista",b),("Mosaico panoramico",mos)],"painel_mosaico.png")
bev=ler("bev_marcacoes.png");mask=binarizar(bev)
pares=[("BEV",bev),("Threshold adaptativo",mask)]
for nome,forma in [("Retangular",cv2.MORPH_RECT),("Eliptico",cv2.MORPH_ELLIPSE),("Cruz",cv2.MORPH_CROSS)]:
 abertura,fechado,k=morfologia(mask,forma,(5,7))
 print("Kernel",nome,"(5x7):\n",k)
 pares.append(("Closing - "+nome,fechado))
 if nome=="Retangular":
  cv2.imwrite(str(SAIDA/"opening.png"),abertura)
  cv2.imwrite(str(SAIDA/"closing.png"),fechado)
  marcadas,selecionadas,stats=selecionar_componentes(bev,fechado,100,1.5)
  print("DIMENSOES MEDIAS:",
       "largura",round(float(np.mean([z[2] for z in selecionadas])),2) if selecionadas else 0,
       "altura",round(float(np.mean([z[3] for z in selecionadas])),2) if selecionadas else 0)
  cv2.imwrite(str(SAIDA/"componentes_filtrados.png"),marcadas)
  pares.append(("Componentes filtrados",marcadas))
painel(pares,"painel_morfologia.png")
# Dados de TP1 incluidos: para seu BEV proprio troque ler("bev_marcacoes.png") por ler("bev_tp1.png")
print("Saidas completas em:",SAIDA)
