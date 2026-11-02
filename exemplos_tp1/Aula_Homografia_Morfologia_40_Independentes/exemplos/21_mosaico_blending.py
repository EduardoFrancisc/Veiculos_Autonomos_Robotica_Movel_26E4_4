"""
EXEMPLO 21/40 — FUNDIR VISTAS POR MÉDIA PONDERADA

O QUE ESTE CODIGO FAZ: Este programa foi elaborado para reduzir a descontinuidade na sobreposição. Cada exercicio
e independente e pode ser executado diretamente, sem apoio.py ou importacao
de outros exemplos. As imagens de entrada ficam na pasta dados.

COMO FUNCIONA: A demonstracao usa máscaras de validade e média dos pixels compartilhados. O programa apresenta os parametros,
imprime medidas quantitativas quando sao relevantes e salva resultados na
pasta saidas/21_mosaico_blending/, criada automaticamente.

POR QUE E IMPORTANTE: o blending melhora a apresentação mas não corrige um alinhamento geométrico incorreto. Na visao veicular, a correspondencia de pontos
auxilia o alinhamento de imagens, enquanto o BEV e a morfologia fornecem
representacoes uteis para encontrar marcacoes de pista.

APLICACAO E CUIDADOS: Este material utiliza duas janelas sobrepostas de uma
mesma fotografia, simulando vistas de cameras. Nao e uma calibracao real
de surround view 360 graus. Para multiplas cameras fisicas sao necessarias
calibracao intrinseca/extrinseca, compensacao de distorcao e tratamento da
paralaxe. A segmentacao BEV trabalha inicialmente com cena controlada.

EXPERIMENTE NA AULA: Compare com o mosaico sem blending.

PASSO A PASSO DA EXECUCAO: 1. Detecte e corresponda pontos ORB nas duas vistas. 2. Estime homografia com RANSAC, relatando numero de inliers e erro de reprojecao. 3. Projete os cantos e ajuste a origem do canvas para evitar cortar a imagem. 4. Remapeie imagens e mascaras de validade pelo warpPerspective. 5. Monte e visualize o mosaico; na regiao compartilhada, discuta a costura e blending. 6. Nao generalize uma homografia unica para objetos em diferentes profundidades.

COMO EXECUTAR: python exemplos/21_mosaico_blending.py
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

a,b,H,inliers,erro,mos,im1,im2,m1,m2,T=construir_mosaico(blending=True)
print("Mosaico com blending:",mos.shape,"Inliers:",int(inliers.sum()),"Erro:",round(erro,3))
cv2.imwrite(str(SAIDA/"mosaico.png"),mos)
painel([("Camera 01",a),("Camera 02",b),("Mosaico suavizado",mos)])
