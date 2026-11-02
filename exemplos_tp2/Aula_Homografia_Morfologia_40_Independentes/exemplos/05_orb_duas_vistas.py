"""
EXEMPLO 05/40 — COMPARAR ORB NAS DUAS IMAGENS

O QUE ESTE CODIGO FAZ: Este programa foi elaborado para detectar pontos-chave separadamente em câmeras simuladas. Cada exercicio
e independente e pode ser executado diretamente, sem apoio.py ou importacao
de outros exemplos. As imagens de entrada ficam na pasta dados.

COMO FUNCIONA: A demonstracao usa duas execuções de ORB e desenho lado a lado. O programa apresenta os parametros,
imprime medidas quantitativas quando sao relevantes e salva resultados na
pasta saidas/05_orb_duas_vistas/, criada automaticamente.

POR QUE E IMPORTANTE: a quantidade e a distribuição das features afetam a homografia. Na visao veicular, a correspondencia de pontos
auxilia o alinhamento de imagens, enquanto o BEV e a morfologia fornecem
representacoes uteis para encontrar marcacoes de pista.

APLICACAO E CUIDADOS: Este material utiliza duas janelas sobrepostas de uma
mesma fotografia, simulando vistas de cameras. Nao e uma calibracao real
de surround view 360 graus. Para multiplas cameras fisicas sao necessarias
calibracao intrinseca/extrinseca, compensacao de distorcao e tratamento da
paralaxe. A segmentacao BEV trabalha inicialmente com cena controlada.

EXPERIMENTE NA AULA: Compare as regiões da cena onde há mais keypoints.

PASSO A PASSO DA EXECUCAO: 1. Leia a imagem e transforme-a em cinza. 2. Crie o detector ORB, variando o numero de features de acordo com a cena. 3. Use detectAndCompute para obter pontos-chave e descritores binarios. 4. Imprima quantidade e formato dos dados e desenhe as features na fotografia. 5. Discuta por que uma imagem com pouca textura pode nao oferecer correspondencias suficientes.

COMO EXECUTAR: python exemplos/05_orb_duas_vistas.py
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

a,b,k1,d1,k2,d2=orb_pares()
print("Features 01:",len(k1),"Features 02:",len(k2))
v1=cv2.drawKeypoints(a,k1,None,color=(0,255,0));v2=cv2.drawKeypoints(b,k2,None,color=(0,255,0))
painel([("ORB primeira vista",v1),("ORB segunda vista",v2)])
