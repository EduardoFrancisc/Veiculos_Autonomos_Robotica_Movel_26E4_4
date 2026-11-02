"""
EXEMPLO 08/40 — OBTER DOIS VIZINHOS PARA CADA DESCRIPTOR

O QUE ESTE CODIGO FAZ: Este programa foi elaborado para introduzir o teste de razão de Lowe. Cada exercicio
e independente e pode ser executado diretamente, sem apoio.py ou importacao
de outros exemplos. As imagens de entrada ficam na pasta dados.

COMO FUNCIONA: A demonstracao usa BFMatcher(...).knnMatch(...,k=2). O programa apresenta os parametros,
imprime medidas quantitativas quando sao relevantes e salva resultados na
pasta saidas/08_matcher_knn/, criada automaticamente.

POR QUE E IMPORTANTE: comparar os dois melhores candidatos permite rejeitar correspondências ambíguas. Na visao veicular, a correspondencia de pontos
auxilia o alinhamento de imagens, enquanto o BEV e a morfologia fornecem
representacoes uteis para encontrar marcacoes de pista.

APLICACAO E CUIDADOS: Este material utiliza duas janelas sobrepostas de uma
mesma fotografia, simulando vistas de cameras. Nao e uma calibracao real
de surround view 360 graus. Para multiplas cameras fisicas sao necessarias
calibracao intrinseca/extrinseca, compensacao de distorcao e tratamento da
paralaxe. A segmentacao BEV trabalha inicialmente com cena controlada.

EXPERIMENTE NA AULA: Imprima as distâncias dos dois vizinhos.

PASSO A PASSO DA EXECUCAO: 1. Extraia descritores ORB separadamente de duas vistas sobrepostas. 2. Empregue BFMatcher e distancia de Hamming para consultar pontos correspondentes. 3. Quando usar knnMatch, compare dois candidatos e aplique a razao de Lowe. 4. Desenhe os pares resultantes e avalie sua quantidade e distribuicao espacial. 5. Lembre-se de que match visual ainda nao e prova de coerencia geometrica.

COMO EXECUTAR: python exemplos/08_matcher_knn.py
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
pares=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2,d1,k=2)
validos=[p for p in pares if len(p)==2]
print("Descritores analisados:",len(pares),"Duplas validas:",len(validos))
print("Primeiras distancias (melhor,segundo):",[(p[0].distance,p[1].distance) for p in validos[:5]])
vis=cv2.drawMatchesKnn(b,k2,a,k1,validos[:25],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
painel([("Dois vizinhos por feature",vis)])
