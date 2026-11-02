"""
EXEMPLO 04/40 — DETECTAR KEYPOINTS COM ORB

O QUE ESTE CODIGO FAZ: Este programa foi elaborado para encontrar pontos de interesse na primeira vista. Cada exercicio
e independente e pode ser executado diretamente, sem apoio.py ou importacao
de outros exemplos. As imagens de entrada ficam na pasta dados.

COMO FUNCIONA: A demonstracao usa cv2.ORB_create, detectAndCompute e drawKeypoints. O programa apresenta os parametros,
imprime medidas quantitativas quando sao relevantes e salva resultados na
pasta saidas/04_detectar_orb/, criada automaticamente.

POR QUE E IMPORTANTE: pontos estáveis são candidatos ao alinhamento entre vistas. Na visao veicular, a correspondencia de pontos
auxilia o alinhamento de imagens, enquanto o BEV e a morfologia fornecem
representacoes uteis para encontrar marcacoes de pista.

APLICACAO E CUIDADOS: Este material utiliza duas janelas sobrepostas de uma
mesma fotografia, simulando vistas de cameras. Nao e uma calibracao real
de surround view 360 graus. Para multiplas cameras fisicas sao necessarias
calibracao intrinseca/extrinseca, compensacao de distorcao e tratamento da
paralaxe. A segmentacao BEV trabalha inicialmente com cena controlada.

EXPERIMENTE NA AULA: Mude nfeatures=500 para nfeatures=2500.

PASSO A PASSO DA EXECUCAO: 1. Leia a imagem e transforme-a em cinza. 2. Crie o detector ORB, variando o numero de features de acordo com a cena. 3. Use detectAndCompute para obter pontos-chave e descritores binarios. 4. Imprima quantidade e formato dos dados e desenhe as features na fotografia. 5. Discuta por que uma imagem com pouca textura pode nao oferecer correspondencias suficientes.

COMO EXECUTAR: python exemplos/04_detectar_orb.py
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


a=ler("visao_01.png");g=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
orb=cv2.ORB_create(nfeatures=1600,fastThreshold=12)
k,d=orb.detectAndCompute(g,None)
print("Keypoints ORB:",len(k),"Descritores:",None if d is None else d.shape)
vis=cv2.drawKeypoints(a,k,None,color=(0,255,0),flags=0)
painel([("Imagem",a),("Pontos ORB",vis)])
