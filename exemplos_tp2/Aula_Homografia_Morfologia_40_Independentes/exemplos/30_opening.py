"""
EXEMPLO 30/40 — ABERTURA MORFOLÓGICA

O QUE ESTE CODIGO FAZ: Este programa foi elaborado para remover ruído isolado. Cada exercicio
e independente e pode ser executado diretamente, sem apoio.py ou importacao
de outros exemplos. As imagens de entrada ficam na pasta dados.

COMO FUNCIONA: A demonstracao usa cv2.morphologyEx(...,MORPH_OPEN,kernel). O programa apresenta os parametros,
imprime medidas quantitativas quando sao relevantes e salva resultados na
pasta saidas/30_opening/, criada automaticamente.

POR QUE E IMPORTANTE: opening aplica erosão seguida por dilatação e remove pequenos artefatos. Na visao veicular, a correspondencia de pontos
auxilia o alinhamento de imagens, enquanto o BEV e a morfologia fornecem
representacoes uteis para encontrar marcacoes de pista.

APLICACAO E CUIDADOS: Este material utiliza duas janelas sobrepostas de uma
mesma fotografia, simulando vistas de cameras. Nao e uma calibracao real
de surround view 360 graus. Para multiplas cameras fisicas sao necessarias
calibracao intrinseca/extrinseca, compensacao de distorcao e tratamento da
paralaxe. A segmentacao BEV trabalha inicialmente com cena controlada.

EXPERIMENTE NA AULA: Compare a quantidade de componentes antes e depois.

PASSO A PASSO DA EXECUCAO: 1. Transforme a BEV em mascara binaria por threshold adaptativo. 2. Crie um elemento estruturante com forma e dimensoes declaradas no codigo. 3. Execute erosao, dilatacao, opening e/ou closing segundo o objetivo deste exemplo. 4. Compare o resultado ao original e verifique se lacunas ou ruídos foram reduzidos. 5. Observe que o kernel pode remover detalhes validos ou unir marcas proximas.

COMO EXECUTAR: python exemplos/30_opening.py
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

img=ler("bev_marcacoes.png");mask=binarizar(img)
k=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
open_=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
print("Pixels antes/depois do opening:",int(np.count_nonzero(mask)),int(np.count_nonzero(open_)))
painel([("Mascara inicial",mask),("Opening",open_)])
