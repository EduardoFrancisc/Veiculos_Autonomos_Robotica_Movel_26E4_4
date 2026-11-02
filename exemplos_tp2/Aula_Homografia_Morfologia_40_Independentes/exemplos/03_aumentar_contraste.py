"""
EXEMPLO 03/40 — PREPARAR CONTRASTE PARA ENCONTRAR FEATURES

O QUE ESTE CODIGO FAZ: Este programa foi elaborado para demonstrar o impacto da equalização local na visibilidade de textura. Cada exercicio
e independente e pode ser executado diretamente, sem apoio.py ou importacao
de outros exemplos. As imagens de entrada ficam na pasta dados.

COMO FUNCIONA: A demonstracao usa CLAHE em tons de cinza e comparação visual. O programa apresenta os parametros,
imprime medidas quantitativas quando sao relevantes e salva resultados na
pasta saidas/03_aumentar_contraste/, criada automaticamente.

POR QUE E IMPORTANTE: as condições de iluminação influenciam detectores de pontos. Na visao veicular, a correspondencia de pontos
auxilia o alinhamento de imagens, enquanto o BEV e a morfologia fornecem
representacoes uteis para encontrar marcacoes de pista.

APLICACAO E CUIDADOS: Este material utiliza duas janelas sobrepostas de uma
mesma fotografia, simulando vistas de cameras. Nao e uma calibracao real
de surround view 360 graus. Para multiplas cameras fisicas sao necessarias
calibracao intrinseca/extrinseca, compensacao de distorcao e tratamento da
paralaxe. A segmentacao BEV trabalha inicialmente com cena controlada.

EXPERIMENTE NA AULA: Teste clipLimit=1.0 e 4.0.

PASSO A PASSO DA EXECUCAO: 1. Localize os arquivos PNG dentro de dados e leia-os pelo OpenCV. 2. Confira altura, largura e canais para evitar processamento de entrada invalida. 3. Execute a operacao introdutoria indicada no titulo; compare entrada e saida. 4. Inspecione a figura salva e relacione o resultado a qualidade das futuras correspondencias.

COMO EXECUTAR: python exemplos/03_aumentar_contraste.py
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
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
melhor=clahe.apply(g)
print("Media de intensidade antes/depois:",round(float(g.mean()),1),round(float(melhor.mean()),1))
painel([("Cinza",g),("CLAHE",melhor)])
