"""
EXEMPLO 25/40 — INSPECIONAR IMAGEM BEV

O QUE ESTE CODIGO FAZ: Este programa foi elaborado para introduzir a vista superior com faixas e ruído. Cada exercicio
e independente e pode ser executado diretamente, sem apoio.py ou importacao
de outros exemplos. As imagens de entrada ficam na pasta dados.

COMO FUNCIONA: A demonstracao usa abrir BEV didática e imagem BEV da aula anterior. O programa apresenta os parametros,
imprime medidas quantitativas quando sao relevantes e salva resultados na
pasta saidas/25_inspecionar_bev/, criada automaticamente.

POR QUE E IMPORTANTE: o BEV facilita observar marcações aproximadamente verticais. Na visao veicular, a correspondencia de pontos
auxilia o alinhamento de imagens, enquanto o BEV e a morfologia fornecem
representacoes uteis para encontrar marcacoes de pista.

APLICACAO E CUIDADOS: Este material utiliza duas janelas sobrepostas de uma
mesma fotografia, simulando vistas de cameras. Nao e uma calibracao real
de surround view 360 graus. Para multiplas cameras fisicas sao necessarias
calibracao intrinseca/extrinseca, compensacao de distorcao e tratamento da
paralaxe. A segmentacao BEV trabalha inicialmente com cena controlada.

EXPERIMENTE NA AULA: Troque o nome por dados/bev_tp1.png.

PASSO A PASSO DA EXECUCAO: 1. Abra a visao superior BEV em um arquivo de entrada real ou controlado. 2. Converta em tons de cinza e calcule limiar local adaptativo gaussiano. 3. Compare regioes selecionadas com as marcas visiveis e outras manchas da pista. 4. Varie blockSize e C, explicando por que a escala espacial do limiar importa. 5. Salve uma mascara binaria que sera analisada nas etapas seguintes.

COMO EXECUTAR: python exemplos/25_inspecionar_bev.py
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


a=ler("bev_marcacoes.png");b=ler("bev_tp1.png")
print("BEV simplificada:",a.shape,"BEV do exercicio anterior:",b.shape)
painel([("BEV com marcacoes",a),("BEV extraida do TP1",b)])
