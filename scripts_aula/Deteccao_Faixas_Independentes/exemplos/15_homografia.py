"""EXEMPLO 15/32 - Calcular a matriz de transformacao projetiva

OBJETIVO: Calcular a matriz de transformacao projetiva.
COMO FUNCIONA: cv2.getPerspectiveTransform calcula matriz 3x3 entre dois conjuntos de quatro pontos.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/15_homografia.png e matriz 3x3 impressa.
COMO EXECUTAR: python exemplos/15_homografia.py
EXPERIMENTE: Observe o efeito de alterar os pontos de destino.
DEPENDENCIAS: cv2, numpy; nao importa nenhum outro arquivo .py.
"""

from pathlib import Path
import cv2
import numpy as np

# Os caminhos sao relativos a este arquivo, nao ao diretorio do Colab.
RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_IMAGEM = RAIZ / "dados" / "estrada_sintetica.png"
PASTA_SAIDA = RAIZ / "saida"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# Leitura real de um arquivo PNG com OpenCV.
imagem = cv2.imread(str(CAMINHO_IMAGEM))
if imagem is None:
    raise FileNotFoundError(f"Nao foi possivel ler: {CAMINHO_IMAGEM}")
altura, largura = imagem.shape[:2]

def montar_painel(*pares):
    """Monta comparacoes lado a lado; cada par e (legenda, imagem)."""
    quadrinhos = []
    for legenda, img in pares:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vista = cv2.resize(img, (640, 360))
        cv2.rectangle(vista, (0, 0), (640, 40), (22, 31, 45), -1)
        cv2.putText(vista, legenda, (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        quadrinhos.append(vista)
    if len(quadrinhos) == 2:
        return np.hstack(quadrinhos)
    if len(quadrinhos) == 4:
        return np.vstack((np.hstack(quadrinhos[:2]),
                          np.hstack(quadrinhos[2:])))
    raise ValueError("Use exatamente dois ou quatro quadros.")

# Quatro pontos selecionados manualmente para ESTE video sintetico.
# Ordem: superior esquerdo, superior direito, inferior direito, inferior esquerdo.
pontos_origem = np.float32([
    [570, 430], [710, 430], [1150, 710], [130, 710]
])
pontos_destino = np.float32([
    [290, 0], [990, 0], [990, 719], [290, 719]
])
matriz = cv2.getPerspectiveTransform(pontos_origem, pontos_destino)
matriz_inversa = cv2.getPerspectiveTransform(pontos_destino, pontos_origem)
bev = cv2.warpPerspective(imagem, matriz, (largura, altura))

print("Matriz da homografia (3x3):", matriz)
print("A matriz depende das coordenadas em pixels da camera.")
cv2.imwrite(str(PASTA_SAIDA / "15_homografia.png"),
            montar_painel(("Perspectiva camera", imagem), ("Perspectiva BEV", bev)))
