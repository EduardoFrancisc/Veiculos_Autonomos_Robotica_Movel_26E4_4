"""EXEMPLO 22/32 - Montar o painel completo do Item A em um frame

OBJETIVO: Montar o painel completo do Item A em um frame.
COMO FUNCIONA: Reune imagem original, BEV, mascara HLS+Sobel e Canny+Hough.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/22_painel_item_a.png com quatro etapas e PNG com linhas na camera.
COMO EXECUTAR: python exemplos/22_painel_item_a.py
EXPERIMENTE: Use o painel para explicar passo a passo o processamento.
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

# Segmentacao no espaco BEV: saturacao HLS + bordas Sobel X.
hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
canal_l = hls[:, :, 1]
canal_s = hls[:, :, 2]
cinza_bev = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(cinza_bev, cv2.CV_64F, 1, 0, ksize=3)
magnitude = np.abs(sobel_x)
sobel_8bits = np.uint8(255 * magnitude / (magnitude.max() + 1e-9))
# O amarelo tem saturacao alta, enquanto o branco exige luminosidade alta.
amarelo = (canal_s >= 90) & (canal_l >= 75)
branco = (canal_l >= 165) & (canal_s < 95)
bordas_sobel = sobel_8bits >= 23
mascara = np.uint8((amarelo | branco | bordas_sobel) * 255)
mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE,
                           np.ones((5, 5), dtype=np.uint8))

arestas = cv2.Canny(mascara, 60, 150)
segmentos = cv2.HoughLinesP(arestas, rho=1, theta=np.pi / 180,
                           threshold=35, minLineLength=35, maxLineGap=70)
visao_hough = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
if segmentos is not None:
    for x1, y1, x2, y2 in segmentos.reshape(-1, 4):
        cv2.line(visao_hough, (x1, y1), (x2, y2), (0, 120, 255), 3)

# Nesta etapa, o resultado e somente Hough (sem regiao polinomial preenchida).
cv2.imwrite(str(PASTA_SAIDA / "22_painel_item_a.png"),
            montar_painel(("1. Original", imagem), ("2. BEV", bev),
                          ("3. HLS + Sobel", mascara), ("4. Canny + Hough", visao_hough)))
