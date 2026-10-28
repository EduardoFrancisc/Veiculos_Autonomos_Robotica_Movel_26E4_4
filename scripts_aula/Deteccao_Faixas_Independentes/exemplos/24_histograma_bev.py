"""EXEMPLO 24/32 - Localizar faixas por histograma de pixels

OBJETIVO: Localizar faixas por histograma de pixels.
COMO FUNCIONA: Soma pixels da mascara na metade inferior do BEV para encontrar bases das faixas.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/24_histograma_bev.png com mascara e grafico de barras desenhado.
COMO EXECUTAR: python exemplos/24_histograma_bev.py
EXPERIMENTE: Observe em que colunas se concentram os pixels das faixas.
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

# O histograma acumula pixels brancos na metade inferior da mascara.
histograma = np.sum(mascara[altura // 2:, :] > 0, axis=0)
meio = largura // 2
pico_esquerdo = int(np.argmax(histograma[:meio]))
pico_direito = int(np.argmax(histograma[meio:]) + meio)
grafico = np.zeros((altura, largura, 3), dtype=np.uint8)
escala = 580 / max(float(histograma.max()), 1.0)
for x, valor in enumerate(histograma):
    y_topo = altura - 1 - int(valor * escala)
    cv2.line(grafico, (x, altura - 1), (x, y_topo), (0, 200, 255), 1)
for px in (pico_esquerdo, pico_direito):
    cv2.line(grafico, (px, 0), (px, altura - 1), (255, 100, 60), 3)
print("Picos estimados: esquerda=", pico_esquerdo, ", direita=", pico_direito)
cv2.imwrite(str(PASTA_SAIDA / "24_histograma_bev.png"),
            montar_painel(("Mascara no BEV", mascara), ("Histograma inferior", grafico)))
