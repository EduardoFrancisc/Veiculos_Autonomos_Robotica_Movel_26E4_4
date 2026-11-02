"""EXEMPLO 20/32 - Filtrar segmentos geometricamente plausiveis

OBJETIVO: Filtrar segmentos geometricamente plausiveis.
COMO FUNCIONA: Descarta linhas muito horizontais ou distantes das duas laterais previstas.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/20_filtrar_segmentos.png com segmentos todos e filtrados.
COMO EXECUTAR: python exemplos/20_filtrar_segmentos.py
EXPERIMENTE: Altere o limite abs(x2-x1)<=120 e verifique as exclusoes.
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

filtrados = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
aceitos = 0
if segmentos is not None:
    for x1, y1, x2, y2 in segmentos.reshape(-1, 4):
        meio_x = (x1 + x2) / 2
        if abs(x2 - x1) <= 120 and (meio_x < 510 or meio_x > 770):
            cv2.line(filtrados, (x1, y1), (x2, y2), (50, 230, 50), 3)
            aceitos += 1
print("Segmentos totais:", 0 if segmentos is None else len(segmentos))
print("Segmentos geometricamente aceitos:", aceitos)
cv2.imwrite(str(PASTA_SAIDA / "20_filtrar_segmentos.png"),
            montar_painel(("Todos: Hough", visao_hough), ("Filtrados", filtrados)))
