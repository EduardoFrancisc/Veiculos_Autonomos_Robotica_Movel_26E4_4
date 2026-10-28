"""EXEMPLO 30/32 - Estimar raio de curvatura em metros

OBJETIVO: Estimar raio de curvatura em metros.
COMO FUNCIONA: Converte A e B do polyfit para escalas metricas e aplica formula da curvatura.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/30_raio_curvatura.png; raio esquerdo, direito e medio no terminal.
COMO EXECUTAR: python exemplos/30_raio_curvatura.py
EXPERIMENTE: Compare: raio muito grande significa trecho praticamente reto.
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

# Hough indica segmentos. Para ajustar curvas, usamos pixels de suporte
# da mascara nas laterais do BEV, evitando o centro da imagem.
suporte = np.zeros_like(mascara)
if segmentos is not None:
    for x1, y1, x2, y2 in segmentos.reshape(-1, 4):
        x_medio = (x1 + x2) / 2
        if abs(x2 - x1) <= 120 and (x_medio < 510 or x_medio > 770):
            cv2.line(suporte, (x1, y1), (x2, y2), 255, 13)
suporte = cv2.bitwise_and(suporte, mascara)
# Alternativa didatica se poucos segmentos sustentam as faixas.
if cv2.countNonZero(suporte) < 250:
    suporte = mascara.copy()
ys, xs = np.nonzero(suporte)
regioes = (ys > 80) & (((xs > 195) & (xs < 515)) |
                         ((xs > 765) & (xs < 1085)))
ys, xs = ys[regioes], xs[regioes]
selecao_esquerda = xs < largura // 2
selecao_direita = xs > largura // 2
if np.count_nonzero(selecao_esquerda) < 50 or np.count_nonzero(selecao_direita) < 50:
    raise RuntimeError("Faixas insuficientes para ajustar os polinomios.")
# Relacao x = A*y^2 + B*y + C, segundo grau em cada lateral.
ajuste_esquerda = np.polyfit(ys[selecao_esquerda], xs[selecao_esquerda], 2)
ajuste_direita = np.polyfit(ys[selecao_direita], xs[selecao_direita], 2)

# Amostragem das curvas para visualizacao.
linhas_y = np.arange(altura, dtype=np.float64)
x_esquerda = np.clip(np.polyval(ajuste_esquerda, linhas_y), 0, largura - 1)
x_direita = np.clip(np.polyval(ajuste_direita, linhas_y), 0, largura - 1)
pts_esquerda = np.column_stack((x_esquerda.astype(int), linhas_y.astype(int))).astype(np.int32)
pts_direita = np.column_stack((x_direita.astype(int), linhas_y.astype(int))).astype(np.int32)

# Conversoes fixadas no enunciado (validas apenas com geometria calibrada).
metros_por_pixel_x = 3.7 / 700
metros_por_pixel_y = 30 / 720
y_rodape = altura - 1
centro_veiculo_px = largura / 2
pos_esquerda = np.polyval(ajuste_esquerda, y_rodape)
pos_direita = np.polyval(ajuste_direita, y_rodape)
centro_faixa_px = (pos_esquerda + pos_direita) / 2
desvio_px = centro_veiculo_px - centro_faixa_px
desvio_m = desvio_px * metros_por_pixel_x  # Positivo: veiculo a direita.

def raio_curvatura_m(coeficientes):
    """Raio R = (1 + (dx/dy)^2)^(3/2) / abs(d2x/dy2), em metros."""
    a = coeficientes[0] * metros_por_pixel_x / (metros_por_pixel_y ** 2)
    b = coeficientes[1] * metros_por_pixel_x / metros_por_pixel_y
    y_m = y_rodape * metros_por_pixel_y
    return ((1 + (2 * a * y_m + b) ** 2) ** 1.5) / (abs(2 * a) + 1e-12)

raio_esquerda = raio_curvatura_m(ajuste_esquerda)
raio_direita = raio_curvatura_m(ajuste_direita)
raio_medio = (raio_esquerda + raio_direita) / 2

marcada = bev.copy()
cv2.polylines(marcada, [pts_esquerda], False, (255, 255, 255), 5)
cv2.polylines(marcada, [pts_direita], False, (0, 230, 255), 5)
cv2.putText(marcada, f"Raio medio: {raio_medio:.1f} m", (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
print(f"Raio esquerda = {raio_esquerda:.2f} m")
print(f"Raio direita  = {raio_direita:.2f} m")
print(f"Raio medio    = {raio_medio:.2f} m")
cv2.imwrite(str(PASTA_SAIDA / "30_raio_curvatura.png"),
            montar_painel(("BEV", bev), ("Curvatura em metros", marcada)))
