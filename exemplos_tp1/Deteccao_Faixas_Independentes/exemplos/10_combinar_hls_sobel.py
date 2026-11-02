"""EXEMPLO 10/32 - Combinar saturacao HLS e Sobel X

OBJETIVO: Combinar saturacao HLS e Sobel X.
COMO FUNCIONA: Une limiares de amarelo/branco e gradiente de luminosidade.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/10_combinar_hls_sobel.png com fontes e mascara final.
COMO EXECUTAR: python exemplos/10_combinar_hls_sobel.py
EXPERIMENTE: Experimente remover uma das condicoes e comparar o resultado.
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

hls = cv2.cvtColor(imagem, cv2.COLOR_BGR2HLS)
l = hls[:, :, 1]
s = hls[:, :, 2]
cor = (s >= 90) & (l >= 75)
branco = (l >= 165) & (s < 95)
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
grad = cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3)
abs_grad = np.abs(grad)
sobel_norm = np.uint8(255 * abs_grad / (abs_grad.max() + 1e-9))
mascara_cor = np.uint8((cor | branco) * 255)
mascara_sobel = np.uint8((sobel_norm >= 23) * 255)
combinacao = cv2.bitwise_or(mascara_cor, mascara_sobel)
cv2.imwrite(str(PASTA_SAIDA / "10_combinar_hls_sobel.png"),
            montar_painel(("Original", imagem), ("HLS", mascara_cor),
                          ("Sobel X", mascara_sobel), ("HLS + Sobel", combinacao)))
