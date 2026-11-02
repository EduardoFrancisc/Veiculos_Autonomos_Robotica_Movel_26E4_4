"""EXEMPLO 09/32 - Comparar gradientes Sobel X e Sobel Y

OBJETIVO: Comparar gradientes Sobel X e Sobel Y.
COMO FUNCIONA: Calcula derivadas X e Y e compara as orientacoes das bordas.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/09_sobel_x_y.png com original, X, Y e combinacao.
COMO EXECUTAR: python exemplos/09_sobel_x_y.py
EXPERIMENTE: Identifique qual componente detecta melhor as linhas longitudinais.
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

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
grad_x = cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(cinza, cv2.CV_64F, 0, 1, ksize=3)
x = cv2.convertScaleAbs(grad_x)
y = cv2.convertScaleAbs(grad_y)
combinada = cv2.addWeighted(x, 0.5, y, 0.5, 0)
cv2.imwrite(str(PASTA_SAIDA / "09_sobel_x_y.png"),
            montar_painel(("Original", imagem), ("Sobel X", x),
                          ("Sobel Y", y), ("X + Y", combinada)))
