"""EXEMPLO 08/32 - Aplicar gradiente Sobel na direcao X

OBJETIVO: Aplicar gradiente Sobel na direcao X.
COMO FUNCIONA: Calcula derivada horizontal usando cv2.Sobel; destaca bordas verticais.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/08_sobel_eixo_x.png com original e mapa de gradiente.
COMO EXECUTAR: python exemplos/08_sobel_eixo_x.py
EXPERIMENTE: Experimente mudar ksize=3 para ksize=5.
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
sobel_x = cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3)
modulo = np.abs(sobel_x)
sobel_normalizado = np.uint8(255 * modulo / (modulo.max() + 1e-9))
print("Sobel calcula diferencas de intensidade: valores negativos existem antes de abs().")
cv2.imwrite(str(PASTA_SAIDA / "08_sobel_eixo_x.png"),
            montar_painel(("Original", imagem), ("Sobel X", sobel_normalizado)))
