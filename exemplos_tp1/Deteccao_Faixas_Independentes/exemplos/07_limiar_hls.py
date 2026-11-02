"""EXEMPLO 07/32 - Limiarizar cores branca e amarela no HLS

OBJETIVO: Limiarizar cores branca e amarela no HLS.
COMO FUNCIONA: Aplica limiares distintos para amarelo saturado e branco luminoso.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/07_limiar_hls.png mostrando duas mascaras e combinacao.
COMO EXECUTAR: python exemplos/07_limiar_hls.py
EXPERIMENTE: Mude o limiar do branco de L>=165 para L>=220.
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
mascara_amarela = np.uint8(((s >= 90) & (l >= 75)) * 255)
mascara_branca = np.uint8(((l >= 165) & (s < 95)) * 255)
combinada = cv2.bitwise_or(mascara_amarela, mascara_branca)
cv2.imwrite(str(PASTA_SAIDA / "07_limiar_hls.png"),
            montar_painel(("Original", imagem), ("Amarelo", mascara_amarela),
                          ("Branco", mascara_branca), ("Combinacao", combinada)))
