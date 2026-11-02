"""EXEMPLO 13/32 - Limitar analise a regiao de interesse (ROI)

OBJETIVO: Limitar analise a regiao de interesse (ROI).
COMO FUNCIONA: Cria mascara de poligono trapezoidal e elimina pixels fora da via.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/13_regiao_interesse.png com ROI e imagem recortada.
COMO EXECUTAR: python exemplos/13_regiao_interesse.py
EXPERIMENTE: Altere uma das coordenadas do trapezio.
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

vertices = np.array([[130, 710], [570, 430], [710, 430], [1150, 710]], dtype=np.int32)
mascara_roi = np.zeros((altura, largura), dtype=np.uint8)
cv2.fillPoly(mascara_roi, [vertices], 255)
recorte = cv2.bitwise_and(imagem, imagem, mask=mascara_roi)
desenho = imagem.copy()
cv2.polylines(desenho, [vertices], True, (30, 240, 255), 5)
cv2.imwrite(str(PASTA_SAIDA / "13_regiao_interesse.png"),
            montar_painel(("Trapezio da ROI", desenho), ("Somente ROI", recorte)))
