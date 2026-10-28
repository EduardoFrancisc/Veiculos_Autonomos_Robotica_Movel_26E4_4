"""EXEMPLO 14/32 - Marcar os quatro pontos manuais da perspectiva

OBJETIVO: Marcar os quatro pontos manuais da perspectiva.
COMO FUNCIONA: Desenha as coordenadas de origem sobre a imagem antes da homografia.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/14_quatro_pontos.png com trapezio e numeros.
COMO EXECUTAR: python exemplos/14_quatro_pontos.py
EXPERIMENTE: Posicione os pontos em outra fotografia e compare a area capturada.
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

pontos = np.float32([[570, 430], [710, 430], [1150, 710], [130, 710]])
marcada = imagem.copy()
for indice, ponto in enumerate(pontos, 1):
    xy = tuple(ponto.astype(int))
    cv2.circle(marcada, xy, 10, (0, 40, 255), -1)
    cv2.putText(marcada, str(indice), (xy[0] + 12, xy[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.polylines(marcada, [pontos.astype(np.int32)], True, (0, 255, 255), 4)
print("Pontos selecionados manualmente (x,y):", pontos)
cv2.imwrite(str(PASTA_SAIDA / "14_quatro_pontos.png"),
            montar_painel(("Imagem original", imagem), ("Quatro pontos", marcada)))
