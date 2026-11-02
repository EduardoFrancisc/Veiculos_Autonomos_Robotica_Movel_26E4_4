"""EXEMPLO 12/32 - Comparar limiares inferior e superior do Canny

OBJETIVO: Comparar limiares inferior e superior do Canny.
COMO FUNCIONA: Executa Canny duas vezes: limiar permissivo versus conservador.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/12_comparar_canny.png em quatro quadros.
COMO EXECUTAR: python exemplos/12_comparar_canny.py
EXPERIMENTE: Conte quantos pixels de borda aparecem em cada configuracao.
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

cinza = cv2.GaussianBlur(cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY), (5, 5), 0)
sensivel = cv2.Canny(cinza, 25, 75)
conservador = cv2.Canny(cinza, 120, 240)
medio = cv2.Canny(cinza, 60, 150)
print("Arestas sensiveis:", cv2.countNonZero(sensivel))
print("Arestas conservadoras:", cv2.countNonZero(conservador))
cv2.imwrite(str(PASTA_SAIDA / "12_comparar_canny.png"),
            montar_painel(("Original", imagem), ("Canny 25/75", sensivel),
                          ("Canny 60/150", medio), ("Canny 120/240", conservador)))
