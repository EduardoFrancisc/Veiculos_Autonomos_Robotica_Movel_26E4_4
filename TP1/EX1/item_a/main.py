import sys
from pathlib import Path


PASTA_APOIO = Path(__file__).resolve().parents[2] / "scripts_aula" / "OpenCV_Sistemas_Autonomos_36_Exemplos" / "exemplos"
sys.path.insert(0, str(PASTA_APOIO))

import cv2
from TP1.EX1.item_a.apoio import (
    iniciar,
    carregar_imagem,
    metadados,
    cinza,
    quadrantes,
    clahe_colorido,
    painel,
    histograma_comparado,
    salvar_imagem,
)

# 1) Carrega imagem de camera frontal de veiculo (KITTI por padrao; aceita --imagem).
a, _ = iniciar(__file__)

out = Path(__file__).resolve().parent / "resultados" / Path(__file__).stem
out.mkdir(parents=True, exist_ok=True)

img = carregar_imagem(a.imagem)

# 2) Metadados: resolucao, dtype, canais e tamanho em bytes (impresso no terminal).
m = metadados(img, a.imagem)

# 3) Quadrantes e brilho medio de cada regiao.
y = cinza(img)
brilho = {k: float(r.mean()) for k, r in quadrantes(y).items()}
print("Brilho medio por quadrante (cinza, 0-255):", brilho)

# 4) Equalizacao adaptativa (CLAHE) e comparacao de histogramas em um unico plot.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
eq = clahe.apply(y)
colorida, _l, _leq = clahe_colorido(img)

painel(
    [img, colorida],
    ["Original", "Apos CLAHE (canal L do Lab)"],
    out / "original_equalizada.png",
    a,
)
histograma_comparado(y, eq, out / "histogramas.png", a)
salvar_imagem(out / "equalizada_cinza.png", eq)
salvar_imagem(out / "equalizada_colorida.png", colorida)

print("Item A concluido em:", out)