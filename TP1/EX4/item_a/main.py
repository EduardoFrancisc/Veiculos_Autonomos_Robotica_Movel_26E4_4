import importlib.util
import sys
from pathlib import Path

PACOTE = Path(__file__).resolve().parents[3] / "scripts_aula" / "Transformacoes_2D_40_Independentes_Comentados" / "exemplos"
sys.path.insert(0, str(PACOTE))

spec13 = importlib.util.spec_from_file_location("t13", PACOTE / "13_ordem_transformacoes.py")
t13 = importlib.util.module_from_spec(spec13)
spec13.loader.exec_module(t13)

spec05 = importlib.util.spec_from_file_location("t05", PACOTE / "05_rotacao_centro.py")
t05 = importlib.util.module_from_spec(spec05)
spec05.loader.exec_module(t05)

import cv2
import numpy as np

out = Path(__file__).resolve().parent / "resultados" / "main"
out.mkdir(parents=True, exist_ok=True)


caminho_img = Path(__file__).resolve().parents[3] / "scripts_aula" / "Transformacoes_2D_40_Independentes_Comentados" / "dados" / "frontal_kitti.png"
img = cv2.imdecode(__import__("numpy").fromfile(str(caminho_img), dtype=__import__("numpy").uint8), cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(caminho_img)
h, w = img.shape[:2]

def salvar(nome, im):
    p = out / nome
    cv2.imwrite(str(p), im)
    print("Salvo:", p)
    return p

# 1) Translacao: desloca +80 px horizontal, +30 px vertical.
M_trans = np.float32([[1, 0, 80], [0, 1, 30]])
print("M_translacao=\n", M_trans)
trans = t13.affine_warp(img, M_trans)
salvar("01_translacao.png", trans)

# 2) Rotacao em torno do centro (+10 graus).
rot, M_rot = t05.rotate_zoom(img, 10, 1.0)
print("M_rotacao_centro=\n", M_rot)
salvar("02_rotacao.png", rot)

# 3) Escala (zoom digital) centrado (fator 1.2).
M_zoom = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.2)
print("M_zoom=\n", M_zoom)
zoom = cv2.warpAffine(img, M_zoom, (w, h), borderMode=cv2.BORDER_REFLECT_101)
salvar("03_zoom.png", zoom)

# 4) Transformacao afim customizada 2x3.
M_afim = np.float32([[1.0, 0.12, -25], [0.02, 0.98, 10]])
print("M_afim_customizada=\n", M_afim)
afim = t13.affine_warp(img, M_afim)
salvar("04_afim_customizada.png", afim)

# Composicao: rotacao + translacao via multiplicacao de matrizes (homogeneas).
T = t13.H([[1, 0, 70], [0, 1, 0]])
R = t13.H(cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1))
RT = R @ T   # aplica T primeiro, depois R
TR = T @ R   # aplica R primeiro, depois T
print("R@T=\n", RT)
print("T@R=\n", TR)
print("Iguais?", bool(np.allclose(RT, TR)))
comp_rt = t13.affine_warp(img, RT[:2])
comp_tr = t13.affine_warp(img, TR[:2])
salvar("05_composicao_RT.png", comp_rt)
salvar("06_composicao_TR.png", comp_tr)

# Painel comparativo.
painel = np.hstack([cv2.resize(img, (320, 180)), cv2.resize(trans, (320, 180)),
                    cv2.resize(rot, (320, 180)), cv2.resize(zoom, (320, 180)),
                    cv2.resize(afim, (320, 180)), cv2.resize(comp_rt, (320, 180))])
cv2.imwrite(str(out / "painel_transformacoes.png"), painel)

# Comentarios de aplicacao no pre-processamento para redes de direcao autonoma.
print("""
APLICACAO NO PRE-PROCESSAMENTO (CNN de direcao autonoma):
- Translacao: simula deslocamento aparente da camera / pequenas perturbacoes espaciais;
  util em data augmentation, mas rótulos geometricos (angulo de direcao, bounding boxes)
  devem acompanhar a mudanca.
- Rotacao em torno do centro: simula variacoes de orientacao da camera; util em
  augmentation para robustez, mas não equivale a mudar fisicamente a direcao do veiculo.
- Escala (zoom digital): simula mudanca de distancia focal / aproximacao; util para
  multi-escala e robustez a variacoes de campo de visao.
- Transformacao afim customizada: corrige cisalhamento/escala/anisotropia de cameras;
  util em calibracao e normalizacao de entrada para a rede.
- Composicao de transformacoes: a ordem importa (R@T != T@R); em pipelines de
  alinhamento, alterar a ordem desloca pontos e objetos para posicoes incorretas.
""")
print("EX4 item A concluido em:", out)