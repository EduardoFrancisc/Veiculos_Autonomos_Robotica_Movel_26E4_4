"""
EX1 - Item A: alinhamento de cameras multi-view por homografia (TP2).

Este script usa como base os exemplos independentes da pasta
"exemplos_tp2/Aula_Homografia_Morfologia_40_Independentes" (ORB, BFMatcher,
teste de Lowe, RANSAC, warpPerspective e mosaico panoramico).
"""

from pathlib import Path

import cv2
import numpy as np

# Raiz do repositorio (TP2/EX1/item_a -> raiz).
ROOT = Path(__file__).resolve().parents[3]
EXEMPLOS = ROOT / "exemplos_tp2" / "Aula_Homografia_Morfologia_40_Independentes"
DADOS = EXEMPLOS / "dados"
OUT = Path(__file__).resolve().parent / "resultados" / "main"
OUT.mkdir(parents=True, exist_ok=True)


def ler(nome):
    caminho = DADOS / nome
    imagem = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise FileNotFoundError(f"Imagem ausente ou invalida: {caminho}")
    return imagem


def painel(pares, destino="resultado.png", largura=560, altura=310):
    """Grid de imagens (tuplas (legenda, imagem)) salvo em OUT/destino."""
    quadros = []
    for legenda, imagem in pares:
        if len(imagem.shape) == 2:
            imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (largura, 43), (22, 31, 46), -1)
        cv2.putText(img, legenda[:43], (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.66, (255, 255, 255), 2, cv2.LINE_AA)
        quadros.append(img)
    ncol = min(3, len(quadros))
    while len(quadros) % ncol:
        quadros.append(np.full_like(quadros[0], (235, 235, 235)))
    linhas = [np.hstack(quadros[i:i + ncol]) for i in range(0, len(quadros), ncol)]
    resultado = np.vstack(linhas)
    arquivo = OUT / destino
    ok, dados = cv2.imencode(arquivo.suffix if arquivo.suffix else ".png", resultado)
    if not ok:
        raise OSError(f"Falha ao salvar painel: {arquivo}")
    dados.tofile(str(arquivo))
    print("Imagem salva:", arquivo)
    return resultado


def orb_pares(nfeatures=2500):
    """Detecta keypoints/descritores ORB nas duas vistas de transito."""
    a, b = ler("visao_01.png"), ler("visao_02.png")
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=12)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        raise RuntimeError("Nao foram detectados descritores ORB. Use cenas com textura.")
    return a, b, k1, d1, k2, d2


def matches_lowe(d1, d2, ratio=0.80):
    """BFMatcher + knnMatch + filtro pela razao de Lowe."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pares = bf.knnMatch(d2, d1, k=2)  # referencia: vista 02 -> vista 01
    bons = [m for par in pares if len(par) == 2
            for m, n in [par] if m.distance < ratio * n.distance]
    return bons


def estimar_homografia(ratio=0.80, limiar=3.0):
    """Calcula H com cv2.findHomography e RANSAC; retorna inliers e erro medio."""
    a, b, k1, d1, k2, d2 = orb_pares()
    bons = matches_lowe(d1, d2, ratio)
    if len(bons) < 4:
        raise RuntimeError(f"Somente {len(bons)} matches. Precisa de pelo menos 4.")
    src = np.float32([k2[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in bons]).reshape(-1, 1, 2)
    H, mascara = cv2.findHomography(src, dst, cv2.RANSAC, limiar)
    if H is None or mascara is None:
        raise RuntimeError("Homografia nao encontrada; verifique sobreposicao e matches.")
    inliers = mascara.ravel().astype(bool)
    projetados = cv2.perspectiveTransform(src, H)
    erros = np.linalg.norm((projetados - dst).reshape(-1, 2), axis=1)
    erro_medio = float(np.mean(erros[inliers])) if inliers.any() else float("nan")
    return a, b, H, inliers, erro_medio, bons, k1, k2, erros, src, dst


def construir_mosaico(blending=True):
    """Projeta as vistas em um canvas comum e monta o mosaico panoramico."""
    a, b, H, inliers, erro, bons, k1, k2, erros, src, dst = estimar_homografia()
    h1, w1 = a.shape[:2]
    h2, w2 = b.shape[:2]
    c1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    c2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    p2 = cv2.perspectiveTransform(c2, H).reshape(-1, 2)
    todos = np.vstack([c1.reshape(-1, 2), p2])
    xmin, ymin = np.floor(todos.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(todos.max(axis=0)).astype(int)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    tam = (int(xmax - xmin), int(ymax - ymin))
    im1 = cv2.warpPerspective(a, T, tam)
    im2 = cv2.warpPerspective(b, T @ H, tam)
    m1 = cv2.warpPerspective(np.full((h1, w1), 255, np.uint8), T, tam) > 0
    m2 = cv2.warpPerspective(np.full((h2, w2), 255, np.uint8), T @ H, tam) > 0
    mosaico = im1.copy()
    apenas2 = m2 & ~m1
    mosaico[apenas2] = im2[apenas2]
    if blending:
        comum = m1 & m2
        mosaico[comum] = np.clip(
            0.5 * im1[comum].astype(float) + 0.5 * im2[comum].astype(float),
            0, 255).astype(np.uint8)
    return a, b, H, inliers, erro, mosaico, T


def main():
    a, b, H, inliers, erro, mosaico, T = construir_mosaico(blending=True)

    # Resultado numerico exigido: inliers do RANSAC e erro de reprojeção medio.
    print("Inliers RANSAC:", int(inliers.sum()), "de", len(inliers))
    print("Erro de reprojeção médio (px):", round(erro, 4))
    print("Matriz de homografia H (vista 02 -> vista 01):\n", H)
    print("Matriz de translação do canvas T:\n", T)

    arquivo_mosaico = OUT / "mosaico_panoramico.png"
    ok, dados = cv2.imencode(".png", mosaico)
    if not ok:
        raise OSError(f"Falha ao codificar mosaico: {arquivo_mosaico}")
    dados.tofile(str(arquivo_mosaico))
    print("Mosaico panorâmico salvo em:", arquivo_mosaico)

    painel([("Camera 01", a), ("Camera 02", b), ("Mosaico panorâmico", mosaico)],
           "painel_mosaico.png")

    # Comentário técnico: uso de homografia multi-câmera em surround view (360°).
    # Em veículos modernos, as câmeras ao redor (frontal, laterais e traseira)
    # são calibradas e suas imagens adjacentes são alinhadas por homografia
    # (ou modelos geométricos mais completos) para compor uma vista panorâmica
    # ou bird's-eye view ao redor do veículo. A homografia assume, em geral, um
    # plano de referência (solo); objetos fora desse plano geram paralaxe e
    # fantasmas. Por isso, sistemas reais combinam calibração intrínseca/
    # extrínseca, correção de distorção, alinhamento por homografia, blending
    # e, quando necessário, modelos 3D/estéreo para maior precisão.


if __name__ == "__main__":
    main()
    print("EX1 item A concluído em:", OUT)
