"""
EX3 - Item B: Harris, Shi-Tomasi e fluxo optico Lucas-Kanade (TP2).

Baseado nos exemplos independentes de
"exemplos_tp2/Aula_Formas_Keypoints_40_Independentes".
"""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXEMPLOS = ROOT / "exemplos_tp2" / "Aula_Formas_Keypoints_40_Independentes"
DADOS = EXEMPLOS / "dados"
OUT = Path(__file__).resolve().parent / "resultados" / "main"
OUT.mkdir(parents=True, exist_ok=True)


def carregar(nome):
    caminho = DADOS / nome
    imagem = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise FileNotFoundError(f"Imagem ausente ou invalida: {caminho}")
    return imagem


def salvar(nome, imagem):
    arquivo = OUT / nome
    ok, dados = cv2.imencode(arquivo.suffix if arquivo.suffix else ".png", imagem)
    if not ok:
        raise OSError(f"Falha ao salvar imagem: {arquivo}")
    dados.tofile(str(arquivo))
    print("Imagem salva:", arquivo)
    return arquivo


def detectar_harris(img, max_pontos=200):
    gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    t0 = perf_counter()
    r = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    r = cv2.dilate(r, None)
    limite = float(r.max()) * 0.01
    coords = np.argwhere(r > limite)
    coords = sorted(coords, key=lambda yx: float(r[yx[0], yx[1]]), reverse=True)
    pontos = []
    for y, x in coords:
        if all((x - a) ** 2 + (y - b) ** 2 >= 7 ** 2 for a, b in pontos):
            pontos.append((int(x), int(y)))
            if len(pontos) >= max_pontos:
                break
    ms = (perf_counter() - t0) * 1000
    return pontos, ms


def detectar_shi(img, max_pontos=200, quality=0.01, distance=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = perf_counter()
    pontos = cv2.goodFeaturesToTrack(gray, maxCorners=max_pontos,
                                    qualityLevel=quality, minDistance=distance,
                                    blockSize=7)
    ms = (perf_counter() - t0) * 1000
    if pontos is None:
        pontos = np.empty((0, 1, 2), np.float32)
    else:
        pontos = pontos.astype(np.float32)
    return pontos, ms


def desenhar_pontos(img, pontos, cor=(20, 240, 230)):
    vis = img.copy()
    pts = np.asarray(pontos).reshape(-1, 2)
    for x, y in pts:
        cv2.circle(vis, (round(float(x)), round(float(y))), 3, cor, -1, cv2.LINE_AA)
    return vis


def fluxo_lk(a, b, pontos=None, win=(21, 21), niveis=3):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if pontos is None:
        pontos = cv2.goodFeaturesToTrack(ga, maxCorners=200, qualityLevel=0.01,
                                         minDistance=8, blockSize=7)
    if pontos is None or not len(pontos):
        return np.empty((0, 2)), np.empty((0, 2)), 0.0
    t0 = perf_counter()
    p1, status, erro = cv2.calcOpticalFlowPyrLK(
        ga, gb, pontos, None, winSize=win, maxLevel=niveis,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    ms = (perf_counter() - t0) * 1000
    if p1 is None:
        return np.empty((0, 2)), np.empty((0, 2)), ms
    mask = status.reshape(-1) == 1
    return pontos.reshape(-1, 2)[mask], p1.reshape(-1, 2)[mask], ms


def vetores(a, p0, p1):
    vis = a.copy()
    for (x, y), (u, v) in zip(p0, p1):
        cv2.arrowedLine(vis, (round(x), round(y)), (round(u), round(v)),
                         (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.36)
        cv2.circle(vis, (round(x), round(y)), 2, (20, 20, 220), -1, cv2.LINE_AA)
    return vis


def main():
    # 1) Harris vs Shi-Tomasi em cena de trafego.
    img = carregar("rua_01.jpg")
    pontos_harris, t_harris = detectar_harris(img, max_pontos=200)
    pontos_shi, t_shi = detectar_shi(img, max_pontos=200)

    print("Harris Corner Detector:")
    print(f"  keypoints={len(pontos_harris)}, tempo={t_harris:.2f} ms")
    print("Shi-Tomasi (goodFeaturesToTrack):")
    print(f"  keypoints={len(pontos_shi)}, tempo={t_shi:.2f} ms")

    vis_harris = desenhar_pontos(img, pontos_harris, cor=(0, 0, 255))
    vis_shi = desenhar_pontos(img, pontos_shi, cor=(0, 255, 0))
    salvar("harris_keypoints.png", vis_harris)
    salvar("shi_tomasi_keypoints.png", vis_shi)
    painel = np.hstack([cv2.resize(vis_harris, (420, 260)),
                        cv2.resize(vis_shi, (420, 260))])
    salvar("painel_keypoints.png", painel)

    # 2) Fluxo optico Lucas-Kanade entre frames consecutivos.
    a = carregar("frame_00.png")
    b = carregar("frame_01.png")
    p0, p1, t_lk = fluxo_lk(a, b, pontos=pontos_shi if len(pontos_shi) else None)
    vis_fluxo = vetores(a, p0, p1)
    print("\nFluxo optico Lucas-Kanade:")
    print(f"  pontos rastreados={len(p0)}, tempo={t_lk:.2f} ms")
    salvar("fluxo_optico_lk.png", vis_fluxo)

    # 3) Discussao de estimativa de velocidade relativa.
    # O fluxo optico fornece deslocamento em pixels/frame para pontos rastreados.
    # Com a calibracao intrinseca da camera, a distancia focal e a altura da
    # camera, pode-se converter deslocamento de pixels em movimento angular e,
    # combinado com profundidade (estereo, radar ou suposicao de plano do solo),
    # estimar velocidade relativa de obstaculos. Pontos com fluxo grande e
    # coerente indicam aproximacao rapida; a direcao do vetor indica movimento
    # relativo. Limitacoes: oclusao, iluminacao, textura pobre e movimento da
    # propria camera exigem compensacao egomotion e filtragem robusta.

    print("\nConclusao: Harris e Shi-Tomasi detectam keypoints; Shi-Tomasi alimenta "
          "LK para rastreamento. Fluxo optico permite estimar movimento relativo "
          "de obstaculos quando calibrado com profundidade e egomotion.")


if __name__ == "__main__":
    main()
    print("EX3 item B concluido em:", OUT)
