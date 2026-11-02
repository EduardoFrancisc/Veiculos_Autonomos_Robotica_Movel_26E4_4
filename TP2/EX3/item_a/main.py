"""
EX3 - Item A: HoughCircles para sinais circulares e HoughLinesP para faixas (TP2).

Baseado nos exemplos independentes de
"exemplos_tp2/Aula_Formas_Keypoints_40_Independentes".
"""

from pathlib import Path
from time import perf_counter
import csv

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
    ok, dados = cv2.imencode(arquivo.suffix if arquivo.suffix else ".jpg", imagem)
    if not ok:
        raise OSError(f"Falha ao salvar imagem: {arquivo}")
    dados.tofile(str(arquivo))
    print("Imagem salva:", arquivo)
    return arquivo


def painel(imagens, nomes, arquivo="painel.jpg", colunas=2):
    largura, altura = 520, 320
    blocos = []
    for img, nome in zip(imagens, nomes):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)
        cv2.rectangle(vis, (0, 0), (largura, 32), (14, 25, 41), -1)
        cv2.putText(vis, nome[:56], (12, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (245, 245, 245), 2, cv2.LINE_AA)
        blocos.append(vis)
    while len(blocos) % colunas:
        blocos.append(np.full((altura, largura, 3), 240, np.uint8))
    linhas = [np.hstack(blocos[i:i + colunas]) for i in range(0, len(blocos), colunas)]
    return salvar(arquivo, np.vstack(linhas))


def preparar_circulos(img):
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cinza, (7, 7), 1.5)


def confianca_heuristica(gray, x, y, r):
    e = cv2.Canny(gray, 80, 160)
    dilatada = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
    angulos = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    xs = np.clip(np.rint(x + r * np.cos(angulos)).astype(int), 0, e.shape[1] - 1)
    ys = np.clip(np.rint(y + r * np.sin(angulos)).astype(int), 0, e.shape[0] - 1)
    return float(np.mean(dilatada[ys, xs] > 0))


def detectar_circulos(imagem, dp=1.2, minDist=35, param1=135, param2=32,
                      minRadius=12, maxRadius=80, somente_vermelhos=True):
    gray = preparar_circulos(imagem)
    t0 = perf_counter()
    raw = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                           param1=param1, param2=param2,
                           minRadius=minRadius, maxRadius=maxRadius)
    ms = (perf_counter() - t0) * 1000
    achados = [] if raw is None else np.rint(raw[0]).astype(int).tolist()
    if somente_vermelhos and achados:
        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, (0, 65, 70), (13, 255, 255)) | cv2.inRange(
            hsv, (168, 65, 70), (179, 255, 255))

        def apoio_vermelho(x, y, r):
            aa = np.linspace(0, 2 * np.pi, 120, endpoint=False)
            valores = []
            for frac in (.80, .87, .94, 1.01, 1.08):
                xs = np.clip(np.rint(x + frac * r * np.cos(aa)).astype(int),
                             0, mascara.shape[1] - 1)
                ys = np.clip(np.rint(y + frac * r * np.sin(aa)).astype(int),
                             0, mascara.shape[0] - 1)
                valores.append(float(np.mean(mascara[ys, xs] > 0)))
            return max(valores)

        candidatos = sorted([(apoio_vermelho(x, y, r), (x, y, r))
                            for x, y, r in achados], reverse=True)
        retidos = []
        for score, (x, y, r) in candidatos:
            if score < .16:
                continue
            if any((x - u) ** 2 + (y - v) ** 2 <
                   max(15, int(min(r, rr) * .75)) ** 2
                   for u, v, rr in retidos):
                continue
            retidos.append((x, y, r))
        achados = retidos
    return gray, achados, ms


def desenhar_circulos(img, circulos, gray=None):
    vis = img.copy()
    for j, (x, y, r) in enumerate(circulos, 1):
        cv2.circle(vis, (x, y), r, (0, 240, 0), 2, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)
        conf = confianca_heuristica(gray if gray is not None else preparar_circulos(img),
                                    x, y, r)
        cv2.putText(vis, f"{j} s={conf:.2f}", (max(2, x - r), max(20, y - r - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 2, cv2.LINE_AA)
    return vis


def desenhar_linhas(img, low=60, high=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), low, high)
    roi = np.zeros_like(canny)
    h, w = canny.shape
    pol = np.array([[(0, h), (w, h), (int(.68 * w), int(.60 * h)),
                     (int(.32 * w), int(.60 * h))]], np.int32)
    cv2.fillPoly(roi, pol, 255)
    canny = cv2.bitwise_and(canny, roi)
    linhas = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=28,
                             minLineLength=35, maxLineGap=45)
    vis = img.copy()
    n = 0
    if linhas is not None:
        for x1, y1, x2, y2 in linhas[:, 0]:
            if abs(y2 - y1) < 7:
                continue
            cv2.line(vis, (x1, y1), (x2, y2), (0, 220, 240), 2, cv2.LINE_AA)
            n += 1
    return vis, n


def main():
    configs = [
        ("sensivel", 1.0, 23, 95, 23),
        ("equilibrado", 1.2, 40, 135, 29),
        ("restrito", 1.5, 60, 170, 40),
    ]

    rows = []
    painel_imgs = []
    painel_labs = []

    for i in range(1, 6):
        img = carregar(f"rua_{i:02d}.jpg")
        print(f"\n===== Cena {i} =====")
        for nome, dp, minDist, param1, param2 in configs:
            gray, circulos, ms = detectar_circulos(
                img, dp=dp, minDist=minDist, param1=param1, param2=param2,
                somente_vermelhos=True)
            print(f"Config {nome}: dp={dp}, minDist={minDist}, param1={param1}, "
                  f"param2={param2}; {len(circulos)} circulo(s); tempo={ms:.2f} ms")
            for x, y, r in circulos:
                conf = confianca_heuristica(gray, x, y, r)
                print(f"  centro=({x}, {y}), raio={r}, confianca_heuristica={conf:.3f}")
                rows.append([i, nome, dp, minDist, param1, param2, x, y, r,
                             round(conf, 3), round(ms, 3)])

            if nome == "equilibrado":
                vis_linhas, n_linhas = desenhar_linhas(img)
                vis_final = desenhar_circulos(vis_linhas, circulos, gray)
                salvar(f"cena_{i:02d}_circulos_linhas.jpg", vis_final)
                painel_imgs.append(vis_final)
                painel_labs.append(f"Cena {i} | C={len(circulos)} L={n_linhas}")

    with (OUT / "tabela_hough.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cena", "config", "dp", "minDist", "param1", "param2",
                    "cx", "cy", "raio", "confianca_heuristica", "tempo_ms"])
        w.writerows(rows)

    painel(painel_imgs, painel_labs, "painel_circulos_linhas.jpg", colunas=2)

    # Comentario tecnico: HoughCircles/HoughLinesP em percepcao veicular.
    # HoughCircles detecta sinais circulares (placas) via acumulador de
    # circunferencias; HoughLinesP detecta faixas. A combinacao permite
    # localizar simultaneamente sinalizacao vertical e horizontal. Os
    # parametros controlam sensibilidade/robustez: dp afeta resolucao do
    # acumulador, minDist evita duplicatas, param1/param2 controlam bordas e
    # votos. A confianca e heuristica (apoio de borda), nao probabilidade
    # do OpenCV. Validacao deve usar imagens reais anotadas e multiplas
    # condicoes de iluminacao/distancia.


if __name__ == "__main__":
    main()
    print("EX3 item A concluido em:", OUT)
