"""EXEMPLO 32/32 - Executar Item B durante 100 frames consecutivos

OBJETIVO: Executar Item B durante 100 frames consecutivos.
COMO FUNCIONA: Integra BEV, HLS/Sobel, Canny/Hough, polyfit grau 2, desvio, curvatura e overlay.
ENTRADA: dados/estrada_120_frames.mp4 (amostra didatica sintetica).
SAIDA ESPERADA: MP4 com 100 frames, PNGs de controle e CSV com desvio/raio por frame.
COMO EXECUTAR: python exemplos/32_item_b_100_frames.py
EXPERIMENTE: Inspecione CSV e altere o video de entrada apenas apos calibrar SRC/DST.
DEPENDENCIAS: cv2, numpy, csv (biblioteca padrao); nao importa nenhum outro arquivo .py.
VALIDACAO: CSV informa falhas em vez de inventar medidas. Escalas e pontos sao de exemplo."""

from pathlib import Path
import argparse
import csv
import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
VIDEO_PADRAO = RAIZ / "dados" / "estrada_120_frames.mp4"
PASTA_SAIDA = RAIZ / "saida"
PASTA_SAIDA.mkdir(exist_ok=True, parents=True)
LARGURA, ALTURA = 1280, 720
SRC = np.float32([[570, 430], [710, 430], [1150, 710], [130, 710]])
DST = np.float32([[290, 0], [990, 0], [990, 719], [290, 719]])
M = cv2.getPerspectiveTransform(SRC, DST)
M_INVERSA = cv2.getPerspectiveTransform(DST, SRC)

def criar_mascara(bev):
    # HLS identifica cores; Sobel X identifica transicoes de intensidade.
    hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
    l, s = hls[:, :, 1], hls[:, :, 2]
    cinza = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    grad = np.abs(cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3))
    sobel = np.uint8(255 * grad / (grad.max() + 1e-9))
    amarelo = (s >= 90) & (l >= 75)
    branco = (l >= 165) & (s < 95)
    mascara = np.uint8((amarelo | branco | (sobel >= 23)) * 255)
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE,
                            np.ones((5, 5), dtype=np.uint8))


def detectar_segmentos(mascara):
    # Bordas de Canny -> segmentos da transformacao probabilistica de Hough.
    arestas = cv2.Canny(mascara, 60, 150)
    linhas = cv2.HoughLinesP(arestas, 1, np.pi / 180, 35,
                            minLineLength=35, maxLineGap=70)
    view = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    if linhas is not None:
        for x1, y1, x2, y2 in linhas.reshape(-1, 4):
            cv2.line(view, (x1, y1), (x2, y2), (0, 120, 255), 2)
    return linhas, view


def painel_4(original, bev, mascara, hough):
    pares = [("1. ORIGINAL", original), ("2. BEV", bev),
             ("3. HLS + SOBEL", mascara), ("4. CANNY + HOUGH", hough)]
    blocos = []
    for titulo, img in pares:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (640, 360))
        cv2.rectangle(img, (0, 0), (640, 38), (20, 30, 45), -1)
        cv2.putText(img, titulo, (15, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        blocos.append(img)
    return np.vstack((np.hstack(blocos[:2]), np.hstack(blocos[2:])))

def ajustar_faixas(mascara, segmentos):
    """Agrupa pixels suportados pelas linhas de Hough e faz dois polyfit."""
    suporte = np.zeros_like(mascara)
    if segmentos is not None:
        for x1, y1, x2, y2 in segmentos.reshape(-1, 4):
            medio_x = (x1 + x2) / 2
            if abs(x2 - x1) <= 120 and (medio_x < 510 or medio_x > 770):
                cv2.line(suporte, (x1, y1), (x2, y2), 255, 13)
    suporte = cv2.bitwise_and(suporte, mascara)
    if cv2.countNonZero(suporte) < 250:
        suporte = mascara.copy()
    ys, xs = np.nonzero(suporte)
    valido = (ys > 80) & (((xs > 195) & (xs < 515)) |
                             ((xs > 765) & (xs < 1085)))
    ys, xs = ys[valido], xs[valido]
    esq, dir_ = xs < LARGURA // 2, xs > LARGURA // 2
    if np.count_nonzero(esq) < 50 or np.count_nonzero(dir_) < 50:
        return None, None
    return np.polyfit(ys[esq], xs[esq], 2), np.polyfit(ys[dir_], xs[dir_], 2)


def medidas_em_metros(esq, dir_):
    """Desvio lateral e raio de curvatura: escalas exigidas na tarefa."""
    mx = 3.7 / 700
    my = 30 / 720
    y = ALTURA - 1
    centro = (np.polyval(esq, y) + np.polyval(dir_, y)) / 2
    desvio = (LARGURA / 2 - centro) * mx

    def raio(fit):
        a = fit[0] * mx / my**2
        b = fit[1] * mx / my
        return (1 + (2 * a * y * my + b)**2)**1.5 / (abs(2 * a) + 1e-12)

    return float(desvio), float((raio(esq) + raio(dir_)) / 2)


def preencher_faixa(frame, esq, dir_):
    """Gera poligono no BEV, faz projecao inversa e mistura semitransparente."""
    y = np.arange(ALTURA, dtype=np.float64)
    x_e = np.clip(np.polyval(esq, y), 0, LARGURA - 1)
    x_d = np.clip(np.polyval(dir_, y), 0, LARGURA - 1)
    p_e = np.column_stack((x_e.astype(int), y.astype(int))).astype(np.int32)
    p_d = np.column_stack((x_d.astype(int), y.astype(int))).astype(np.int32)
    colorida_bev = np.zeros_like(frame)
    cv2.fillPoly(colorida_bev, [np.vstack((p_e, p_d[::-1]))], (30, 190, 45))
    cv2.polylines(colorida_bev, [p_e], False, (255, 255, 255), 7)
    cv2.polylines(colorida_bev, [p_d], False, (0, 235, 255), 7)
    colorida_camera = cv2.warpPerspective(colorida_bev, M_INVERSA,
                                          (LARGURA, ALTURA))
    mascara_validade = cv2.cvtColor(colorida_camera, cv2.COLOR_BGR2GRAY) > 0
    final = frame.copy()
    mescla = cv2.addWeighted(frame, 0.65, colorida_camera, 0.35, 0)
    final[mascara_validade] = mescla[mascara_validade]
    return final

def processar_frame(frame):
    """Pipeline completo: BEV -> HLS/Sobel -> Canny/Hough -> polyfit -> metricas."""
    frame = cv2.resize(frame, (LARGURA, ALTURA))
    bev = cv2.warpPerspective(frame, M, (LARGURA, ALTURA))
    mascara = criar_mascara(bev)
    segmentos, hough = detectar_segmentos(mascara)
    esq, dir_ = ajustar_faixas(mascara, segmentos)
    painel = painel_4(frame, bev, mascara, hough)
    if esq is None:
        # Nao inventar valores caso a deteccao falhe.
        cv2.putText(frame, "Faixas nao identificadas", (35, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return frame, painel, None, None
    desvio, raio = medidas_em_metros(esq, dir_)
    final = preencher_faixa(frame, esq, dir_)
    cv2.putText(final, f"Desvio: {desvio:+.3f} m", (35, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    cv2.putText(final, f"Raio de curvatura: {raio:.1f} m", (35, 142),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    return final, painel, desvio, raio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(VIDEO_PADRAO),
                        help="Substitui o video sintetico por outro MP4 calibrado")
    args = parser.parse_args()
    captura = cv2.VideoCapture(args.video)
    if not captura.isOpened():
        raise FileNotFoundError(f"Video nao abriu: {args.video}")
    caminho = PASTA_SAIDA / "32_item_b_100_frames.mp4"
    escritor = cv2.VideoWriter(str(caminho),
                               cv2.VideoWriter_fourcc(*"mp4v"), 20,
                               (LARGURA, ALTURA))
    if not escritor.isOpened():
        raise RuntimeError("Nao foi possivel abrir VideoWriter mp4v.")
    linhas_csv = []
    deteccoes = 0
    try:
        for indice in range(100):
            ok, frame = captura.read()
            if not ok:
                raise RuntimeError(f"Video tem menos que 100 frames: {indice} lidos.")
            resultado, painel, desvio, raio = processar_frame(frame)
            escritor.write(resultado)
            if indice in (0, 99):
                cv2.imwrite(str(PASTA_SAIDA / f"32_resultado_frame_{indice:03d}.png"), resultado)
                cv2.imwrite(str(PASTA_SAIDA / f"32_painel_frame_{indice:03d}.png"), painel)
            if desvio is not None:
                deteccoes += 1
                print(f"Frame {indice+1:03d}/100 | desvio={desvio:+.4f} m | raio={raio:.1f} m")
            else:
                print(f"Frame {indice+1:03d}/100 | deteccao indisponivel")
            linhas_csv.append((indice, desvio, raio))
    finally:
        captura.release()
        escritor.release()
    with (PASTA_SAIDA / "32_metricas.csv").open("w", newline="", encoding="utf-8") as arquivo:
        tabela = csv.writer(arquivo)
        tabela.writerow(["frame", "desvio_m", "raio_curvatura_m"])
        tabela.writerows(linhas_csv)
    print(f"Item B: 100 frames consecutivos; deteccoes validas: {deteccoes}/100")
    print(f"Video: {caminho}")
    print("Metricas: saida/32_metricas.csv")


if __name__ == "__main__":
    main()
