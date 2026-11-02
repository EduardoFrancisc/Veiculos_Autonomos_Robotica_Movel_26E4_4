"""EXEMPLO 23/32 - Executar Item A durante 50 frames consecutivos

OBJETIVO: Executar Item A durante 50 frames consecutivos.
COMO FUNCIONA: Para cada frame, aplica BEV, HLS+Sobel, Canny, Hough e reprojecao inversa.
ENTRADA: dados/estrada_120_frames.mp4 (amostra didatica sintetica).
SAIDA ESPERADA: MP4 de 50 frames, paineis PNG e overlays PNG; contagem no terminal.
COMO EXECUTAR: python exemplos/23_item_a_50_frames.py
EXPERIMENTE: Inspecione os frames 0 e 49. Para video externo: --video caminho.mp4.
DEPENDENCIAS: cv2, numpy; nao importa nenhum outro arquivo .py.
ATENCAO: os quatro pontos foram escolhidos para a amostra do pacote."""

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

def processar_frame(frame):
    """Retorna o painel e a sobreposicao de segmentos na perspectiva original."""
    frame = cv2.resize(frame, (LARGURA, ALTURA))
    bev = cv2.warpPerspective(frame, M, (LARGURA, ALTURA))
    mascara = criar_mascara(bev)
    segmentos, vista_hough = detectar_segmentos(mascara)
    camada_bev = np.zeros_like(frame)
    if segmentos is not None:
        for x1, y1, x2, y2 in segmentos.reshape(-1, 4):
            medio = (x1 + x2) / 2
            if abs(x2 - x1) <= 120 and (medio < 510 or medio > 770):
                cv2.line(camada_bev, (x1, y1), (x2, y2), (0, 230, 0), 7)
    camada_camera = cv2.warpPerspective(camada_bev, M_INVERSA,
                                        (LARGURA, ALTURA))
    sobreposicao = cv2.addWeighted(frame, 1, camada_camera, 0.9, 0)
    painel = painel_4(frame, bev, mascara, vista_hough)
    qtd = 0 if segmentos is None else len(segmentos)
    return painel, sobreposicao, qtd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(VIDEO_PADRAO))
    args = parser.parse_args()
    captura = cv2.VideoCapture(args.video)
    if not captura.isOpened():
        raise FileNotFoundError(f"Video nao abriu: {args.video}")
    caminho_mp4 = PASTA_SAIDA / "23_item_a_50_frames.mp4"
    escritor = cv2.VideoWriter(str(caminho_mp4),
                               cv2.VideoWriter_fourcc(*"mp4v"), 20,
                               (LARGURA, ALTURA))
    if not escritor.isOpened():
        raise RuntimeError("VideoWriter mp4v nao esta disponivel.")
    total = 0
    try:
        for indice in range(50):
            valido, frame = captura.read()
            if not valido:
                raise RuntimeError(f"Video acabou antes de 50 frames: {indice} lidos.")
            painel, resultado, qtd = processar_frame(frame)
            escritor.write(painel)
            if indice in (0, 49):
                cv2.imwrite(str(PASTA_SAIDA / f"23_painel_frame_{indice:03d}.png"), painel)
                cv2.imwrite(str(PASTA_SAIDA / f"23_overlay_frame_{indice:03d}.png"), resultado)
            print(f"Frame {indice + 1:03d}/50 | segmentos de Hough: {qtd}")
            total += 1
    finally:
        captura.release()
        escritor.release()
    print(f"Item A concluido: {total} frames continuos; saida: {caminho_mp4}")


if __name__ == "__main__":
    main()
