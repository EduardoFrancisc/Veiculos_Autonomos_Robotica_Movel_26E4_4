import argparse
import sys
from pathlib import Path

PACOTE = Path(__file__).resolve().parents[3] / "scripts_aula" / "Deteccao_Faixas_Independentes" / "exemplos"
sys.path.insert(0, str(PACOTE))

import cv2
import importlib.util

spec = importlib.util.spec_from_file_location("det_faixas", PACOTE / "23_item_a_50_frames.py")
det = importlib.util.module_from_spec(spec)
spec.loader.exec_module(det)

FRAMES = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(det.VIDEO_PADRAO))
    args = parser.parse_args()

    out = Path(__file__).resolve().parent / "resultados" / "main"
    out.mkdir(parents=True, exist_ok=True)

    captura = cv2.VideoCapture(args.video)
    if not captura.isOpened():
        raise FileNotFoundError(f"Video nao abriu: {args.video}")

    caminho_mp4 = out / "item_a_50_frames.mp4"
    escritor = cv2.VideoWriter(str(caminho_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                               (det.LARGURA, det.ALTURA))
    if not escritor.isOpened():
        raise RuntimeError("VideoWriter mp4v nao disponivel.")

    total = 0
    try:
        for indice in range(FRAMES):
            valido, frame = captura.read()
            if not valido:
                raise RuntimeError(f"Video acabou antes de {FRAMES} frames: {indice} lidos.")
            painel, sobreposicao, qtd = det.processar_frame(frame)
            escritor.write(painel)
            if indice in (0, FRAMES - 1):
                cv2.imwrite(str(out / f"painel_frame_{indice:03d}.png"), painel)
                cv2.imwrite(str(out / f"overlay_frame_{indice:03d}.png"), sobreposicao)
            print(f"Frame {indice + 1:03d}/{FRAMES} | segmentos de Hough: {qtd}")
            total += 1
    finally:
        captura.release()
        escritor.release()

    print(f"Item A concluido: {total} frames continuos; saida: {caminho_mp4}")


if __name__ == "__main__":
    main()