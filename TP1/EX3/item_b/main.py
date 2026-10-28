import argparse
import csv
import importlib.util
import sys
from pathlib import Path

PACOTE = Path(__file__).resolve().parents[3] / "scripts_aula" / "Deteccao_Faixas_Independentes" / "exemplos"
sys.path.insert(0, str(PACOTE))

spec = importlib.util.spec_from_file_location("det_b", PACOTE / "32_item_b_100_frames.py")
det = importlib.util.module_from_spec(spec)
spec.loader.exec_module(det)

FRAMES = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(det.VIDEO_PADRAO))
    args = parser.parse_args()

    out = Path(__file__).resolve().parent / "resultados" / "main"
    out.mkdir(parents=True, exist_ok=True)

    import cv2
    captura = cv2.VideoCapture(args.video)
    if not captura.isOpened():
        raise FileNotFoundError(f"Video nao abriu: {args.video}")

    caminho = out / "item_b_100_frames.mp4"
    escritor = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                               (det.LARGURA, det.ALTURA))
    if not escritor.isOpened():
        raise RuntimeError("Nao foi possivel abrir VideoWriter mp4v.")

    linhas_csv = []
    deteccoes = 0
    try:
        for indice in range(FRAMES):
            ok, frame = captura.read()
            if not ok:
                raise RuntimeError(f"Video tem menos que {FRAMES} frames: {indice} lidos.")
            resultado, painel, desvio, raio = det.processar_frame(frame)
            escritor.write(resultado)
            if indice in (0, FRAMES - 1):
                cv2.imwrite(str(out / f"resultado_frame_{indice:03d}.png"), resultado)
                cv2.imwrite(str(out / f"painel_frame_{indice:03d}.png"), painel)
            if desvio is not None:
                deteccoes += 1
                print(f"Frame {indice + 1:03d}/{FRAMES} | desvio={desvio:+.4f} m | raio={raio:.1f} m")
            else:
                print(f"Frame {indice + 1:03d}/{FRAMES} | deteccao indisponivel")
            linhas_csv.append((indice, desvio, raio))
    finally:
        captura.release()
        escritor.release()

    with (out / "metricas.csv").open("w", newline="", encoding="utf-8") as arquivo:
        tabela = csv.writer(arquivo)
        tabela.writerow(["frame", "desvio_m", "raio_curvatura_m"])
        tabela.writerows(linhas_csv)

    print(f"Item B: {FRAMES} frames consecutivos; deteccoes validas: {deteccoes}/{FRAMES}")
    print(f"Video: {caminho}")
    print("Metricas: resultados/main/metricas.csv")


if __name__ == "__main__":
    main()