"""
EX4 - Item B: YOLOv8n integrado ao loop do MetaDrive (TP2).

Baseado nos exemplos independentes de
"exemplos_tp2/MetaDrive_YOLO_40_Independentes/exemplos"
(22_carregar_yolov8n, 26_desenhar_boxes, 28_yolo_20_steps,
 31_baseline_500_steps, 33_comparar_fps, 34_frame_skip_2,
 36_yolo_resolucao_320 e 39_grafico_series_temporais).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from collections import Counter, defaultdict

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

ROOT = Path(__file__).resolve().parents[3]
EXEMPLOS = ROOT / "exemplos_tp2" / "MetaDrive_YOLO_40_Independentes"
DADOS = EXEMPLOS / "dados"
OUT = Path(__file__).resolve().parent / "resultados" / "main"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES_TRAFEGO = {"person", "bicycle", "car", "motorcycle", "bus", "truck",
                   "traffic light", "stop sign"}


def salvar(nome, imagem):
    caminho = OUT / nome
    if imagem is None:
        raise ValueError(f"Imagem invalida: {nome}")
    ok, dados = cv2.imencode(caminho.suffix if caminho.suffix else ".png", imagem)
    if not ok:
        raise OSError(f"Falha ao gravar {caminho}")
    dados.tofile(str(caminho))
    print("Imagem salva:", caminho)
    return caminho


def salvar_csv(nome, registros):
    caminho = OUT / nome
    if not registros:
        raise ValueError("Sem registros.")
    colunas = list(registros[0].keys())
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colunas)
        w.writeheader()
        w.writerows(registros)
    print("CSV:", caminho, "| registros:", len(registros))
    return caminho


def escrever_texto(nome, texto):
    p = OUT / nome
    p.write_text(texto, encoding="utf-8")
    print("Relatorio:", p)
    return p


def abrir_metadrive(render=True, largura=640, altura=360, semente=42, densidade=0.15):
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        from metadrive.component.sensors.rgb_camera import RGBCamera
    except ImportError as e:
        raise RuntimeError("MetaDrive nao instalado.") from e
    cfg = {
        "use_render": bool(render),
        "image_observation": True,
        "norm_pixel": False,
        "stack_size": 1,
        "sensors": {"rgb_camera": (RGBCamera, largura, altura)},
        "vehicle_config": {"image_source": "rgb_camera"},
        "traffic_density": float(densidade),
        "horizon": 1100,
        "start_seed": int(semente),
        "num_scenarios": 1024,
        "log_level": 50,
    }
    return MetaDriveEnv(cfg)


def obter_frame_bgr(obs):
    if not isinstance(obs, dict) or "image" not in obs:
        raise RuntimeError(f"Observacao inesperada: {type(obs)}")
    frame = np.asarray(obs["image"])
    if frame.ndim == 4:
        frame = frame[..., -1]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise RuntimeError(f"Formato inesperado: {frame.shape}")
    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * (255.0 if float(np.max(frame)) <= 1.01 else 1.0),
                        0, 255).astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def sequencia_acoes(n, semente=42):
    rng = np.random.default_rng(semente)
    return [(float(rng.uniform(-0.12, 0.12)),
             float(rng.uniform(0.65, 0.95))) for _ in range(n)]


def velocidade_agente(env):
    try:
        return float(env.agent.speed)
    except (AttributeError, TypeError, ValueError):
        return float("nan")


def carregar_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("Execute pip install ultralytics") from e
    pesos = ROOT / "yolov8n.pt"
    return YOLO(str(pesos))


def detectar_yolo(modelo, frame, imgsz=640, conf=0.25, memoria=None):
    resultados = modelo.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
    det = resultados[0]
    saida = frame.copy()
    contagem = Counter()
    if memoria is not None:
        memoria.clear()
    if det.boxes is not None:
        boxes = det.boxes.xyxy.cpu().numpy()
        cls = det.boxes.cls.cpu().numpy().astype(int)
        probs = det.boxes.conf.cpu().numpy()
        for coords, id_, p in zip(boxes, cls, probs):
            nome = str(det.names[int(id_)])
            if nome not in CLASSES_TRAFEGO:
                continue
            contagem[nome] += 1
            x1, y1, x2, y2 = map(int, coords)
            if memoria is not None:
                memoria.append((x1, y1, x2, y2, nome, float(p)))
            cv2.rectangle(saida, (x1, y1), (x2, y2), (30, 220, 80), 2)
            cv2.putText(saida, f"{nome} {float(p):.2f}", (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 220, 80), 2)
    return saida, dict(contagem)


def executar_experimento(steps, render=True, yolo=False, skip=1, imgsz=640,
                         seed=42, tamanho=(640, 360), save_video=True,
                         prefix="exp", modelo=None):
    """Executa `steps` steps e retorna linhas + resumo."""
    from time import perf_counter
    if yolo and modelo is None:
        modelo = carregar_yolo()
    if not yolo:
        modelo = None
    if modelo is not None:
        dummy = np.zeros((tamanho[1], tamanho[0], 3), np.uint8)
        for _ in range(2):
            modelo.predict(source=dummy, imgsz=imgsz, verbose=False)
    env = abrir_metadrive(render, tamanho[0], tamanho[1], seed)
    writer = None
    if save_video:
        out = OUT / f"feed_{prefix}_{steps}.mp4"
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 12, tamanho)
        if not writer.isOpened():
            raise RuntimeError("Nao foi possivel criar MP4.")
    obs = None
    episodio = 0
    ultima_contagem = {}
    ultima_boxes = []
    linhas = []
    acoes = sequencia_acoes(steps, seed)
    try:
        obs, info = env.reset(seed=seed)
        for idx, acao in enumerate(acoes):
            t0 = perf_counter()
            obs, recompensa, done, trunc, info = env.step(acao)
            sim_ms = (perf_counter() - t0) * 1000
            frame = obter_frame_bgr(obs)
            yolo_ms = 0.0
            anotado = frame.copy()
            atualizou = 0
            contagem = {}
            if modelo is not None:
                if idx % max(1, int(skip)) == 0:
                    ty = perf_counter()
                    anotado, contagem = detectar_yolo(
                        modelo, anotado, imgsz=imgsz, memoria=ultima_boxes)
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except ImportError:
                        pass
                    yolo_ms = (perf_counter() - ty) * 1000
                    ultima_contagem = contagem
                    atualizou = 1
                elif ultima_boxes:
                    for x1, y1, x2, y2, nome, p in ultima_boxes:
                        cv2.rectangle(anotado, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(anotado, f"{nome} {p:.2f} (antigo)",
                                    (x1, max(20, y1 - 7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)
                    cv2.putText(anotado, "YOLO: boxes anteriores (skip)",
                                (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 200, 255), 2)
                    contagem = ultima_contagem
            total_ms = (perf_counter() - t0) * 1000
            vel = velocidade_agente(env)
            registro = {
                "step": idx,
                "episodio": episodio,
                "velocidade_env": vel,
                "yolo_ms": round(yolo_ms, 4),
                "sim_ms": round(sim_ms, 4),
                "total_ms": round(total_ms, 4),
                "fps_instantaneo": round(1000.0 / max(total_ms, 1e-8), 3),
                "deteccao_atualizada": int(atualizou),
            }
            for nome in CLASSES_TRAFEGO:
                registro[nome] = int(contagem.get(nome, 0)) if modelo is not None else 0
            linhas.append(registro)
            if writer:
                cv2.putText(anotado,
                            f"Step {idx + 1}/{steps} | {1000.0 / max(total_ms, 1e-8):.1f} FPS",
                            (8, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2)
                writer.write(anotado)
            if idx in (0, steps // 2, steps - 1):
                salvar(f"{prefix}_step_{idx:04d}.png", anotado)
            if (idx + 1) % 100 == 0:
                print(f"[{prefix}] Step {idx + 1}/{steps} | "
                      f"FPS {1000.0 / max(total_ms, 1e-8):.1f} | "
                      f"objs {sum(contagem.values())}")
            if done or trunc:
                episodio += 1
                if idx < steps - 1:
                    obs, info = env.reset(seed=seed + episodio)
    finally:
        if writer is not None:
            writer.release()
        env.close()
    tempo = sum(x["total_ms"] for x in linhas) / 1000.0
    resumo = {
        "prefix": prefix,
        "steps": len(linhas),
        "yolo": bool(yolo),
        "skip": int(skip),
        "imgsz": int(imgsz),
        "fps_efetivo": len(linhas) / max(tempo, 1e-9),
        "latencia_media_ms": float(np.mean([x["total_ms"] for x in linhas])),
        "p95_ms": float(np.percentile([x["total_ms"] for x in linhas], 95)),
        "updates_yolo": sum(x["deteccao_atualizada"] for x in linhas),
        "objetos_totais": int(sum(
            x[n] for x in linhas for n in CLASSES_TRAFEGO)),
    }
    csvp = salvar_csv(f"metricas_{steps}_{prefix}.csv", linhas)
    (OUT / f"resumo_{steps}_{prefix}.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RESUMO [{prefix}]:", resumo)
    return linhas, resumo, csvp


def plotar_serie_temporal(linhas, prefix="yolo"):
    if plt is None:
        print("matplotlib indisponivel; pulando grafico.")
        return None
    steps = [r["step"] for r in linhas]
    categorias = [c for c in CLASSES_TRAFEGO if any(r[c] > 0 for r in linhas)]
    fig, ax = plt.subplots(figsize=(12, 6))
    for cat in categorias:
        ax.plot(steps, [r[cat] for r in linhas], label=cat, linewidth=1.2)
    ax.plot(steps, [sum(r[c] for c in CLASSES_TRAFEGO) for r in linhas],
            label="total", linewidth=2, color="black")
    ax.set_xlabel("Step")
    ax.set_ylabel("Objetos detectados")
    ax.set_title("Serie temporal de deteccoes YOLOv8n (MetaDrive)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    caminho = OUT / f"serie_temporal_deteccoes_{prefix}.png"
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print("Grafico salvo:", caminho)
    return caminho


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500,
                        help="Steps do experimento principal (padrao: 500)")
    parser.add_argument("--render", action="store_true", default=True,
                        help="Renderizacao do MetaDrive (padrao: True)")
    parser.add_argument("--bench-steps", type=int, default=100,
                        help="Steps dos benchmarks de otimizacao")
    parser.add_argument("--no-bench", action="store_true",
                        help="Pula benchmarks de skip/resolucao")
    args = parser.parse_args()

    # Pre-carrega YOLO antes do MetaDrive (evita conflito de DLLs no Windows).
    modelo_yolo = carregar_yolo()

    # 1. Baseline sem YOLO (FPS de referencia).
    linhas_base, resumo_base, _ = executar_experimento(
        steps=args.steps, render=args.render, yolo=False, skip=1, imgsz=640,
        prefix="baseline_sem_yolo")

    # 2. YOLO ativo (feed com bounding boxes).
    linhas_yolo, resumo_yolo, _ = executar_experimento(
        steps=args.steps, render=args.render, yolo=True, skip=1, imgsz=640,
        prefix="yolo_ativo", modelo=modelo_yolo)

    plotar_serie_temporal(linhas_yolo, prefix="yolo_ativo")

    # 3. Benchmarks de otimizacao (frame skip e resolucao reduzida).
    bench = []
    if not args.no_bench:
        for skip, imgsz, nome in [(2, 640, "skip2_img640"),
                                  (1, 320, "skip1_img320"),
                                  (2, 320, "skip2_img320")]:
            _, r, _ = executar_experimento(
                steps=args.bench_steps, render=args.render, yolo=True,
                skip=skip, imgsz=imgsz, prefix=nome, modelo=modelo_yolo)
            bench.append(r)

    # Degradacao de FPS.
    fps_base = resumo_base["fps_efetivo"]
    fps_yolo = resumo_yolo["fps_efetivo"]
    degradacao = (1.0 - fps_yolo / max(fps_base, 1e-9)) * 100.0
    print(f"FPS baseline: {fps_base:.2f} | FPS YOLO: {fps_yolo:.2f} | "
          f"Degradacao: {degradacao:.1f}%")

    relatorio = f"""# EX4 - Item B: YOLOv8n no loop do MetaDrive

## Configuracao
- MetaDriveEnv com use_render={args.render}, image_observation=True.
- Camera frontal 640x360; inferencia YOLOv8n (COCO) a cada frame.
- Classes monitoradas: {sorted(CLASSES_TRAFEGO)}.

## FPS
- Baseline sem YOLO: {fps_base:.2f} FPS ({resumo_base['latencia_media_ms']:.1f} ms/frame)
- YOLO ativo (skip=1, imgsz=640): {fps_yolo:.2f} FPS ({resumo_yolo['latencia_media_ms']:.1f} ms/frame)
- Degradacao: {degradacao:.1f}%

## Deteccoes (500 steps YOLO)
- Objetos totais: {resumo_yolo['objetos_totais']}
- Updates YOLO: {resumo_yolo['updates_yolo']}/{resumo_yolo['steps']}
- Serie temporal: serie_temporal_deteccoes_yolo_ativo.png

## Benchmarks de otimizacao ({args.bench_steps} steps)
"""
    for r in bench:
        relatorio += (f"- {r['prefix']}: {r['fps_efetivo']:.2f} FPS "
                      f"(skip={r['skip']}, imgsz={r['imgsz']}, "
                      f"lat={r['latencia_media_ms']:.1f} ms)\n")
    relatorio += """
## Discussao
A adicao de um detector profundo no loop de simulacao aumenta a latencia
por frame e reduz o FPS efetivo. O YOLOv8n em CPU domina o custo quando
executado em todos os frames e em resolucao 640. Estrategias que minimizam
a queda de FPS: (a) frame skip (inferir a cada N frames e reutilizar as
ultimas caixas nos frames intermediarios), (b) reducao da resolucao de
inferencia (imgsz=320), (c) combinacao de skip + resolucao reduzida,
(d) execucao em GPU com batch/async e (e) detector mais leve ou pruning.
O frame skip reduz proporcionalmente o custo do YOLO, enquanto a resolucao
reduzida diminui o custo por inferencia; a combinacao oferece o melhor
compromisso, ao custo de menor fidelidade temporal/espacial das deteccoes.
"""
    escrever_texto("relatorio_ex4_item_b.md", relatorio)
    print("EX4 item B concluido em:", OUT)


if __name__ == "__main__":
    main()
    print("EX4 item B concluido em:", OUT)
