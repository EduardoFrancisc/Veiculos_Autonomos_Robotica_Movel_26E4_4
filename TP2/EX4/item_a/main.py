from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXEMPLOS = ROOT / "exemplos_tp2" / "MetaDrive_YOLO_40_Independentes"
DADOS = EXEMPLOS / "dados"
TP1_DADOS = ROOT / "exemplos_tp1" / "Deteccao_Faixas_Independentes" / "dados"
OUT = Path(__file__).resolve().parent / "resultados" / "main"
OUT.mkdir(parents=True, exist_ok=True)


def salvar(nome, imagem):
    caminho = OUT / nome
    if imagem is None:
        raise ValueError(f"Imagem invalida para salvar: {nome}")
    ok, dados = cv2.imencode(caminho.suffix if caminho.suffix else ".png", imagem)
    if not ok:
        raise OSError(f"Falha ao gravar {caminho}")
    dados.tofile(str(caminho))
    print("Imagem salva:", caminho)
    return caminho


def salvar_csv(nome, registros):
    caminho = OUT / nome
    if not registros:
        raise ValueError("Nao ha registros a salvar; o teste falhou antes do primeiro step.")
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
    """Instancia o MetaDrive REAL com camera frontal RGB.

    use_render e image_observation seguem o enunciado (observacao visual).
    """
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        from metadrive.component.sensors.rgb_camera import RGBCamera
    except ImportError as e:
        raise RuntimeError(
            "MetaDrive nao instalado. pip install metadrive-simulator"
        ) from e
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
    """Recupera o frame RGB da observacao e converte para BGR uint8."""
    if not isinstance(obs, dict) or "image" not in obs:
        raise RuntimeError(f"Observacao inesperada: {type(obs)}")
    frame = np.asarray(obs["image"])
    if frame.ndim == 4:
        frame = frame[..., -1]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise RuntimeError(f"Formato inesperado do frame: {frame.shape}")
    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * (255.0 if float(np.max(frame)) <= 1.01 else 1.0),
                        0, 255).astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def sequencia_acoes(n, semente=42):
    """Politica aleatoria ilustrativa (mesma semente para reprodutibilidade)."""
    rng = np.random.default_rng(semente)
    return [(float(rng.uniform(-0.12, 0.12)),
             float(rng.uniform(0.65, 0.95))) for _ in range(n)]


def velocidade_agente(env):
    try:
        return float(env.agent.speed)
    except (AttributeError, TypeError, ValueError):
        return float("nan")


def detectar_faixas(frame):
    """Detector de faixas do TP1 (adaptado para resolucoes variaveis).

    Pipeline: HLS + Sobel -> ROI -> Canny -> HoughLinesP -> separacao
    esquerda/direita -> ajuste quadratico -> desvio lateral e curvatura.
    Na ausencia de duas faixas validas devolve None (sem inventar medidas).
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    l, s = hsv[:, :, 1], hsv[:, :, 2]
    branco = (l >= 165) & (s <= 125)
    amarelo = (s >= 90) & (l >= 65)
    y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(y, (5, 5), 0)
    gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    lim = np.percentile(gx, 85)
    grad = (gx > max(16.0, lim)).astype(np.uint8) * 255
    mask = ((branco | amarelo).astype(np.uint8) * 255)
    mask = cv2.bitwise_or(mask, grad)
    roi = np.zeros_like(mask)
    pol = np.array([[(0, h - 1), (int(0.43 * w), int(0.60 * h)),
                     (int(0.57 * w), int(0.60 * h)), (w - 1, h - 1)]], np.int32)
    cv2.fillPoly(roi, pol, 255)
    mask = cv2.bitwise_and(mask, roi)
    bordas = cv2.Canny(mask, 60, 150)
    linhas = cv2.HoughLinesP(bordas, 1, np.pi / 180, 35,
                             minLineLength=24, maxLineGap=30)
    anotado = frame.copy()
    cv2.polylines(anotado, pol, True, (100, 100, 100), 1)
    pontos = {"esquerda": [], "direita": []}
    if linhas is not None:
        for x1, y1, x2, y2 in linhas.reshape(-1, 4):
            dy = y2 - y1
            dx = x2 - x1
            if abs(dy) < 12:
                continue
            m = dx / float(dy)
            if abs(m) < 0.10 or abs(m) > 2.0:
                continue
            xm = (x1 + x2) / 2
            lado = "esquerda" if (xm < w / 2 and m < 0) else (
                "direita" if (xm > w / 2 and m > 0) else None)
            if lado:
                pontos[lado].extend([(y1, x1), (y2, x2)])
                cv2.line(anotado, (x1, y1), (x2, y2), (20, 210, 255), 2)
    ajustes = {}
    for lado, dados in pontos.items():
        if len(dados) >= 6:
            ar = np.array(dados, dtype=float)
            ys, xs = ar[:, 0], ar[:, 1]
            if np.ptp(ys) >= 20:
                ajuste = np.polyfit(ys, xs, 2)
                ajustes[lado] = ajuste
                yy = np.arange(int(0.61 * h), h, 2)
                xx = np.clip(np.polyval(ajuste, yy), 0, w - 1).astype(int)
                pts = np.column_stack([xx, yy]).astype(np.int32)
                cv2.polylines(anotado, [pts], False, (0, 255, 30), 3)
    desvio_px = None
    curvatura_px = None
    if len(ajustes) == 2:
        c1 = np.polyval(ajustes["esquerda"], h - 1)
        c2 = np.polyval(ajustes["direita"], h - 1)
        if (c2 - c1) > 0.15 * w:
            centro = (c1 + c2) / 2
            desvio_px = float(w / 2 - centro)
            y0 = float(0.85 * h)
            radios = []
            for fit in ajustes.values():
                a, b, _ = fit
                if abs(a) > 1e-8:
                    radios.append((1 + (2 * a * y0 + b) ** 2) ** 1.5 /
                                  (2 * abs(a)))
            curvatura_px = float(np.mean(radios)) if radios else None
            cv2.line(anotado, (int(centro), h - 1), (int(w / 2), h - 1),
                     (255, 255, 255), 3)
    status = (f"Desvio: {desvio_px:+.1f} px"
              if desvio_px is not None else "Faixas insuficientes")
    cv2.rectangle(anotado, (0, 0), (w, 36), (15, 15, 15), -1)
    cv2.putText(anotado, status, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (230, 230, 230), 2)
    return {"imagem": anotado, "mascara": mask, "bordas": bordas,
            "desvio_px": desvio_px, "curvatura_raio_px": curvatura_px,
            "status_valido": desvio_px is not None}


def executar_200_steps(steps=200, render=True, seed=42, tamanho=(640, 360),
                      save_video=True):
    """Executa `steps` steps com politica aleatoria e detector de faixas."""
    acoes = sequencia_acoes(steps, seed)
    env = abrir_metadrive(render, tamanho[0], tamanho[1], seed)
    writer = None
    if save_video:
        out = OUT / f"feed_metadrive_faixas_{steps}.mp4"
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 12, tamanho)
        if not writer.isOpened():
            raise RuntimeError("Nao foi possivel criar o MP4 de saida.")
    obs = None
    episodio = 0
    linhas = []
    try:
        obs, info = env.reset(seed=seed)
        for idx, acao in enumerate(acoes):
            t0 = perf_counter()
            obs, recompensa, done, trunc, info = env.step(acao)
            sim_ms = (perf_counter() - t0) * 1000
            frame = obter_frame_bgr(obs)
            ta = perf_counter()
            res = detectar_faixas(frame)
            faixa_ms = (perf_counter() - ta) * 1000
            anotado = res["imagem"]
            desvio_px = res["desvio_px"]
            curvatura_px = res["curvatura_raio_px"]
            total_ms = (perf_counter() - t0) * 1000
            vel = velocidade_agente(env)
            registro = {
                "step": idx,
                "episodio": episodio,
                "velocidade_env": vel,
                "desvio_px": desvio_px,
                "curvatura_raio_px": curvatura_px,
                "status_valido": bool(res["status_valido"]),
                "sim_ms": round(sim_ms, 4),
                "faixas_ms": round(faixa_ms, 4),
                "total_ms": round(total_ms, 4),
                "fps_instantaneo": round(1000.0 / max(total_ms, 1e-8), 3),
            }
            linhas.append(registro)
            if writer:
                cv2.putText(anotado,
                            f"Step {idx + 1}/{steps} | desvio={desvio_px if desvio_px is not None else 'NA'} px",
                            (8, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2)
                writer.write(anotado)
            if idx in (0, steps // 2, steps - 1):
                salvar(f"step_{idx:04d}_overlay.png", anotado)
            if (idx + 1) % 50 == 0:
                print(f"Step {idx + 1}/{steps} | desvio={desvio_px} | "
                      f"raio={curvatura_px} | vel={vel:.3f}")
            if done or trunc:
                episodio += 1
                if idx < steps - 1:
                    obs, info = env.reset(seed=seed + episodio)
    finally:
        if writer is not None:
            writer.release()
        env.close()
    csvp = salvar_csv(f"metricas_{steps}_steps.csv", linhas)
    resumo = {
        "steps": len(linhas),
        "faixas_validas": sum(1 for x in linhas if x["status_valido"]),
        "velocidade_media": float(np.nanmean([x["velocidade_env"] for x in linhas])),
        "desvio_medio_px": float(np.nanmean(
            [x["desvio_px"] for x in linhas if x["desvio_px"] is not None])),
        "metadados": "Medidas reais do MetaDrive; desvio/curvatura em pixels do plano da imagem (sem calibracao metrica).",
    }
    (OUT / f"resumo_{steps}.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESUMO:", resumo)
    return linhas, resumo, csvp


def comparar_real_simulado():
    """Aplica o detector em imagem real (KITTI/TP1) e em frame simulado."""
    # Imagem real (KITTI) disponivel no pacote MetaDrive.
    real = None
    caminho_real = DADOS / "frontal_kitti.png"
    if caminho_real.exists():
        real = cv2.imdecode(np.fromfile(str(caminho_real), dtype=np.uint8),
                            cv2.IMREAD_COLOR)
    if real is None:
        # Fallback: imagem sintetica do TP1.
        caminho_tp1 = TP1_DADOS / "estrada_sintetica.png"
        if caminho_tp1.exists():
            real = cv2.imdecode(np.fromfile(str(caminho_tp1), dtype=np.uint8),
                                cv2.IMREAD_COLOR)
    if real is None:
        raise FileNotFoundError("Nenhuma imagem real/sintetica encontrada para comparacao.")
    real = cv2.resize(real, (640, 360))

    # Frame simulado do MetaDrive.
    env = abrir_metadrive(True, 640, 360, 42)
    try:
        obs, info = env.reset(seed=42)
        sim = obter_frame_bgr(obs)
    finally:
        env.close()

    r_real = detectar_faixas(real)
    r_sim = detectar_faixas(sim)
    painel = np.hstack([r_real["imagem"],
                        cv2.resize(r_sim["imagem"], (real.shape[1], real.shape[0]))])
    # Rotulos
    cv2.rectangle(painel, (0, 0), (640, 30), (0, 0, 0), -1)
    cv2.rectangle(painel, (640, 0), (1280, 30), (0, 0, 0), -1)
    cv2.putText(painel, "REAL (TP1/KITTI)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(painel, "SIMULADO (MetaDrive)", (650, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    salvar("comparacao_real_vs_simulado.png", painel)

    # Comparacao adicional com imagem sintetica do TP1 (mesmo detector).
    sintetica = None
    caminho_sint = TP1_DADOS / "estrada_sintetica.png"
    if caminho_sint.exists():
        sintetica = cv2.imdecode(np.fromfile(str(caminho_sint), dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
    if sintetica is not None:
        sintetica = cv2.resize(sintetica, (640, 360))
        r_sint = detectar_faixas(sintetica)
        painel2 = np.hstack([r_sint["imagem"],
                             cv2.resize(r_sim["imagem"], (640, 360))])
        cv2.rectangle(painel2, (0, 0), (640, 30), (0, 0, 0), -1)
        cv2.rectangle(painel2, (640, 0), (1280, 30), (0, 0, 0), -1)
        cv2.putText(painel2, "SINTETICA (TP1)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(painel2, "SIMULADO (MetaDrive)", (650, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        salvar("comparacao_sintetica_vs_simulado.png", painel2)

    print("Real valido:", r_real["status_valido"],
          "| Simulado valido:", r_sim["status_valido"])
    return r_real, r_sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200,
                        help="Numero de steps (padrao: 200)")
    parser.add_argument("--render", action="store_true", default=True,
                        help="Usar renderizacao do MetaDrive (padrao: True)")
    parser.add_argument("--no-video", action="store_true",
                        help="Nao gravar o feed MP4")
    args = parser.parse_args()

    linhas, resumo, csvp = executar_200_steps(
        steps=args.steps, render=args.render, save_video=not args.no_video)

    r_real, r_sim = comparar_real_simulado()

    relatorio = f"""# EX4 - Item A: MetaDrive em modo observacao

## Configuracao
- MetaDriveEnv com use_render={args.render}, image_observation=True.
- Camera frontal RGB (640x360), frames capturados a cada step.
- Detector de faixas do TP1 aplicado sobre os frames simulados.
- Desvio lateral exibido em tempo real no feed sobreposto.

## Resultados
- Steps executados: {len(linhas)}
- Faixas validas: {resumo['faixas_validas']}/{len(linhas)}
- Velocidade media do agente: {resumo['velocidade_media']:.3f}
- Desvio medio (px): {resumo['desvio_medio_px']:.2f}
- CSV: {csvp.name} ({len(linhas)} registros)
- Feed: feed_metadrive_faixas_{args.steps}.mp4
- Comparacao visual: comparacao_real_vs_simulado.png, comparacao_sintetica_vs_simulado.png
- Validade detector -> real (KITTI): {r_real['status_valido']}; simulado (MetaDrive): {r_sim['status_valido']}

## Observacao
Desvio/curvatura estao no plano da imagem (pixels) por ausencia de
calibracao metrica da camera MetaDrive; valores em metros exigiriam
homografia calibrada para a cena simulada.
"""
    escrever_texto("relatorio_ex4_item_a.md", relatorio)
    print("EX4 item A concluido em:", OUT)


if __name__ == "__main__":
    main()
    print("EX4 item A concluido em:", OUT)
