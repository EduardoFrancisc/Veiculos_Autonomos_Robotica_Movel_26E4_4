import sys
from pathlib import Path
import numpy as np

PASTA_APOIO = Path(__file__).resolve().parents[2] / "scripts_aula" / "OpenCV_Sistemas_Autonomos_36_Exemplos" / "exemplos"
sys.path.insert(0, str(PASTA_APOIO))

from apoio import iniciar, rodar_stream, salvar_json

a, _ = iniciar(__file__)
out = Path(__file__).resolve().parent / "resultados" / Path(__file__).stem
out.mkdir(parents=True, exist_ok=True)

# 1) Leitura do stream + pipeline (resize, HSV, canal V, bilateral) por frame.
# 2) Mede tempo por frame (ms) e FPS; grava feed_processado.mp4 e CSV/JSON.
linhas, resumo = rodar_stream(a, out, gravar=True, verbose=True)

# Estatisticas por etapa para a discussao de viabilidade.
etapas = ["resize_ms", "hsv_ms", "extracao_v_ms", "bilateral_ms", "proc_ms", "loop_ms"]
medias = {k: float(np.mean([float(r[k]) for r in linhas])) for k in etapas}
fps_proc = resumo["fps_processamento"]
fps_loop = resumo["fps_loop"]
prazo_ms = resumo["prazo_ms"]
acima = resumo["percentual_acima_prazo"]

print("\n=== MEDIAS POR ETAPA (ms) ===")
for k in etapas:
    print(f"{k:16s}: {medias[k]:8.3f}")
print(f"FPS processamento : {fps_proc:8.2f}")
print(f"FPS loop          : {fps_loop:8.2f}")
print(f"Prazo da meta     : {prazo_ms:8.3f} ms ({a.fps_alvo:g} FPS)")
print(f"Frames acima prazo: {acima:6.2f}%")

# 3) Discussao de viabilidade em SBC (Raspberry Pi 4, 4 GB, Cortex-A72).
# Estimativa conservadora: Pi 4 costuma ser ~4-6x mais lento que um PC x86
# neste tipo de pipeline OpenCV single-thread; bilateral e a etapa dominante.
fator_pi = 5.0
loop_pi_ms = medias["loop_ms"] * fator_pi
fps_pi = 1000.0 / loop_pi_ms if loop_pi_ms > 0 else 0.0
viavel = loop_pi_ms <= prazo_ms

discussao = f"""
DISCUSSAO DE VIABILIDADE - Raspberry Pi 4 (4 GB, Cortex-A72)
Medicao neste PC: loop medio = {medias['loop_ms']:.2f} ms/frame -> {fps_loop:.1f} FPS.
Etapa dominante: bilateral ({medias['bilateral_ms']:.2f} ms), seguida de resize ({medias['resize_ms']:.2f} ms).
Estimativa conservadora para Pi 4 (fator {fator_pi:.1f}x): {loop_pi_ms:.2f} ms/frame -> {fps_pi:.1f} FPS.
Meta de tempo real adotada: {prazo_ms:.2f} ms/frame ({a.fps_alvo:g} FPS).
Conclusao: {'VIÁVEL' if viavel else 'NÃO VIÁVEL'} no Pi 4 para {a.fps_alvo:g} FPS nas condicoes medidas.
Observacoes:
 - Medicao em PC nao equivale a Pi; executar na placa para confirmar (ex. 35_raspberry_pi.py).
 - Se nao viavel: reduzir resolucao de entrada, diminuir d/sigma do bilateral, usar cv2.setNumThreads,
   desabilitar gravacao (gravar=False) e/ou baixar a meta de FPS.
 - Gravacao de video e impressao por frame (verbose) consomem tempo e nao devem contar para FPS de capacidade.
"""
print(discussao)

salvar_json(out / "viabilidade.json", {
    "medias_ms": medias,
    "fps_processamento": fps_proc,
    "fps_loop": fps_loop,
    "prazo_ms": prazo_ms,
    "percentual_acima_prazo": acima,
    "fator_pi_estimado": fator_pi,
    "loop_pi_ms_estimado": loop_pi_ms,
    "fps_pi_estimado": fps_pi,
    "viavel_pi_estimado": viavel,
})
(out / "viabilidade.md").write_text(discussao.strip() + "\n", encoding="utf-8")

print("Item B concluido em:", out)