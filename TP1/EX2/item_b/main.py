import sys
from pathlib import Path

PACOTE_REDE = Path(__file__).resolve().parents[3] / "scripts_aula" / "Protocolos_Veiculares_Exemplos_Separados" / "23_camera_udp_curto"
sys.path.insert(0, str(PACOTE_REDE))

from apoio import iniciar, parametros_stream
from rede_laboratorio import comparar

COMENTARIO = """
Conclusao (streaming de camera em veiculo autônomo):
UDP descarta frames incompletos sem esperar recuperacao de bytes anteriores,
favorecendo atualidade (baixa latência/jitter) quando o consumidor tolera lacunas.
TCP entrega fluxo confiavel e ordenado, podendo reter dados novos durante a
recuperacao, aumentando latência/jitter em perdas. Para percepção de faixas/
obstáculos, priorizar atualidade -> UDP com IDs, remontagem limitada e supervisão.
Para gravação/transferência que exige todos os bytes -> TCP.
"""


def main():
    a, _ = iniciar(__file__, frames=320, ciclos=20)
    out = Path(__file__).resolve().parent / "resultados" / Path(__file__).stem
    out.mkdir(parents=True, exist_ok=True)

    config = parametros_stream(a)
    if config["frames"] < 300:
        config["frames"] = 320

    resultado = comparar(out, config, ordem=("udp", "tcp"))
    print("Comparacao concluida:", {p: {k: resultado[p][k] for k in
          ["media_latencia_ms", "jitter_std_ms", "perda_frames_pct"]} for p in resultado})
    print(COMENTARIO)
    print("EX2 item B concluido em:", out)


if __name__ == "__main__":
    main()