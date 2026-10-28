import sys
from pathlib import Path

PACOTE_CAN = Path(__file__).resolve().parents[3] / "scripts_aula" / "Protocolos_Veiculares_Exemplos_Separados" / "05_can_virtual"
sys.path.insert(0, str(PACOTE_CAN))

from apoio import iniciar
import can_laboratorio as canlab

COMENTARIO_CAN = """
CAN 2.0A (Classical CAN): ID padrao de 11 bits, payload maximo de 8 bytes,
taxa de dados ate ~1 Mbit/s (fase unica). Adequado a sinais pequenos e
redes curtas/medias, mas limita throughput e exige fragmentacao para dados maiores.

CAN FD: IDs de 11 ou 29 bits, payload de ate 64 bytes e fase de dados com
bitrate maior (BRS), tipicamente 2-5 Mbit/s (dependente de controlador,
transceptor, comprimento e projeto da rede). Reduz fragmentacao e latencia
para payloads maiores, mantendo compatibilidade no arbitramento.
"""


def main():
    a, _ = iniciar(__file__, ciclos=50)
    out = Path(__file__).resolve().parent / "resultados" / Path(__file__).stem
    out.mkdir(parents=True, exist_ok=True)

    # 1) Publicador e assinante em processos distintos (interface='virtual'),
    #    IDs 0x100 (velocidade), 0x101 (RPM), 0x102 (temperatura) a 10 Hz.
    resumo = canlab.can_processos(out, ciclos=a.ciclos, hz=a.hz)

    # 2/3/4) CSV (mensagens_can.csv), plot (series_can.png) e resumo ja gerados.
    print("Resumo CAN:", resumo)
    print(COMENTARIO_CAN)

    print("EX2 item A concluido em:", out)


if __name__ == "__main__":
    main()