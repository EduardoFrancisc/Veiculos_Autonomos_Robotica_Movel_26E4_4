"""
EX1 - Item B: segmentacao morfologica de marcacoes de pista em imagens BEV (TP2).

Baseado nos exemplos independentes de "exemplos_tp2/
Aula_Homografia_Morfologia_40_Independentes" (threshold adaptativo, opening,
closing, kernels e componentes conectados).
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXEMPLOS = ROOT / "exemplos_tp2" / "Aula_Homografia_Morfologia_40_Independentes"
DADOS = EXEMPLOS / "dados"
OUT = Path(__file__).resolve().parent / "resultados" / "main"
OUT.mkdir(parents=True, exist_ok=True)


def ler(nome):
    caminho = DADOS / nome
    imagem = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise FileNotFoundError(f"Imagem ausente ou invalida: {caminho}")
    return imagem


def salvar(nome, imagem):
    arquivo = OUT / nome
    ok, dados = cv2.imencode(arquivo.suffix if arquivo.suffix else ".png", imagem)
    if not ok:
        raise OSError(f"Falha ao salvar imagem: {arquivo}")
    dados.tofile(str(arquivo))
    print("Imagem salva:", arquivo)
    return arquivo


def painel(pares, destino="resultado.png", largura=560, altura=310):
    quadros = []
    for legenda, imagem in pares:
        if len(imagem.shape) == 2:
            imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (largura, 43), (22, 31, 46), -1)
        cv2.putText(img, legenda[:43], (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.66, (255, 255, 255), 2, cv2.LINE_AA)
        quadros.append(img)
    ncol = min(3, len(quadros))
    while len(quadros) % ncol:
        quadros.append(np.full_like(quadros[0], (235, 235, 235)))
    linhas = [np.hstack(quadros[i:i + ncol]) for i in range(0, len(quadros), ncol)]
    resultado = np.vstack(linhas)
    salvar(destino, resultado)
    return resultado


def binarizar(img):
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.adaptiveThreshold(cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, -12)


def morfologia(mask, forma=cv2.MORPH_RECT, tamanho=(5, 7)):
    kernel = cv2.getStructuringElement(forma, tamanho)
    abertura = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # erosao -> dilatacao
    fechamento = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel)  # dilatacao -> erosao
    return abertura, fechamento, kernel


def selecionar_componentes(img, mascara, area_min=100, razao_min=1.5):
    n, rotulos, stats, centros = cv2.connectedComponentsWithStats(mascara, 8)
    saida = img.copy()
    aceitos = []
    for i in range(1, n):  # 0 e o fundo
        x, y, w, h, area = map(int, stats[i])
        razao = h / max(w, 1)
        if area >= area_min and razao >= razao_min:
            aceitos.append((x, y, w, h, area, razao))
            cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if aceitos:
        print("Marcacoes aprovadas:", len(aceitos))
        print("Largura media (px):", round(float(np.mean([v[2] for v in aceitos])), 2))
        print("Altura media (px):", round(float(np.mean([v[3] for v in aceitos])), 2))
        print("Area media (px^2):", round(float(np.mean([v[4] for v in aceitos])), 2))
    else:
        print("Nenhuma marcacao atende aos filtros de area e aspect ratio.")
    return saida, aceitos, stats


def main():
    bev = ler("bev_marcacoes.png")
    mask = binarizar(bev)
    salvar("threshold_adaptativo.png", mask)

    formas = [
        ("Retangular", cv2.MORPH_RECT),
        ("Eliptico", cv2.MORPH_ELLIPSE),
        ("Cruz", cv2.MORPH_CROSS),
    ]

    resultados = [("BEV", bev), ("Threshold adaptativo", mask)]
    componentes_final = None
    aceitos_final = []

    for nome, forma in formas:
        abertura, fechamento, kernel = morfologia(mask, forma, (5, 7))
        print(f"\nKernel {nome} (5x7):\n{kernel}")
        salvar(f"opening_{nome.lower()}.png", abertura)
        salvar(f"closing_{nome.lower()}.png", fechamento)
        resultados.append((f"Opening - {nome}", abertura))
        resultados.append((f"Closing - {nome}", fechamento))

        if nome == "Retangular":
            componentes_final, aceitos_final, stats = selecionar_componentes(
                bev, fechamento, area_min=100, razao_min=1.5)
            salvar("componentes_filtrados.png", componentes_final)
            resultados.append(("Componentes filtrados", componentes_final))

    painel(resultados, "painel_morfologia.png")

    print("\nQuantidade de marcacoes detectadas:", len(aceitos_final))
    if aceitos_final:
        print("Dimensoes medias das marcacoes:")
        print("  largura media (px):", round(float(np.mean([v[2] for v in aceitos_final])), 2))
        print("  altura media (px):", round(float(np.mean([v[3] for v in aceitos_final])), 2))
        print("  area media (px^2):", round(float(np.mean([v[4] for v in aceitos_final])), 2))
    else:
        print("  (nenhuma marcacao detectada)")

    # Comentario tecnico: morfologia em BEV para veiculos autonomos.
    # Em pipelines de percepcao, a imagem bird's-eye view (BEV) concentra as
    # marcacoes de pista em regioes alongadas. O threshold adaptativo lida com
    # iluminacao nao uniforme; opening remove ruido/speckles e closing fecha
    # pequenas lacunas. Kernels retangulares preservam tracos longos, elipticos
    # suavizam contornos e em cruz favorecem conectividade ortogonal. Apos a
    # morfologia, connectedComponentsWithStats permite filtrar candidatos por
    # area e aspect ratio, reduzindo falsos positivos antes da medicao das
    # marcacoes e do planejamento de trajetoria.


if __name__ == "__main__":
    main()
    print("EX1 item B concluido em:", OUT)
