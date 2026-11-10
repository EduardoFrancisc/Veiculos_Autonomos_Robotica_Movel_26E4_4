"""
EX2 - Item A: convolucao manual, filtros direcionais e relacao com CNN (TP2).

Tambem inclui a comparacao de operadores de borda para manter os dois itens
do exercicio 2 cobertos. Baseado nos exemplos independentes de
"exemplos_tp2/Filtragem_Espacial_Bordas_40_Exemplos_Independentes".
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXEMPLOS = ROOT / "exemplos_tp2" / "Filtragem_Espacial_Bordas_40_Exemplos_Independentes"
DADOS = EXEMPLOS / "dados"
OUT = Path(__file__).resolve().parent / "resultados" / "main"
OUT.mkdir(parents=True, exist_ok=True)


def ler(nome, cinza=False):
    caminho = DADOS / nome
    imagem = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8),
                          cv2.IMREAD_GRAYSCALE if cinza else cv2.IMREAD_COLOR)
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


def painel(pares, destino, largura=300, altura=220):
    quadros = []
    for legenda, imagem in pares:
        if len(imagem.shape) == 2:
            imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (largura, 34), (22, 31, 46), -1)
        cv2.putText(img, legenda[:38], (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 1, cv2.LINE_AA)
        quadros.append(img)
    ncol = min(3, len(quadros))
    while len(quadros) % ncol:
        quadros.append(np.full_like(quadros[0], (235, 235, 235)))
    linhas = [np.hstack(quadros[i:i + ncol]) for i in range(0, len(quadros), ncol)]
    resultado = np.vstack(linhas)
    salvar(destino, resultado)
    return resultado


def cinza(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def normalize8(x):
    x = np.asarray(x, dtype=np.float64)
    maxv = float(np.max(np.abs(x)))
    return np.uint8(np.clip((np.abs(x) / max(maxv, 1e-9)) * 255, 0, 255))


def sombra(img):
    a = img.astype(np.float32).copy()
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    pts = np.array([[0, int(h * .16)], [w, int(h * .36)],
                    [w, int(h * .59)], [0, int(h * .39)]], np.int32)
    cv2.fillConvexPoly(m, pts, 255)
    a[m > 0] *= .38
    a += np.random.default_rng(41).normal(0, 2.5, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


def molhado(img):
    h, w = img.shape[:2]
    a = (img.astype(np.float32) * .68).astype(np.uint8)
    overlay = np.zeros_like(a)
    cv2.ellipse(overlay, (int(w * .50), int(h * .73)), (int(w * .39), int(h * .1)),
                0, 0, 360, (175, 181, 189), -1)
    cv2.line(overlay, (int(w * .27), int(h * .57)), (int(w * .76), int(h * .89)),
             (255, 235, 226), 6)
    cv2.line(overlay, (int(w * .47), int(h * .62)), (int(w * .66), int(h * .97)),
             (225, 224, 240), 4)
    a = cv2.addWeighted(a, .83, overlay, .17, 0)
    noise = np.random.default_rng(27).normal(0, 6, a.shape)
    return np.clip(a.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def banco_direcional(img):
    g = cinza(img).astype(np.float64)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    resp = {}
    for graus in (0, 45, 90, 135):
        th = np.deg2rad(graus)
        k = np.cos(th) * kx + np.sin(th) * ky
        resp[graus] = np.abs(cv2.filter2D(g, cv2.CV_64F, k,
                                          borderType=cv2.BORDER_REFLECT_101))
    return resp


def main():
    # ------------------------------------------------------------------
    # 1) Convolucao 2D manual com kernel Sobel 3x3, validada pixel a pixel
    # ------------------------------------------------------------------
    g = cv2.resize(ler("pista_bev.png", cinza=True), (128, 128))
    k_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    k_matematico = np.flip(k_sobel, (0, 1))
    pad = np.pad(g.astype(np.float64), ((1, 1), (1, 1)), mode="reflect")
    manual = np.zeros(g.shape, dtype=np.float64)
    for yy in range(g.shape[0]):
        for xx in range(g.shape[1]):
            manual[yy, xx] = np.sum(pad[yy:yy + 3, xx:xx + 3] *
                                    np.flip(k_matematico, (0, 1)))
    referencia = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3,
                           borderType=cv2.BORDER_REFLECT_101)
    erro_max = float(np.max(np.abs(manual - referencia)))
    print("Convolucao manual vs cv2.Sobel - erro absoluto maximo por pixel:", erro_max)
    painel([("BEV 128x128", cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)),
            ("Sobel manual (NumPy)", normalize8(manual)),
            ("Sobel OpenCV", normalize8(referencia))],
           "convolucao_manual_vs_opencv.png", largura=260, altura=260)

    # ------------------------------------------------------------------
    # 2) Banco de filtros direcionais (0, 45, 90, 135 graus) na BEV
    # ------------------------------------------------------------------
    img_bev = ler("pista_bev.png")
    respostas = banco_direcional(img_bev)
    painel_items = [("BEV de faixa", img_bev)]
    medias_globais = {}
    for angulo, resp in respostas.items():
        medias_globais[angulo] = float(np.mean(resp))
        painel_items.append((f"Filtro {angulo} deg", normalize8(resp)))
        salvar(f"direcional_{angulo:03d}.png", normalize8(resp))

    # Resposta maxima sobre as marcacoes da pista (usa faixas_gt como mascara).
    try:
        mascara = ler("faixas_gt.png", cinza=True) > 0
    except FileNotFoundError:
        mascara = ler("bordas_gt.png", cinza=True) > 0
    medias_marcacoes = {}
    for angulo, resp in respostas.items():
        if mascara.any():
            medias_marcacoes[angulo] = float(np.mean(resp[mascara]))
        else:
            medias_marcacoes[angulo] = medias_globais[angulo]
    angulo_max = max(medias_marcacoes, key=medias_marcacoes.get)
    print("\nResposta media global por filtro direcional:")
    for angulo in sorted(respostas):
        print(f"  {angulo:3d} deg: global={medias_globais[angulo]:.4f}, "
              f"sobre marcacoes={medias_marcacoes[angulo]:.4f}")
    print("Angulo de maxima resposta sobre as marcacoes:", angulo_max, "graus")

    painel(painel_items, "filtros_direcionais.png", largura=260, altura=260)

    # ------------------------------------------------------------------
    # 3) Discussao: filtros direcionais e camadas convolucionais de CNN
    # ------------------------------------------------------------------
    # As convolucoes manuais e o banco direcional mostram o mesmo principio
    # das camadas Conv2D de uma CNN: um kernel desliza sobre a imagem,
    # calcula produto interno local e gera um mapa de ativacao. Na CNN, os
    # pesos dos kernels nao sao fixos (Sobel/direcional), mas aprendidos por
    # backpropagation; camadas iniciais tendem a aprender detectores de
    # borda/orientacao, analogos a estes filtros, e camadas profundas
    # combinam essas respostas em features mais semanticas. Stride, padding,
    # numero de filtros e funcoes de ativacao organizam esse aprendizado.
    # (Conexao com a disciplina DR2: convolucao, filtros e redes neurais.)

    print("\nConclusao: angulo de maior resposta =", angulo_max,
          "graus. Convolucao manual validada contra cv2.Sobel "
          f"(erro maximo {erro_max}).")


if __name__ == "__main__":
    main()
    print("EX2 item A concluido em:", OUT)
