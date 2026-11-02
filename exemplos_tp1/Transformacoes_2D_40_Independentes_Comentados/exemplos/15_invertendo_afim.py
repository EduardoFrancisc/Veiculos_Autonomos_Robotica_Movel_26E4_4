"""EXEMPLO 15 — Inversao aproximada de uma transformacao

OBJETIVO E CONTEXTO:
O exemplo aplica uma rotação acompanhada de pequeno zoom e, em seguida, calcula a matriz inversa com cv2.invertAffineTransform para tentar recuperar a imagem original. A comparação permite observar que a geometria pode ser aproximadamente revertida, embora pixels perdidos, recortes e interpolações impeçam uma recuperação perfeita. Esse procedimento é importante em alinhamento, registro de imagens e projeção de resultados de volta à imagem de referência.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/15_invertendo_afim/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
invertAffineTransform calcula a matriz inversa; aplica-se ao resultado anterior.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
A inversa matematica da coordenada nao recupera informacao eliminada ou degradada pela interpolacao.

EXERCICIO PARA OS ALUNOS:
Analisar quais partes foram cortadas e nao podem ser recuperadas.

EXECUCAO:
python exemplos/15_invertendo_afim.py
Nao e necessario executar os demais exemplos antes deste.
As dependencias externas sao cv2 e numpy; o exemplo de CNN
admite TensorFlow opcionalmente. As imagens ficam em dados/."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

# Resolucao robusta de caminhos: funciona ao executar a partir de qualquer pasta.
BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / 'dados'
OUT = BASE / 'saidas'
OUT.mkdir(parents=True, exist_ok=True)

# Funcoes privativas deste arquivo. Nenhum modulo Python adicional do pacote e necessario.

def img(nome='frontal_kitti.png'):
    """Carrega uma imagem BGR da pasta dados. A leitura usa a raiz baseada em __file__,
    portanto o programa funciona independentemente do diretorio de trabalho.
    Lanca um erro compreensivel se a imagem nao estiver disponivel."""
    p = DADOS / nome
    x = cv2.imread(str(p))
    if x is None:
        raise FileNotFoundError(p)
    return x

def save(exemplo, nome, x):
    """Cria a pasta deste exercicio e salva uma imagem com cv2.imwrite.
    Retorna o caminho e interrompe com erro se a gravacao falhar."""
    folder = OUT / exemplo
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / nome
    if not cv2.imwrite(str(file), x):
        raise OSError(file)
    print('Salvo:', file)
    return file

def letterbox(x, shape=(224, 224), color=(114, 114, 114)):
    """Padroniza o tamanho sem alterar a proporcao da cena.
    Primeiro calcula escala=min(largura_destino/largura, altura_destino/altura).
    Redimensiona uniformemente e distribui o padding a esquerda/direita e topo/baixo.
    Retorna imagem, fator de escala e posicao do canto superior esquerdo do conteudo.
    A tupla shape segue a ordem (largura, altura), como as funcoes de resize do OpenCV."""
    dest_w, dest_h = shape
    h, w = x.shape[:2]
    escala = min(dest_w / w, dest_h / h)
    nw = max(1, int(round(w * escala)))
    nh = max(1, int(round(h * escala)))
    resized = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR)
    pl = (dest_w - nw) // 2
    pt = (dest_h - nh) // 2
    pr = dest_w - nw - pl
    pb = dest_h - nh - pt
    y = cv2.copyMakeBorder(resized, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=color)
    return (y, escala, (pl, pt))

def panel(exemplo, nomes, images, cols=3):
    """Cria um painel comparativo para a aula sem janelas GUI.
    Cada imagem e convertida quando necessario de cinza a BGR, passa por letterbox
    para nao ser distorcida, e recebe legenda; o painel final e salvo no disco.
    Assim o mesmo codigo funciona em terminal, VS Code e Colab."""
    tw, th = (360, 220)
    cell_h = th + 38
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((cell_h * rows, tw * cols, 3), 245, np.uint8)
    for j, (label, a) in enumerate(zip(nomes, images)):
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        thumb, _, _ = letterbox(a, (tw, th), color=(30, 30, 30))
        y = j // cols * cell_h
        x = j % cols * tw
        canvas[y:y + th, x:x + tw] = thumb
        cv2.putText(canvas, label[:44], (x + 9, y + th + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 36, 50), 1, cv2.LINE_AA)
    return save(exemplo, 'painel.png', canvas)

def affine_warp(x, M, border=cv2.BORDER_CONSTANT):
    """Aplica a matriz afim 2x3 com cv2.warpAffine sem alterar o tamanho externo.
    INTER_LINEAR estima pixels entre coordenadas e borderMode define o preenchimento.
    A transformacao descreve coordenadas geometricas; nao altera automaticamente anotacoes."""
    h, w = x.shape[:2]
    return cv2.warpAffine(x, np.asarray(M, dtype=np.float32), (w, h), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=(30, 30, 30))

def rotate_zoom(x, angle=0, scale=1.0, border=cv2.BORDER_REFLECT_101):
    """Calcula rotacao e escala em torno do centro e aplica a transformacao.
    angle e medido em graus e scale e multiplicador (1 = tamanho constante).
    Devolve a imagem transformada e a matriz 2x3 para inspecao do estudante."""
    h, w = x.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return (cv2.warpAffine(x, m, (w, h), borderMode=border), m)

TAG = '15_invertendo_afim'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Carrega a fotografia BGR original, usada como referencia para comparar antes e depois.
    a = img()

    # Constroi e aplica rotacao/escala em torno do centro; a matriz resultante descreve o
    # mapeamento geometrico.
    b, M = rotate_zoom(a, 8, 1.05)

    # Encontra a matriz inversa da transformacao afim; perdas por interpolacao nao sao
    # reversiveis.
    inv = cv2.invertAffineTransform(M)

    # Transforma a imagem com warpAffine e uma matriz 2x3, preservando o tamanho definido para o
    # canvas.
    c = affine_warp(b, inv, border=cv2.BORDER_REFLECT_101)

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('M=\n', M, '\nM_inversa=\n', inv)

    # Reune original e transformacoes em um painel legendado, salvo em arquivo de imagem.
    panel(TAG, ['Original', 'Deformada', 'Inversa aproximada'], [a, b, c])

if __name__ == '__main__':
    main()
