"""EXEMPLO 35 — Demonstracao integrada de transformacoes

OBJETIVO E CONTEXTO:
Este programa integrador demonstra a aplicacao conjunta das transformacoes: sobre uma imagem frontal, ele executa translação, rotação em torno do centro, escala, matriz afim personalizada e a composição entre rotação e translação. Todas as matrizes são impressas no terminal, enquanto os resultados são salvos individualmente e reunidos em um painel comparativo. O exercício consolida as quatro transformações fundamentais e demonstra que a multiplicação de matrizes homogêneas permite organizar etapas de pré-processamento de forma matematicamente consistente.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/35_transformacoes_integradas/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
aplicar translacao, rotacao, escala, afim personalizada e matriz composta R@T.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
Uma unica demonstracao integra a geometria de pré-processamento aplicada a imagens de camera frontal.

EXERCICIO PARA OS ALUNOS:
Reproduzir os resultados com parametros menores e interpretar cada matriz impressa.

EXECUCAO:
python exemplos/35_transformacoes_integradas.py
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

def H(M):
    """Converte matriz afim 2x3 para a representacao homogenea 3x3.
    A ultima linha [0,0,1] permite compor translacao e transformacoes lineares
    usando multiplicacao matricial com vetores-coluna."""
    return np.vstack([np.asarray(M, dtype=float), [0.0, 0.0, 1.0]])

TAG = '35_transformacoes_integradas'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Carrega a fotografia BGR original, usada como referencia para comparar antes e depois.
    a = img()

    # Obtem as dimensoes da imagem: o OpenCV organiza primeiro altura (linhas), depois largura
    # (colunas).
    h, w = a.shape[:2]

    # Define explicitamente os seis coeficientes da matriz afim 2x3; a terceira coluna contem
    # translacoes.
    T = np.float32([[1, 0, 65], [0, 1, 22]])

    # Constroi e aplica rotacao/escala em torno do centro; a matriz resultante descreve o
    # mapeamento geometrico.
    R = cv2.getRotationMatrix2D((w / 2, h / 2), +10, 1.0)

    # Constroi e aplica rotacao/escala em torno do centro; a matriz resultante descreve o
    # mapeamento geometrico.
    Z = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.1)

    # Define explicitamente os seis coeficientes da matriz afim 2x3; a terceira coluna contem
    # translacoes.
    A = np.float32([[1, 0.1, -20], [-0.02, 0.98, 20]])

    # H(R) @ H(T) combina primeiro translacao e depois rotacao. A ultima linha e dispensada em
    # warpAffine.
    C = (H(R) @ H(T))[:2, :]

    # Percorre as amostras ou configuracoes; cada repeticao realiza as operacoes explicadas no
    # bloco indentado.
    for name, m in [('Translacao', T), ('Rotacao', R), ('Zoom', Z), ('Afim personalizada', A), ('R@T', C)]:
        print(name, '=\n', m)

    # Transforma a imagem com warpAffine e uma matriz 2x3, preservando o tamanho definido para o
    # canvas.
    ims = [a, affine_warp(a, T), affine_warp(a, R), affine_warp(a, Z), affine_warp(a, A), affine_warp(a, C)]

    # Reune original e transformacoes em um painel legendado, salvo em arquivo de imagem.
    panel(TAG, ['Original', 'Translacao', 'Rotacao', 'Zoom', 'Afim', 'Composicao R@T'], ims, 3)

    # Transforma a imagem com warpAffine e uma matriz 2x3, preservando o tamanho definido para o
    # canvas.
    for name, m in [('translacao', T), ('rotacao', R), ('escala', Z), ('afim', A), ('composicao', C)]:
        save(TAG, name + '.png', affine_warp(a, m))

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('APLICACOES: translacao (deslocamento simulado), rotacao (alinhamento), zoom (escala), afim (registro), composicao (pipeline)')

if __name__ == '__main__':
    main()
