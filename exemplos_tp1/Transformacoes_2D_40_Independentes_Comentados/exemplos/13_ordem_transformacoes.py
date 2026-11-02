"""EXEMPLO 13 — Ordem de aplicacao de transformacoes

OBJETIVO E CONTEXTO:
O código constrói matrizes de rotação e translação e compara os produtos R × T e T × R, mostrando que a ordem das operações modifica o resultado final. Ele aplica as duas composições à mesma imagem e imprime as matrizes calculadas. A lição é importante para pipelines de alinhamento de câmeras, nos quais alterar a ordem das operações pode deslocar pontos e objetos para posições incorretas.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/13_ordem_transformacoes/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
R@T aplica primeiro T e depois R a vetores-coluna; T@R altera a sequencia.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
Multiplicacao de matrizes geralmente nao e comutativa: R@T e T@R produzem cenas diferentes.

EXERCICIO PARA OS ALUNOS:
Comparar numericamente as duas matrizes e os painéis gerados.

EXECUCAO:
python exemplos/13_ordem_transformacoes.py
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

TAG = '13_ordem_transformacoes'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Carrega a fotografia BGR original, usada como referencia para comparar antes e depois.
    a = img()

    # Obtem as dimensoes da imagem: o OpenCV organiza primeiro altura (linhas), depois largura
    # (colunas).
    h, w = a.shape[:2]

    # Constroi a matriz de translacao em coordenadas homogeneas para compor com outras matrizes.
    T = H([[1, 0, 70], [0, 1, 0]])

    # Constroi e aplica rotacao/escala em torno do centro; a matriz resultante descreve o
    # mapeamento geometrico.
    R = H(cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1))

    # Em vetores-coluna, R @ T corresponde a aplicar T primeiro e R depois.
    A = R @ T

    # A troca da ordem altera o ponto sobre o qual a rotacao atua.
    B = T @ R

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('R@T=\n', A, '\nT@R=\n', B, '\nIguais?', np.allclose(A, B))

    # Reune original e transformacoes em um painel legendado, salvo em arquivo de imagem.
    panel(TAG, ['Original', 'Aplicar T depois R', 'Aplicar R depois T'], [a, affine_warp(a, A[:2]), affine_warp(a, B[:2])])

if __name__ == '__main__':
    main()
