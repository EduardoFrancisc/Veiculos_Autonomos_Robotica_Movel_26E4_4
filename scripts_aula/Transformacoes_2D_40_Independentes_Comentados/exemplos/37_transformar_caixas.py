"""EXEMPLO 37 — Transformar caixa de anotacao

OBJETIVO E CONTEXTO:
O exemplo desenha uma caixa geométrica sobre a imagem, aplica uma rotação afim e transforma também os quatro vértices da anotação com cv2.transform. O resultado demonstra que modificar uma imagem sem atualizar suas anotações provoca divergências entre os dados e as respostas esperadas. Essa coerência é indispensável no treinamento de redes de detecção de objetos, segmentação e percepção de veículos autônomos.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/37_transformar_caixas/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
transformar os quatro cantos de uma caixa com a mesma matriz afim usada na imagem.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
Anotacoes precisam acompanhar a transformacao; o eixo alinhado final pode exigir uma nova caixa envolvente.

EXERCICIO PARA OS ALUNOS:
Comparar a caixa original com os vertices recalculados.

EXECUCAO:
python exemplos/37_transformar_caixas.py
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

TAG = '37_transformar_caixas'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Carrega a fotografia BGR original, usada como referencia para comparar antes e depois.
    a = img()

    # Obtem as dimensoes da imagem: o OpenCV organiza primeiro altura (linhas), depois largura
    # (colunas).
    h, w = a.shape[:2]

    # Mantem os calculos numericos em float32 para evitar perda de precisao em operacoes
    # geometricas.
    box = np.float32([[w * 0.38, h * 0.55], [w * 0.58, h * 0.55], [w * 0.58, h * 0.9], [w * 0.38, h * 0.9]])

    # Constroi e aplica rotacao/escala em torno do centro; a matriz resultante descreve o
    # mapeamento geometrico.
    R = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0)

    # Executa uma operacao OpenCV sobre os pixels ou as coordenadas da fotografia.
    dst = cv2.transform(box[None], R)[0]

    # Transforma a imagem com warpAffine e uma matriz 2x3, preservando o tamanho definido para o
    # canvas.
    b = affine_warp(a, R)

    # Cria uma copia para desenhar anotacoes sem perder a fotografia original.
    q = a.copy()

    # Executa uma operacao OpenCV sobre os pixels ou as coordenadas da fotografia.
    cv2.polylines(q, [box.astype(np.int32)], True, (0, 0, 255), 3)

    # Executa uma operacao OpenCV sobre os pixels ou as coordenadas da fotografia.
    cv2.polylines(b, [dst.astype(np.int32)], True, (0, 0, 255), 3)

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Vertices da caixa original -> novos vertices:', np.c_[box, dst])

    # Reune original e transformacoes em um painel legendado, salvo em arquivo de imagem.
    panel(TAG, ['Caixa original', 'Caixa transformada'], [q, b], 2)

if __name__ == '__main__':
    main()
