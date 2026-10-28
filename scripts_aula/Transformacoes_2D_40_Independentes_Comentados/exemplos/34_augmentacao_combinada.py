"""EXEMPLO 34 — Combinacao de augmentations

OBJETIVO E CONTEXTO:
O código monta quatro variantes de uma mesma imagem por combinações aleatórias de rotação entre −10 e +10 graus, zoom entre 0,9 e 1,1 e espelhamento opcional. Uma semente fixa permite repetir os resultados durante a aula. O exemplo mostra como diferentes operações de data augmentation podem ser combinadas para aumentar a diversidade visual dos exemplos de treinamento e reforça a necessidade de preservar a coerência dos rótulos.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/34_augmentacao_combinada/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
sortear angulo, escala e espelhamento com semente fixa para reproduzir o experimento.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
Combinar transformacoes amplia diversidade, mas transformacoes invalidas ou rótulos inconsistentes prejudicam o treinamento.

EXERCICIO PARA OS ALUNOS:
Mudar a semente e observar novas amostras; conferir as variacoes permitidas.

EXECUCAO:
python exemplos/34_augmentacao_combinada.py
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

def rotate_zoom(x, angle=0, scale=1.0, border=cv2.BORDER_REFLECT_101):
    """Calcula rotacao e escala em torno do centro e aplica a transformacao.
    angle e medido em graus e scale e multiplicador (1 = tamanho constante).
    Devolve a imagem transformada e a matriz 2x3 para inspecao do estudante."""
    h, w = x.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return (cv2.warpAffine(x, m, (w, h), borderMode=border), m)

TAG = '34_augmentacao_combinada'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Carrega a fotografia BGR original, usada como referencia para comparar antes e depois.
    a = img()

    # Fixa uma semente para que os sorteios do experimento sejam reproduziveis.
    rng = np.random.default_rng(42)

    # Inicializa o conjunto de resultados com a fotografia original.
    images = [a]

    # Inicializa as legendas do painel; as novas amostras receberao seus parametros sortidos.
    titles = ['Original']

    # Rotacao e zoom sao sorteados; o espelhamento e opcional, mas exige tratamento adequado dos
    # rótulos.
    for j in range(4):
        angle = float(rng.uniform(-10, 10))
        scale = float(rng.uniform(0.9, 1.1))
        b, _ = rotate_zoom(a, angle, scale)
        if rng.random() < 0.5:
            b = cv2.flip(b, 1)
            f = 'flip'
        else:
            f = 'sem flip'
        images.append(b)
        titles.append(f'{angle:+.1f}g {scale:.2f} {f}')

    # Reune original e transformacoes em um painel legendado, salvo em arquivo de imagem.
    panel(TAG, titles, images, 3)

if __name__ == '__main__':
    main()
