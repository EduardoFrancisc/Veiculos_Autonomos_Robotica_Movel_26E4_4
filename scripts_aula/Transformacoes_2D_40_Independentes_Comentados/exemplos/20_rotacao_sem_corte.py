"""EXEMPLO 20 — Rotacao com area de saida expandida

OBJETIVO E CONTEXTO:
O programa gira a imagem em 25 graus e recalcula as dimensões da área de saída para acomodar o conteúdo rotacionado sem o corte provocado por um canvas fixo. Em seguida, ajusta os termos de translação da matriz e executa cv2.warpAffine. A técnica é útil em alinhamento e preparação de dados quando é necessário preservar objetos e anotações próximos às bordas.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/20_rotacao_sem_corte/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
calcular a caixa envolvente da imagem rotacionada e ajustar a translacao da matriz.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
Aumentar o canvas ajuda a manter objetos nas bordas, embora inclua regioes preenchidas.

EXERCICIO PARA OS ALUNOS:
Testar 10 e 45 graus e conferir a mudanca de tamanho.

EXECUCAO:
python exemplos/20_rotacao_sem_corte.py
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

TAG = '20_rotacao_sem_corte'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Carrega a fotografia BGR original, usada como referencia para comparar antes e depois.
    a = img()

    # Obtem as dimensoes da imagem: o OpenCV organiza primeiro altura (linhas), depois largura
    # (colunas).
    h, w = a.shape[:2]

    # Define o angulo de rotacao em graus para calcular o tamanho do canvas expandido.
    ang = 25

    # Constroi e aplica rotacao/escala em torno do centro; a matriz resultante descreve o
    # mapeamento geometrico.
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)

    # Obtém o valor absoluto do componente de cosseno presente na matriz de rotacao.
    cos = abs(M[0, 0])

    # Obtém o valor absoluto do componente de seno presente na matriz de rotacao.
    sin = abs(M[0, 1])

    # Calcula a largura do conteudo depois da escala unica do letterbox.
    nw = int(h * sin + w * cos)

    # Calcula a altura do conteudo depois da escala unica do letterbox.
    nh = int(h * cos + w * sin)

    # Ajusta a translacao horizontal para centralizar a imagem no canvas expandido.
    M[0, 2] += nw / 2 - w / 2

    # Ajusta a translacao vertical para centralizar a imagem no canvas expandido.
    M[1, 2] += nh / 2 - h / 2

    # Executa uma operacao OpenCV sobre os pixels ou as coordenadas da fotografia.
    b = cv2.warpAffine(a, M, (nw, nh), borderMode=cv2.BORDER_CONSTANT)

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Tamanho original', w, h, 'tamanho novo', nw, nh, 'M=\n', M)

    # Reune original e transformacoes em um painel legendado, salvo em arquivo de imagem.
    panel(TAG, ['Original', 'Canvas expandido'], [a, b], 2)

if __name__ == '__main__':
    main()
