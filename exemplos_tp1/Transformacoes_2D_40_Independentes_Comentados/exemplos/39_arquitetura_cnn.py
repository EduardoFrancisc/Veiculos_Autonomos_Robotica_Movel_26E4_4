"""EXEMPLO 39 — Contrato de entrada de uma CNN

OBJETIVO E CONTEXTO:
O código mostra uma arquitetura ilustrativa de CNN em Keras preparada para receber imagens RGB padronizadas de 224 × 224 × 3 pixels e produzir uma saída numérica denominada steering. Se o TensorFlow estiver disponível, exibe a estrutura da rede; caso contrário, mantém a demonstração de pré-processamento em OpenCV. Não é realizado treinamento, pois o pacote não possui ângulos de direção verdadeiros. A finalidade é conectar o trabalho geométrico à etapa posterior de modelagem, explicando o contrato de entrada esperado por uma CNN.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/39_arquitetura_cnn/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
construir, quando TensorFlow estiver disponivel, uma CNN com entrada 224x224x3.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
A arquitetura ilustra o consumo dos tensores produzidos pelo pre-processamento, nao um controlador treinado.

EXERCICIO PARA OS ALUNOS:
Examinar model.summary() e contar parametros; nao iniciar treinamento sem rótulos reais.

EXECUCAO:
python exemplos/39_arquitetura_cnn.py
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

TAG = '39_arquitetura_cnn'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Contrato de entrada: RGB float32, 224x224x3, padronizado com medias do TREINO')

    # O modelo e apenas ilustrativo, nao existe aprendizado sem um dataset rotulado real.
    try:
        import tensorflow as tf
        m = tf.keras.Sequential([tf.keras.Input((224, 224, 3)), tf.keras.layers.Conv2D(16, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(1, name='steering_rad')])
        m.summary()
        print('ATENCAO: so arquitetura; nao treinar com steering simulado do pacote!')
    except ImportError:
        print('TensorFlow nao instalado: demonstracao de OpenCV executada normalmente.')

    # Carrega a fotografia BGR original, usada como referencia para comparar antes e depois.
    a = img()

    # Aplica escala unica e padding proporcional; evita distorcer a relacao largura/altura dos
    # objetos.
    b, _, _ = letterbox(a)

    # Reune original e transformacoes em um painel legendado, salvo em arquivo de imagem.
    panel(TAG, ['Imagem original', 'Entrada CNN 224x224'], [a, b], 2)

if __name__ == '__main__':
    main()
