"""EXEMPLO 38 — Treino, validacao e risco de vazamento

OBJETIVO E CONTEXTO:
O programa demonstra como dividir 20 imagens em conjuntos distintos de treino e validação antes de qualquer data augmentation. Ele apresenta os índices escolhidos, verifica se não existe sobreposição e calcula média e desvio-padrão RGB somente a partir do treino. O exercício introduz o problema do vazamento de dados: versões transformadas do mesmo quadro não devem aparecer simultaneamente no treinamento e na avaliação, pois isso poderia produzir resultados excessivamente otimistas.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/38_comparar_estatisticas_splits/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
separar amostras antes de augmentation e calcular estatisticas exclusivamente no treino.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
Frames adjacentes do mesmo video podem ser correlacionados; em experimento rigoroso separe por sequencia/viagem.

EXERCICIO PARA OS ALUNOS:
Comparar as medias RGB do treino com as do conjunto inteiro.

EXECUCAO:
python exemplos/38_comparar_estatisticas_splits.py
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

def read20():
    """Le exatamente 20 quadros frontais e seus metadados CSV.
    Valida a quantidade esperada e devolve tuplas (imagem BGR, metadados, nome).
    O campo steering_simulado_rad e didatico e NAO representa direcao real."""
    import csv
    files = sorted((DADOS / 'camera_frontal_20').glob('frontal_*.jpg'))
    if len(files) != 20:
        raise ValueError(f'Esperadas 20 imagens, presentes {len(files)}')
    with open(DADOS / 'metadados_20.csv', encoding='utf-8') as f:
        meta = {r['arquivo']: r for r in csv.DictReader(f)}
    return [(cv2.imread(str(p)), meta[p.name], p.name) for p in files]

def grid_4x5(images, labels=None, w=252, h=176):
    """Organiza exatamente 20 imagens em quatro linhas e cinco colunas.
    Acrescenta identificadores e devolve uma imagem BGR de visualizacao,
    separada dos tensores float32 usados para treinamento."""
    if len(images) != 20:
        raise ValueError('Grid exige exatamente 20 imagens')
    canvas = np.full((4 * (h + 34), 5 * w, 3), 250, np.uint8)
    for i, a in enumerate(images):
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        x = i % 5 * w
        y = i // 5 * (h + 34)
        canvas[y:y + h, x:x + w] = a
        cv2.putText(canvas, (labels[i] if labels else f'Imagem {i + 1:02d}')[:31], (x + 5, y + h + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 45, 60), 1, cv2.LINE_AA)
    return canvas

TAG = '38_comparar_estatisticas_splits'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Le o conjunto de 20 imagens e os metadados; verifica-se a disponibilidade dos arquivos.
    items = read20()

    # Aplica escala unica e padding proporcional; evita distorcer a relacao largura/altura dos
    # objetos.
    rgb = np.stack([cv2.cvtColor(letterbox(x)[0], cv2.COLOR_BGR2RGB).astype('float32') / 255.0 for x, m, n in items])

    # Fixa uma semente para que os sorteios do experimento sejam reproduziveis.
    rng = np.random.default_rng(42)

    # Separe os indices por sequencia/viagem em experimentos reais, para evitar frames
    # correlacionados nos dois grupos.
    ix = rng.permutation(20)

    # Seleciona os 16 indices destinados ao treino antes de qualquer transformacao adicional.
    tr = ix[:16]

    # Seleciona os indices de validacao que permanecem separados do treino.
    val = ix[16:]

    # Calcula estatisticas numéricas. Em pipelines de treinamento, use somente amostras do
    # conjunto de treino.
    mu = rgb[tr].mean((0, 1, 2))

    # Calcula estatisticas numéricas. Em pipelines de treinamento, use somente amostras do
    # conjunto de treino.
    std = rgb[tr].std((0, 1, 2))

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Indices treino:', sorted(tr), '\nIndices val:', sorted(val))

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Sem intersecao?', len(set(tr) & set(val)) == 0)

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('mu treino:', mu, 'std treino:', std)

    # Salva a evidencia visual do processamento na pasta exclusiva deste programa.
    save(TAG, 'grid_train.png', grid_4x5([cv2.cvtColor((rgb[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for i in range(20)]))

if __name__ == '__main__':
    main()
