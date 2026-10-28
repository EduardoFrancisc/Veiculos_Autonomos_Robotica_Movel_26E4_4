"""EXEMPLO 40 — Auditoria independente do dataset

OBJETIVO E CONTEXTO:
Este programa executa uma auditoria autônoma das 20 imagens do dataset: aplica letterbox, separa treino e validação, calcula estatísticas RGB do treino e verifica dimensões, valores finitos e ausência de sobreposição entre as partições. Também simula uma augmentação restrita às amostras de treino e confere que os rótulos demonstrativos mudam de sinal após flip horizontal. A importância desse exercício está em validar o pipeline antes de usar os dados em redes neurais, identificando erros silenciosos de formato, normalização, rotulagem e separação dos conjuntos.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/40_auditoria_dataset/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
recarregar as 20 imagens, padronizar geometria, fazer split e checar valores, rotulos simulados e shapes.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
Validacao de dados evita falhas silenciosas no treinamento; o exemplo nao depende de resultados de outros scripts.

EXERCICIO PARA OS ALUNOS:
Provocar uma condicao incorreta e conferir qual assert detecta o erro.

EXECUCAO:
python exemplos/40_auditoria_dataset.py
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

TAG = '40_auditoria_dataset'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Le o conjunto de 20 imagens e os metadados; verifica-se a disponibilidade dos arquivos.
    entradas = read20()

    # Aplica escala unica e padding proporcional; evita distorcer a relacao largura/altura dos
    # objetos.
    rgb = np.stack([cv2.cvtColor(letterbox(frame, (224, 224))[0], cv2.COLOR_BGR2RGB) for frame, metadata, nome in entradas]).astype(np.float32) / 255.0

    # Confere formato: 20 imagens, 224 de altura, 224 de largura, tres canais RGB.
    assert rgb.shape == (20, 224, 224, 3)

    # Fixa uma semente para que os sorteios do experimento sejam reproduziveis.
    rng = np.random.default_rng(2026)

    # Verificar disjuncao de treino e validacao evita usar a mesma amostra nos dois grupos.
    indices = rng.permutation(20)

    # Reserva 16 dos 20 indices como amostras de treino.
    treino = indices[:16]

    # Reserva os quatro indices restantes para validacao.
    validacao = indices[16:]

    # Confere uma propriedade obrigatoria; a execucao para se a verificacao falhar.
    assert set(treino).isdisjoint(set(validacao))

    # Calcula estatisticas numéricas. Em pipelines de treinamento, use somente amostras do
    # conjunto de treino.
    media = rgb[treino].mean((0, 1, 2))

    # Calcula estatisticas numéricas. Em pipelines de treinamento, use somente amostras do
    # conjunto de treino.
    desvio = np.maximum(rgb[treino].std((0, 1, 2)), 1e-06)

    # Padroniza os pixels pela media e pelo desvio calculados somente sobre o treino.
    zscore = (rgb - media) / desvio

    # Confere uma propriedade obrigatoria; a execucao para se a verificacao falhar.
    assert np.isfinite(zscore).all()

    # Inicializa a lista de imagens originais e espelhadas usadas na checagem.
    dados_augmentados = []

    # Inicializa os valores artificiais de steering, somente para testar inversao de sinal.
    labels_simulados = []

    # Converte os canais de cor, pois a leitura BGR do OpenCV difere da convencao RGB de muitas
    # CNNs.
    for i in treino:
        frame = cv2.cvtColor((rgb[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        theta = float(entradas[i][1]['steering_simulado_rad'])
        dados_augmentados.append(frame)
        labels_simulados.append(theta)
        frame_espelhado = cv2.flip(frame, 1)
        dados_augmentados.append(frame_espelhado)
        labels_simulados.append(-theta)
        assert labels_simulados[-1] == -labels_simulados[-2]

    # Confere uma propriedade obrigatoria; a execucao para se a verificacao falhar.
    assert len(dados_augmentados) == 32

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Dataset:', rgb.shape)

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Treino:', sorted(map(int, treino)))

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Validacao:', sorted(map(int, validacao)))

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Medias RGB do treino:', media)

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Desvios RGB do treino:', desvio)

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Amostras de augmentacao (somente treino):', len(dados_augmentados))

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('APROVADO: resolucao, valores finitos, separacao, augmentation e inversao simulada.')

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('ATENCAO: o steering e SIMULADO e nao deve treinar controle real.')

    # Salva a evidencia visual do processamento na pasta exclusiva deste programa.
    save(TAG, 'grid_4x5_auditado.png', grid_4x5([cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for x in rgb]))

    # Grava arrays e metadados em NPZ comprimido, facilitando carregar o conjunto no NumPy.
    np.savez_compressed(OUT / TAG / 'auditoria_independente.npz', shape=rgb.shape, train_indices=treino, val_indices=validacao, mean_rgb=media, std_rgb=desvio, num_augmentados=np.array(len(dados_augmentados)))

if __name__ == '__main__':
    main()
