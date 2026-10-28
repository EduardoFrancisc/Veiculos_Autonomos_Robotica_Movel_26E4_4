"""EXEMPLO 36 — Pipeline integrado de imagens para CNN

OBJETIVO E CONTEXTO:
Este programa integrado executa um pipeline completo: carrega 20 imagens frontais com variações controladas, aplica letterbox para 224 × 224 e converte os pixels para RGB e valores entre zero e um. Antes da augmentação, separa treino e validação, calcula médias e desvios apenas no treino e aplica z-score. Gera versões com rotação de até 10 graus, zoom de até 10% e flip horizontal, registrando os parâmetros e invertendo o sinal apenas de rótulos demonstrativos simulados. Por fim, salva o dataset em NPZ, imagens processadas, metadados CSV, relatório e grids 4 × 5, permitindo apresentar todas as etapas exigidas pela atividade.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/36_pipeline_dataset_cnn/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
20 frames, letterbox 224x224, RGB, z-score do treino, divisao e augmentation.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
Os steerings sao simulados; os quadros partem de uma mesma sequencia e nao fornecem validacao temporal independente.

EXERCICIO PARA OS ALUNOS:
Inspecionar os dois grids e o arquivo NPZ com dados padronizados e metadados.

EXECUCAO:
python exemplos/36_pipeline_dataset_cnn.py
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

def rotate_zoom(x, angle=0, scale=1.0, border=cv2.BORDER_REFLECT_101):
    """Calcula rotacao e escala em torno do centro e aplica a transformacao.
    angle e medido em graus e scale e multiplicador (1 = tamanho constante).
    Devolve a imagem transformada e a matriz 2x3 para inspecao do estudante."""
    h, w = x.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return (cv2.warpAffine(x, m, (w, h), borderMode=border), m)

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

TAG = '36_pipeline_dataset_cnn'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Le o conjunto de 20 imagens e os metadados; verifica-se a disponibilidade dos arquivos.
    entries = read20()

    # Fixa uma semente para que os sorteios do experimento sejam reproduziveis.
    rng = np.random.default_rng(2026)

    # Aplica escala unica e padding proporcional; evita distorcer a relacao largura/altura dos
    # objetos.
    images_bgr = [letterbox(a, shape=(224, 224))[0] for a, meta, nome in entries]

    # Empilha as imagens em um array de quatro dimensoes: amostra, altura, largura e canais.
    orig = np.stack(images_bgr).astype(np.uint8)

    # Converte os canais de cor, pois a leitura BGR do OpenCV difere da convencao RGB de muitas
    # CNNs.
    orig_rgb = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in orig])

    # Mantem os calculos numericos em float32 para evitar perda de precisao em operacoes
    # geometricas.
    orig_rgb_float = orig_rgb.astype(np.float32) / 255.0

    # IMPORTANTE: o split treino/validacao vem ANTES do data augmentation.
    ix = rng.permutation(20)

    # Seleciona os 16 indices que representam o conjunto de treinamento (80% das 20 imagens).
    ix_train = np.sort(ix[:16])

    # Seleciona os quatro indices restantes para validacao; nao compartilham indices com treino.
    ix_val = np.sort(ix[16:])

    # Media e desvio sao calculados somente nos dados de treinamento para evitar vazamento.
    mu = orig_rgb_float[ix_train].mean(axis=(0, 1, 2))

    # Calcula estatisticas numéricas. Em pipelines de treinamento, use somente amostras do
    # conjunto de treino.
    sd = orig_rgb_float[ix_train].std(axis=(0, 1, 2))

    # Impede desvio-padrao zero ou quase zero no denominador do z-score.
    sd = np.maximum(sd, 1e-06)

    # Estatisticas do treino devem ser reutilizadas na validacao, sem recalculo.
    X_norm = (orig_rgb_float - mu) / sd

    # Cria a lista que armazenara as imagens RGB produzidas pelo aumento de dados.
    aug_rgb = []

    # Rotacao, zoom e flip criam apenas versoes da parte de treino. Nao augmentar validacao.
    aug_steer = []

    # Cria uma lista de metadados para rastrear origem, angulo, zoom, flip e tipo de versao.
    aug_meta = []

    # Percorre APENAS os indices de treino. Cada amostra produz duas versoes:
    # a primeira e fiel ao frame e a segunda recebe perturbacoes geometricas.
    # O rótulo artificial só muda de sinal no caso do flip, o que NAO faz
    # da rotacao ou do zoom transformacoes fisicamente rotuladas para condução.
    for src_index in ix_train:
        a = orig[src_index]
        raw_label = float(entries[src_index][1]['steering_simulado_rad'])
        # Rotacao +/-10 graus e zoom entre 0.9 e 1.1 sao aplicados apenas
        # na segunda versao. Na primeira, mantemos os pixels sem mudancas.
        for version in ('original', 'aug'):
            # Original sem perturbacoes; a outra variante e sorteada.
            if version == 'original':
                b = a.copy()
                ang = 0.0
                scale = 1.0
                flipped = False
                steer = raw_label
            else:
                ang = float(rng.uniform(-10, 10))
                scale = float(rng.uniform(0.9, 1.1))
                b, _ = rotate_zoom(a, ang, scale)
                flipped = bool(rng.random() < 0.5)
                # Reflexao horizontal exige ajuste de sinal de rótulo simetrico.
                if flipped:
                    b = cv2.flip(b, 1)
                steer = -raw_label if flipped else raw_label
            rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            aug_rgb.append(rgb)
            aug_steer.append(steer)
            # Salva a proveniencia dos novos exemplos para auditoria.
            aug_meta.append({'indice_origem': int(src_index), 'versao': version, 'rotacao_graus': round(ang, 3), 'zoom': round(scale, 4), 'flip_horizontal': int(flipped), 'steering_simulado_rad': round(steer, 5), 'rotulo_verdadeiro_disponivel': False})

    # Converte todas as imagens aumentadas para array uint8, pronto para salvar e visualizar.
    aug = np.asarray(aug_rgb, np.uint8)

    # Mantem os calculos numericos em float32 para evitar perda de precisao em operacoes
    # geometricas.
    aug_z = (aug.astype(np.float32) / 255.0 - mu) / sd

    # Seleciona os tensores padronizados pertencentes exclusivamente ao conjunto de treino.
    X_train = X_norm[ix_train]

    # Seleciona os tensores de validacao e aplica as estatisticas definidas no treino.
    X_val = X_norm[ix_val]

    # Rótulos de steering são SIMULADOS e NAO podem ser usados para treinar controle de veiculo.
    steering_simulado = np.asarray([float(meta['steering_simulado_rad']) for a, meta, n in entries], np.float32)

    # Define a pasta onde os resultados deste programa serao gravados.
    folder = OUT / TAG

    # Cria a pasta de saida, mesmo que ela ainda nao exista.
    folder.mkdir(parents=True, exist_ok=True)

    # Grava arrays e metadados em NPZ comprimido, facilitando carregar o conjunto no NumPy.
    np.savez_compressed(folder / 'dataset_224_normalizado_e_aumentado.npz', images_rgb_uint8=orig_rgb, images_zscore=X_norm, train_indices=ix_train, val_indices=ix_val, X_train=X_train, X_val=X_val, X_train_augmented_zscore=aug_z, images_augmented_rgb=aug, steering_SIMULADO=steering_simulado, steering_aug_SIMULADO=np.asarray(aug_steer, np.float32), mean_rgb=mu, std_rgb=sd)

    # Abre um arquivo de saida com fechamento automatico para escrever relatorio ou metadados.
    with open(folder / 'metadados_augmentation.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(aug_meta[0]))
        wr.writeheader()
        wr.writerows(aug_meta)

    # Salva a evidencia visual do processamento na pasta exclusiva deste programa.
    save(TAG, 'grid_4x5_normalizadas.png', grid_4x5(images_bgr, [f'Imagem {i + 1:02d}' for i in range(20)]))

    # Converte os canais de cor, pois a leitura BGR do OpenCV difere da convencao RGB de muitas
    # CNNs.
    aug_bgr = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in aug[:20]]

    # Salva a evidencia visual do processamento na pasta exclusiva deste programa.
    save(TAG, 'grid_4x5_augmented.png', grid_4x5(aug_bgr, [f'Aug {i + 1:02d}' for i in range(20)]))

    # Percorre as amostras ou configuracoes; cada repeticao realiza as operacoes explicadas no
    # bloco indentado.
    for i, x in enumerate(images_bgr):
        save(TAG, f'normalizada_{i + 1:02d}.png', x)

    # Abre um arquivo de saida com fechamento automatico para escrever relatorio ou metadados.
    with open(folder / 'relatorio.txt', 'w', encoding='utf-8') as f:
        f.write(f'Entrada RGB: {orig_rgb.shape}\nZ-score: {X_norm.shape}\n')
        f.write(f'Treino {len(ix_train)} | validacao {len(ix_val)} | aug treino {len(aug)}\n')
        f.write(f'Media RGB (treino) {mu}\nDesvio RGB (treino) {sd}\n')
        f.write('ATENCAO: steerings sao simulados; nao servem como rotulos de conducao.\n')
        f.write('Rotacao/zoom podem exigir correcao fisica dos rotulos verdadeiros; flip requer theta=-theta.\n')
        f.write('Motivos: letterbox preserva proporcao; zscore estabiliza escala; augmentacao amplia variabilidade.\n')

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Imagens originais', orig_rgb.shape, 'treino', X_train.shape, 'validacao', X_val.shape, 'augmentadas', aug.shape)

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Media RGB treino', mu, 'desvio RGB treino', sd)

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Flip demonstrado: theta -> -theta; angulos reais NAO disponiveis.')

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Pipeline completo salvo em:', folder)

if __name__ == '__main__':
    main()
