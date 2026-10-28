# Protocolos de comunicação veicular — exemplos separados

Prof. Vitor Amadeu Souza · Veículos Autônomos e Robótica Móvel

O ZIP contém 36 pastas numeradas, uma por exercício. Cada pasta traz seu código,
os módulos necessários, requisitos, instruções e um notebook específico.
Nos exercícios que usam câmera, o vídeo de direção da Udacity já está incluído
na própria pasta. Não é preciso fornecer uma imagem nem um vídeo de referência.

Você pode copiar qualquer pasta inteira para outro lugar e executá-la sozinha.
Os códigos não dependem dos resultados dos exemplos anteriores. Para aprender,
siga de 01 a 36; para executar, você pode escolher qualquer exemplo.

## Começar no VS Code

1. Extraia o ZIP.
2. Abra a pasta do exercício desejado no VS Code.
3. Leia o `LEIA_ME.md` dessa pasta e os comentários no início do `.py`.
4. Instale as dependências com `python -m pip install -r requirements.txt`.
5. Execute o arquivo numerado: por exemplo, `python 01_ambiente.py --sem-gui`.

O ambiente Python pode ser reutilizado nos 36 exercícios: as dependências são iguais.
Não execute `apoio.py`, `can_laboratorio.py` ou `rede_laboratorio.py` como exercício;
eles são módulos auxiliares chamados pelo código numerado.

## Começar no Colab

Abra o notebook da pasta escolhida. Execute as células em ordem e envie este
ZIP quando solicitado. O notebook extrai somente a pasta daquele exercício.
Cada notebook inclui instalação, execução e visualização dos resultados.

## Atalhos para a tarefa

| Requisito | Pasta |
|---|---|
| Item A: sensores CAN, dois processos, CSV e séries | `14_item_a_processos/` |
| Item B: UDP/TCP, 320 frames cada, feeds e comparação | `28_item_b_comparacao/` |
| Projeto completo dos itens A e B | `36_projeto_integrador/` |
| Servidor e cliente em dois terminais/LAN | `34_relogios_e_lan/` |

`INDICE.csv` lista os 36 temas e os caminhos para cada código e notebook.
As saídas são geradas dentro da pasta do exercício, em `resultados/`.
As medições refletem a máquina utilizada: loopback não representa uma rede
automotiva física. Os detalhes e limitações estão nos códigos e relatórios.

O pacote repete os módulos e os vídeos entre pastas para que cada exemplo possa
ser copiado sozinho. Por isso, seu tamanho é maior que o do pacote original.
