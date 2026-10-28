# Exemplo 10 — Inspecionar o registro produzido

Gera sua própria captura CAN e relê o CSV, sem depender do exemplo 09. Verifica a quantidade por ID e a progressão dos contadores.

Esta pasta é independente das outras 35. Você pode copiar a pasta inteira para
outro local. Mantenha os módulos Python e os demais arquivos que estão nela.
Este exemplo não precisa de imagem nem vídeo.

## VS Code

Abra esta pasta no VS Code. Instale Python 3.10 ou superior e selecione o interpretador.
No terminal aberto nesta pasta, instale uma vez:

```bash
python -m pip install -r requirements.txt
python 10_inspecionar_csv_can.py --sem-gui
```

No Linux/macOS, use `python3` se `python` não estiver disponível.
No Windows, também pode usar `py`. Um ambiente virtual é recomendado:
`python -m venv .venv`; selecione seu interpretador no VS Code antes da instalação.
Não precisa executar outro exemplo antes deste.

## Google Colab

1. Abra `10_inspecionar_csv_can.ipynb` no Colab, com runtime CPU.
2. Execute as células em ordem.
3. Envie `Protocolos_Veiculares_36_Exemplos_Separados.zip` quando solicitado.
4. O notebook extrai somente este exemplo, instala as bibliotecas e executa o `.py`.

No Colab os gráficos e feeds são mostrados no notebook, sem janelas nativas.
O notebook precisa acompanhar o ZIP para a primeira execução.

## Resultado e observação

Saídas: `resultados/10_inspecionar_csv_can/`. Os arquivos são gerados por este exercício.
A contagem esperada é ciclos × 3. Conferir sequências ajuda a detectar ausência e duplicação na aplicação.

Nos exemplos de streaming, `--mostrar` exibe o cliente em uma janela no desktop.
`--log-cada 1` imprime cada frame; `--video arquivo.mp4` permite sua própria mídia.
Os exemplos 28 e 36 usam 320 frames por protocolo e exigem no mínimo 300.
CAN, quando usado, é virtual; a ponte entre processos e os limites de medição
estão explicados no código. As medições de rede locais usam loopback.

## Arquivos auxiliares

`apoio.py`: argumentos, pastas, versões e gráficos.
`can_laboratorio.py`: mensagens, sensores e barramento CAN virtual.
`rede_laboratorio.py`: sockets, JPEG, remontagem e métricas.
`requirements.txt`: dependências para desktop.
`requirements-colab.txt`: dependências sem janelas para Colab.
