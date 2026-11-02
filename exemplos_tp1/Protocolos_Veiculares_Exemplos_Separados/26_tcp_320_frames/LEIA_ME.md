# Exemplo 26 — Medir TCP com pelo menos 300 frames

Envia 320 frames por padrão e mede a mesma latência da aplicação. Identifica IDs não decodificados sem afirmar que isso mede perdas de segmentos TCP.

Esta pasta é independente das outras 35. Você pode copiar a pasta inteira para
outro local. Mantenha os módulos Python e os demais arquivos que estão nela.
O vídeo de entrada e sua licença estão em `dados/`. Não precisa fornecer mídia.

## VS Code

Abra esta pasta no VS Code. Instale Python 3.10 ou superior e selecione o interpretador.
No terminal aberto nesta pasta, instale uma vez:

```bash
python -m pip install -r requirements.txt
python 26_tcp_320_frames.py --sem-gui
```

No Linux/macOS, use `python3` se `python` não estiver disponível.
No Windows, também pode usar `py`. Um ambiente virtual é recomendado:
`python -m venv .venv`; selecione seu interpretador no VS Code antes da instalação.
Não precisa executar outro exemplo antes deste.

## Google Colab

1. Abra `26_tcp_320_frames.ipynb` no Colab, com runtime CPU.
2. Execute as células em ordem.
3. Envie `Protocolos_Veiculares_36_Exemplos_Separados.zip` quando solicitado.
4. O notebook extrai somente este exemplo, instala as bibliotecas e executa o `.py`.

No Colab os gráficos e feeds são mostrados no notebook, sem janelas nativas.
O notebook precisa acompanhar o ZIP para a primeira execução.

## Resultado e observação

Saídas: `resultados/26_tcp_320_frames/`. Os arquivos são gerados por este exercício.
Retransmissões TCP não são contadas por esse código. Para estudá-las, seria necessária instrumentação da pilha ou captura de tráfego.

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
