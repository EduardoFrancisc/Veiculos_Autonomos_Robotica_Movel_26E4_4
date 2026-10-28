# Servidor e cliente em dois terminais

Instale requirements.txt e use o mesmo ambiente nos dois terminais.
Os serviços são extras; os exemplos 23–36 abrem os processos automaticamente.
Para evitar conflito de arquivos, use uma pasta por protocolo.

## UDP — iniciar o cliente primeiro
Terminal 1 (cliente):
```bash
python servicos/cliente_camera.py --protocolo udp --host 127.0.0.1 --porta 5005 --frames 320 --saida resultados/manual_udp --mostrar
```
Terminal 2 (servidor):
```bash
python servicos/servidor_camera.py --protocolo udp --host 127.0.0.1 --porta 5005 --frames 320 --saida resultados/manual_udp
```
O cliente se vincula à porta e o servidor envia para esse endereço.

## TCP — iniciar o servidor primeiro
Terminal 1 (servidor):
```bash
python servicos/servidor_camera.py --protocolo tcp --host 127.0.0.1 --porta 5006 --frames 320 --saida resultados/manual_tcp
```
Terminal 2 (cliente):
```bash
python servicos/cliente_camera.py --protocolo tcp --host 127.0.0.1 --porta 5006 --frames 320 --saida resultados/manual_tcp --mostrar
```
O TCP aceita uma conexão, envia comprimento+cabeçalho+JPEG e finaliza com FIM1.
Use --camera 0 no servidor para trocar o vídeo por câmera do desktop.

## Recalcular métricas do ensaio manual na mesma máquina
Após os dois programas encerrarem, transmitidos.csv e recebidos.csv estarão na
mesma pasta. No terminal, por exemplo para UDP:
```bash
python -c "import sys,json,pathlib; sys.path.insert(0,'.'); from rede_laboratorio import estatisticas; p=pathlib.Path('resultados/manual_udp'); c=json.loads((p/'servidor_config.json').read_text()); estatisticas(p,c)"
```
Troque manual_udp por manual_tcp para TCP. A janela adiciona carga; para o
benchmark de referência, execute sem --mostrar e mantenha a gravação igual.

## Adaptar para dois computadores em LAN
UDP: cliente --host 0.0.0.0 (bind); servidor --host IP_DO_CLIENTE (destino).
TCP: servidor --host 0.0.0.0 (bind); cliente --host IP_DO_SERVIDOR (destino).
Instale o pacote nos dois hosts e permita a porta escolhida no firewall da LAN.
Use --relogio epoch nos dois. Sem --relogios-sincronizados, o cliente não calcula
latência unidirecional; ele salva os demais dados e marca a latência indisponível.
Só use --relogios-sincronizados depois de validar o método de sincronização e
registrar seu offset/incerteza (por exemplo, infraestrutura PTP apropriada).
NTP, sozinho, não garante precisão submilissegundo em qualquer rede.
Não subtraia clocks monotônicos de computadores diferentes.

Os arquivos ficam em hosts distintos; reúna transmitidos.csv, recebidos.csv,
cliente_contadores.json e, para UDP, pacotes_udp.csv em uma pasta para a análise.
Use o mesmo domínio de clock, vídeo, resolução, JPEG, FPS, buffers e contagem.
As instruções LAN são para repetir o experimento; não houve ensaio físico LAN
na validação de referência distribuída neste pacote.
