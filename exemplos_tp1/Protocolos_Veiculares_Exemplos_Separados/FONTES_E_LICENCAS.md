# Fontes primárias e licenças

Consultadas em 06/10/2026.

[1] CiA — história do CAN
https://www.can-cia.org/can-knowledge/history-of-can-technology

[2] Bosch — protocolos CAN
https://www.bosch-semiconductors.com/products/ip-modules/can-protocols/

[3] Bosch — CAN FD
https://www.bosch-semiconductors.com/products/ip-modules/can-protocols/can-fd/

[4] python-can — VirtualBus
https://python-can.readthedocs.io/en/stable/interfaces/virtual.html

[5] python-can — Message
https://python-can.readthedocs.io/en/stable/message.html

[6] python-can — Bus e filtros
https://python-can.readthedocs.io/en/stable/bus.html

[7] Python — multiprocessing
https://docs.python.org/3/library/multiprocessing.html

[8] Python — socket
https://docs.python.org/3/library/socket.html

[9] Python — relógios
https://docs.python.org/3/library/time.html

[10] OpenCV — imencode e imdecode
https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html

[11] IETF — RFC 9293, TCP
https://www.rfc-editor.org/rfc/rfc9293.html

[12] IETF — RFC 768, UDP
https://www.rfc-editor.org/info/rfc768/

[13] IETF — RFC 8085, uso de UDP
https://www.rfc-editor.org/rfc/rfc8085.html

[14] IEEE — 100BASE-T1, 802.3bw
https://www.ieee802.org/3/bw/

[15] IEEE — 1000BASE-T1, 802.3bp
https://www.ieee802.org/3/bp/

[16] IEEE — perfil automotivo TSN, 802.1DG-2025
https://1.ieee802.org/publication-ieee-802-1dg-2025/

[17] Udacity — vídeo de direção e projeto LaneLines
https://github.com/udacity/CarND-LaneLines-P1

[18] Python — struct e ordem de bytes
https://docs.python.org/3/library/struct.html

[19] Texas Instruments — fundamentos e arbitragem CAN
https://www.ti.com/lit/an/sloa101b/sloa101b.pdf

## Mídia incluída
Vídeo: Udacity, CarND-LaneLines-P1, test_videos/solidWhiteRight.mp4.
Origem: https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_videos/solidWhiteRight.mp4
Licença MIT preservada em dados/LICENCA_UDACITY.txt.
Uso: vídeo real de direção para simular uma fonte de câmera. Resolução original
960×540, 25 FPS, 221 frames. O produtor redimensiona para 640×480 e repete o
arquivo para transmitir 320 frames; não são 320 cenas únicas.

## Autoria e resultados
O material didático, diagramas e código do laboratório são preparados para
aula; sinais CAN, modelos de perda e exemplos de offset são artificiais e
identificados. CSVs de timestamps e métricas provêm de execuções reais do
software em loopback. Não foram testados controlador CAN, LAN física ou TSN.
