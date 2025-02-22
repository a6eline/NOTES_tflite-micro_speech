# Documentation for TensorFlow LiteRT Micro for Arduino

This repository is for me to write personal notes/documentation on the [LiteRT](https://ai.google.dev/edge/litert/inference) Arduino's micro_speech example and other files which are required for it. The intention of this is to study the TensorFlow API so I can impement it within AI projects in the future. I am uploading this so that both me and my project partner can understand further how to implement the API onto our [Alarm Clock Project](https://github.com/a6eline/ML_arduino_alarm) using the Nano 33 BLE Sense Rev2 board. Hopefully the notes will be of some use for others who want to use speech recognition within microcotrollers with TensorFlow Lite Micro.

> Status: Ongoing

## Other Sources
This is forked from the original [tflite-micro-arduino-examples](https://github.com/tensorflow/tflite-micro-arduino-examples) GitHub repository. Installation info is all there. 

The [TensorFlow Lite Micro Documentation](https://ai.google.dev/edge/litert/microcontrollers/get_started) will say it needs a C++17 enviroment, but you dont need to adjust the Nano 33 BLE board package's platform.txt file for it manually with -std=gnu++17 etc. Theres no need to install a tool chain for it either, this should work even if the board states it is C++14.

## My Contribution

### Micro_speech Example
Within the examples folder, I have written as much comments as I can within [micro_speech](https://github.com/a6eline/NOTES_tflite-micro_speech/tree/main/examples/micro_speech) example to better understand each file and its usages. It might seem like overcommenting, but it is to keep track of things, theres a lot of variables and types which I can easily lose track of unfortunately.

![image](https://github.com/user-attachments/assets/cc02fff2-43bf-4de9-9d50-1193a5b13f44)
![image](https://github.com/user-attachments/assets/f1715457-ae6d-41f7-94eb-e8597615af4c)

### Micro_speech Documentation 
I have also written a [word documentation](https://docs.google.com/document/d/1WiQw86Ue8yddEHVPHRZVkMrpHQOxgGgfX3WtC2Yl9dU/edit?tab=t.hnpsgq8m388t) for this so that I can gain better understanding of it. 

![image](https://github.com/user-attachments/assets/4a807bc8-bafc-44bb-bc2a-81c570f3ed14)

