
A video tutorial on how to set up and customize is in progress

link to Mekatron server - https://discord.gg/S9dmsWBTha

my discord server for feed back - https://discord.gg/4Px628yhxZ

## requirements

- pyhton 3.7

- about 10 gb free space

- anaconde installed

- gpu that supports CUDNN

## setup

1. create a new environment in anaconda on python 3.7

2. run console and redirect in console to the folder where you have all the code

3. type in console "pip install -r requirements.txt"

4. install [pytorch](https://pytorch.org)

5. install [voicemeeter](https://vb-audio.com/Voicemeeter/banana.htm) - [tutotrial how this work](https://www.youtube.com/watch?v=OeB7UVlPfu0)

6. run id_check

7. type the id of the microphone you will use to capture your speech and the id of the VoiceMeeter Input - on this device we will play the sound. type the ids where the picture shows:

example for id list:

![Desktop Screenshot 2021 07 30 - 14 51 47 26 (2)](https://user-images.githubusercontent.com/73255858/127658741-4c5351d8-006a-4fc0-ad08-211c99322c14.png)

in 'speech to text' file:

![Desktop Screenshot 2021 07 30 - 15 10 28 43 (2)](https://user-images.githubusercontent.com/73255858/127657943-61c363f8-9be4-4990-a7e4-14a909051a15.png)

in 'main' file:

![Desktop Screenshot 2021 07 30 - 15 10 20 62 (2)](https://user-images.githubusercontent.com/73255858/127657947-b1a385dd-c9de-4562-8d8f-9a3317af4f94.png)

8. now you can run 'main.py' and see if everything works

## customizing

1. if you want to change the voice then you need to have a model of that voice and swap it with the TTS file. Once you have done this the voice should change. If you are using a female voice you need to change the waveglow model then the voice should sound better. To change the waveglow model to female, go to line 89 and change the code in the "waveglow_link" variable to the one in line 85. If you changed it, delete the 'waveglow.pt' file and then the new model should download
2. if you want to change the language in speech recognition to Anglican, remove the phrase 'language='pl-PL' in line 12 of the 'speech to text.py' file

## about the project

This project was born in my head when I saw the mekatron website and I thought that it would be cool to be able to speak with the voices of various game characters and not only, so I went on the mekatron discord server and I saw that there are tutorials on how to train models for voice and how to use a speech synthesizer. So I copied the code for the speech synthesizer (all the google colab files are below which I based the code on) and changed it very slightly to make the code more efficient I think and added speech to text and playback of audio files through the microphone.

everything works in 4 seconds delay on my computer with ninvida gtx 960 graphics card and no game running so i'm not sure if it works in less time delay.

If you have any ideas how i can improve the code of 'main.py' and ' speech to text.py' files text me on discord what i should improve and how.

## to-do list

- better documantation and comments
- clean out kod
- better speech to text system
- optimize code
- graphical interface

## resources

- voice synthesis google colab - https://colab.research.google.com/drive/1CJyEC0eQ558DhFSpnIIQ6RlGrqzyg22-?usp=sharing
- trenowanie modelu google colab - https://colab.research.google.com/drive/14BVHQzV2wpenoPDx-FKO8siY66_zp9AB?usp=sharing
