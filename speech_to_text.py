import speech_recognition

def recognize_speech():

    recognizer = speech_recognition.Recognizer()

    try:
        with speech_recognition.Microphone(device_index=5) as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=1)
            audio = recognizer.listen(mic)

            text = recognizer.recognize_google(audio, language='pl-PL')
            text = text.lower()
            if text == '':
                return None
            else:
                text = text + '.'
                print('done')
                return text
    except Exception as e:
        print(str(e))