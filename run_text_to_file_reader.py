import os

import torch

from InferenceInterfaces.LJSpeech_FastSpeech2 import LJSpeech_FastSpeech2
from InferenceInterfaces.LJSpeech_Tacotron2 import LJSpeech_Tacotron2
from InferenceInterfaces.LibriTTS_FastSpeech2 import LibriTTS_FastSpeech2
from InferenceInterfaces.LibriTTS_Tacotron2 import LibriTTS_Tacotron2
from InferenceInterfaces.MultiEnglish_Tacotron2 import MultiEnglish_Tacotron2
from InferenceInterfaces.Nancy_FastSpeech2 import Nancy_FastSpeech2
from InferenceInterfaces.Nancy_Tacotron2 import Nancy_Tacotron2
from InferenceInterfaces.Thorsten_FastSpeech2 import Thorsten_FastSpeech2
from InferenceInterfaces.Thorsten_Tacotron2 import Thorsten_Tacotron2
from InferenceInterfaces.aridialect_Tacotron2 import aridialect_Tacotron2

tts_dict = {
    "fast_thorsten": Thorsten_FastSpeech2,
    "fast_lj"      : LJSpeech_FastSpeech2,
    "fast_libri"   : LibriTTS_FastSpeech2,
    "fast_nancy"   : Nancy_FastSpeech2,

    "taco_thorsten": Thorsten_Tacotron2,
    "taco_lj"      : LJSpeech_Tacotron2,
    "taco_libri"   : LibriTTS_Tacotron2,
    "taco_nancy"   : Nancy_Tacotron2,
    "taco_aridialect"   : aridialect_Tacotron2,

    "taco_multi"   : MultiEnglish_Tacotron2
    }


def read_texts(model_id, sentence, filename, device="cpu", speaker_embedding=None):
    tts = tts_dict[model_id](device=device, speaker_embedding=speaker_embedding)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts

def read_aridialect_sentences(model_id, device):
    tts = tts_dict[model_id](device=device, speaker_embedding="default_speaker_embedding.pt")

    path="/users/michael.pucher"
    with open(os.path.join(path, "data/aridialect/test-text.txt"), "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/test_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        print(sent)
        sentid= sent.split("|")[0]
        wavname=os.path.join(path, "data/aridialect/aridialect_wav16000", sentid+".wav")
        tts.read_to_file(wav_list=[wavname], file_location=output_dir + "/{}.wav".format(sentid))


def read_harvard_sentences(model_id, device):
    tts = tts_dict[model_id](device=device, speaker_embedding="default_speaker_embedding.pt")

    with open("Utility/test_sentences_combined_3.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_03_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))

    with open("Utility/test_sentences_combined_6.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_06_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    #read_harvard_sentences(model_id="fast_lj", device=exec_device)

    read_aridialect_sentences(model_id="taco_aridialect", device=exec_device)
