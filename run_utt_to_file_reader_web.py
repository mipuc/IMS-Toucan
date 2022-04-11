import os

import torch
import argparse
import configparser

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
from audiotsm import wsola, phasevocoder
from audiotsm.io.wav import WavReader, WavWriter


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


def read_write_utt(model_id, device, utt, wav, speaker, model_num, speed, speaker_embedding_type):


    #path="/users/michael.pucher"
    #with open(os.path.join(path, "data/aridialect/owe_test.txt"), "r", encoding="utf8") as f:
    #    sents = f.read().split("\n")
    #output_dir = "audios/test_{}_{}".format(model_id,model_num)
    #if not os.path.isdir(output_dir):
    #    os.makedirs(output_dir)
    #print(utt)
    combined_spemb = None
    if speaker_embedding_type != "":
        spkembedfile = "Models/SpeakerEmbedding/" + speaker + ".pt"
        combined_spemb = torch.load(spkembedfile)
        print(spkembedfile)

    tts = tts_dict[model_id](device=device, speaker_embedding=combined_spemb, speaker_embedding_type=speaker_embedding_type, model_num=model_num)
    tts.read_to_file(wav_list=[utt], file_location=wav)
    if speed != 1.0:
        print("New speed is: "+str(speed))
        with WavReader(wav) as reader:
            with WavWriter(wav+".1", reader.channels, reader.samplerate) as writer:
                #tsm = phasevocoder(reader.channels, speed=speed)
                tsm = wsola(reader.channels, speed=speed)
                tsm.run(reader, writer)
        os.system("mv "+wav+".1 "+wav)




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

    parser = argparse.ArgumentParser(description='Synthesize utterance and write to file')
    parser.add_argument('--utt',
                        type=str,
                        help="Utterance string",
                        default="Hallo Welt!")
    parser.add_argument('--wav',
                        type=str,
                        help="Wav file to write, default output.wav.",
                        default="output.wav")
    parser.add_argument('--voice',
                        type=str,
                        help="Voice to use, consists of speaker name and voice, default owe_voice_owe_taco_oweneutral_50052",
                        default="owe_voice_owe_taco_oweneutral_50052")
    parser.add_argument('--speed',
                        type=float,
                        help="speed to use",
                        default=1.0)
    parser.add_argument('--config',
                        type=str,
                        help="python config file",
                        default="traininf/web.ini")

    args = parser.parse_args()

    os.environ['TOUCAN_CONFIG_FILE'] = args.config
    configparams =  configparser.ConfigParser(allow_no_value=True)
    configparams.read( os.environ.get('TOUCAN_CONFIG_FILE'))
    print(configparams["TRAIN"]["labelfile"])

    #speaker_model = args.voice[10:]
    myspeaker =  args.voice.split("_")[0]
    myspeaker1 =  args.voice.split("_")[2]
    mymodelnum ="_".join(args.voice.split("_")[1:])+".pt"
    print(myspeaker)
    print(myspeaker1)
    print(mymodelnum)

    #voice is adapted
    speaker_embedding_type=""
    if myspeaker != myspeaker1:
        speaker_embedding_type = "combined"
    print(speaker_embedding_type)
    read_write_utt(model_id="taco_aridialect", device=exec_device, utt=args.utt, wav=args.wav, speaker=myspeaker, model_num=os.path.join("voices",mymodelnum), speed=args.speed, speaker_embedding_type=speaker_embedding_type)

    #read_aridialect_sentences(model_id="taco_aridialect", device=exec_device, spklist=args.spklist, alpha=args.alpha, speaker_embedding_type=args.speaker_embedding_type)
    #read_harvard_sentences(model_id="fast_lj", device=exec_device)
