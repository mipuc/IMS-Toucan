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
from Utility.utils import get_most_recent_checkpoint

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


def read_write_utt():

    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    configparams =  configparser.ConfigParser(allow_no_value=True)
    configparams.read( os.environ.get('TOUCAN_CONFIG_FILE'))
    print(configparams["TRAIN"]["labelfile"])

    text = configparams["INF"]["text"]
    output_dir = configparams["INF"]["output_dir"]
    speaker_embedding_type = configparams["TRAIN"]["speaker_embedding_type"]
    speaker = configparams["INF"]["speaker"]
    model = configparams["INF"]["model"]
    model_save_dir = configparams["TRAIN"]["model_save_dir"]
    speed = float(configparams["INF"]["speed"])

    with open(text, "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    sents = [x for x in sents if x != '']

    combined_spemb = None
    if speaker_embedding_type != "":
        spkembedfile = "Models/SpeakerEmbedding/" +  speaker + ".pt"
        combined_spemb = torch.load(spkembedfile)

    model_num = None
    if model != "":
        model_num = model
    else:
        model_num = get_most_recent_checkpoint(model_save_dir)
    print(model_num)
    model_id = os.path.basename(model_num).split('.')[0]

    output_dir = os.path.join(output_dir, model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for index, sent in enumerate(sents):
        print(sent)
        sentid= sent.split("|")[0]
        tts = tts_dict[configparams["TRAIN"]["pipeline"]](device=exec_device, speaker_embedding=combined_spemb, speaker_embedding_type=speaker_embedding_type, model_num=model_num)
        wav = os.path.join(output_dir,sentid+".wav")
        tts.read_to_file(wav_list=[sent.split("|")[1]], file_location=wav)

        if speed != 1.0:
            print("New speed is: "+str(speed))
            with WavReader(wav) as reader:
                with WavWriter(wav+".1", reader.channels, reader.samplerate) as writer:
		    #tsm = phasevocoder(reader.channels, speed=speed)
                    tsm = wsola(reader.channels, speed=speed)
                    tsm.run(reader, writer)
            os.system("mv "+wav+".1 "+wav)



if __name__ == '__main__':
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    parser = argparse.ArgumentParser(description='Synthesize list of utterances and write to directory')
    parser.add_argument('--config',
                        type=str,
                        help="python config file")

    args = parser.parse_args()

    os.environ['TOUCAN_CONFIG_FILE'] = args.config

    read_write_utt()

  
