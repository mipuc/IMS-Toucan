import os

import torch
import argparse

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

def read_aridialect_sentences(model_id, device, spklist=None, alpha="0.5", speaker_embedding_type=None):

    print(speaker_embedding_type)
    path="/users/michael.pucher"
    with open(os.path.join(path, "data/aridialect/test-text-onesent.txt"), "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/test_{}_{}".format(model_id,speaker_embedding_type)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        print(sent)
        sentid= sent.split("|")[0]
        wavname=os.path.join(path, "data/aridialect/aridialect_wav16000", sentid+".wav")
        if spklist is None:
            filename = os.path.basename(wavname)
            spkname = filename[0:filename.find("_")] 
            spkembedfile = "Models/SpeakerEmbedding/" + spkname + ".pt"
            print(filename)
            print(spkname)
            #print(spkembedfile)
            combined_spemb = torch.load(spkembedfile)
            #self.speaker_embedding = combined_spemb.to(self.device)
            tts = tts_dict[model_id](device=device, speaker_embedding=combined_spemb, speaker_embedding_type=speaker_embedding_type)
            tts.read_to_file(wav_list=[wavname], file_location=output_dir + "/{}.wav".format(sentid))
        else:
            with open(spklist, "r", encoding="utf8") as f:
                spkrs = f.readlines()
            print(spkrs)
            for i, spk1 in enumerate(spkrs):
                for j, spk2 in enumerate(spkrs):
                    if j>i:
                        spkembedfile1 = "Models/SpeakerEmbedding/" + spk1.strip() + ".pt"
                        spkembedfile2 = "Models/SpeakerEmbedding/" + spk2.strip() + ".pt"
                        print(spk1.strip())
                        print(spk2.strip())
                        #print(spkembedfile)
                        combined_spemb1 = torch.load(spkembedfile1)
                        combined_spemb2 = torch.load(spkembedfile2)
                        for a in alpha.split(" "):
                            combined_spemb = torch.add(torch.mul(combined_spemb1,float(a)), torch.mul(combined_spemb2,(1-float(a))))
                            combined_spemb_ = combined_spemb.to(device)
                            #self.speaker_embedding = combined_spemb.to(self.device)
                            tts = tts_dict[model_id](device=device, speaker_embedding=combined_spemb_, speaker_embedding_type=speaker_embedding_type)
                            ipolname = "_"+spk1.strip()+"_"+spk2.strip()+"_"+str(a)
                            tts.read_to_file(wav_list=[wavname], file_location=output_dir + "/{}.wav".format(sentid+ipolname))    
        


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

    parser = argparse.ArgumentParser(description='Synthesize/interpolate and read to file')
    parser.add_argument('--spklist',
                        type=str,
                        help="List of speakers in a file to interpolate.",
                        default=None)
    parser.add_argument('--alpha',
                        type=str,
                        help="Interpolation factor as a string separated by blanks.",
                        default="0.5")
    parser.add_argument('--speaker_embedding_type',
                        type=str,
                        help="combined, ecapa, xvector, or dvector",
                        default="combined")

    args = parser.parse_args()

    read_aridialect_sentences(model_id="taco_aridialect", device=exec_device, spklist=args.spklist, alpha=args.alpha, speaker_embedding_type=args.speaker_embedding_type)
    #read_harvard_sentences(model_id="fast_lj", device=exec_device)
