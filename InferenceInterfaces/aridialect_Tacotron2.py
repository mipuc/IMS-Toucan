import os

import librosa.display as lbd
import matplotlib.pyplot as plt
import sounddevice
import soundfile
import torch
import torchaudio
import configparser

from speechbrain.pretrained import EncoderClassifier
from InferenceInterfaces.InferenceArchitectures.InferenceHiFiGAN import HiFiGANGenerator
from InferenceInterfaces.InferenceArchitectures.InferenceTacotron2 import Tacotron2
from Preprocessing.TextFrontend import TextFrontend
from Utility.utils import get_most_recent_checkpoint

class aridialect_Tacotron2(torch.nn.Module):

    def __init__(self, device="cpu", speaker_embedding=None, speaker_embedding_type=None, model_num=None):
        super().__init__()
        configparams =  configparser.ConfigParser(allow_no_value=True)
        configparams.read( os.environ.get('TOUCAN_CONFIG_FILE'))
        self.speaker_embedding = speaker_embedding
        self.speaker_embedding_type = speaker_embedding_type
        self.device = device
        self.spk_embed_dim = None
        self.model_num = model_num
        #print(speaker_embedding_type)
        if speaker_embedding_type != "":
            self.spk_embed_dim = 960
            if isinstance(speaker_embedding, torch.Tensor):
                #ecapa embedding 192 entries
                if self.speaker_embedding_type=="ecapa":
                    self.speaker_embedding = self.speaker_embedding[0:192]
                    self.spk_embed_dim = 192
                #xvector embedding 512 entries
                elif self.speaker_embedding_type=="xvector":
                    self.speaker_embedding = self.speaker_embedding[192:192+512]
                    self.spk_embed_dim = 512
                #dvector embedding 256 entries
                elif self.speaker_embedding_type=="dvector":
                    self.speaker_embedding = self.speaker_embedding[192+512:192+512+256]
                    self.spk_embed_dim = 256
                else:
                    self.speaker_embedding = speaker_embedding
            else:
                self.speaker_embedding = torch.load(os.path.join("Models", "SpeakerEmbedding", speaker_embedding), map_location='cpu').to(torch.device(device)).squeeze(0).squeeze(0)

        #self.text2phone = TextFrontend(language="at-lab", use_word_boundaries=False,
        #                               use_explicit_eos=False, inference=True)
        self.text2phone = TextFrontend(language="at", use_word_boundaries=False,
                                       use_explicit_eos=False, inference=True)
        #self.phone2mel = Tacotron2(path_to_weights=os.path.join("Models", "Tacotron2_aridialect_"+str(self.speaker_embedding_type), "best.pt"), idim=166, odim=80, spk_embed_dim=self.spk_embed_dim, reduction_factor=1).to(torch.device(device))
        #modelname = os.path.join("Models",str(self.model_num)+".pt")
        modelname = self.model_num
        print(modelname)
        self.phone2mel = Tacotron2(path_to_weights=modelname, idim=166, odim=80, spk_embed_dim=self.spk_embed_dim, reduction_factor=1).to(torch.device(device))

        print("Hifimodel: "+configparams["INF"]["hifi_model_dir"])
        model_num = None
        if configparams["INF"]["hifi_model"] != "":
            model_num = model
        else:
            model_num = get_most_recent_checkpoint(configparams["INF"]["hifi_model_dir"])
        print(model_num)

        #self.mel2wav = HiFiGANGenerator(path_to_weights=os.path.join("Models", "HiFiGAN_aridialect", "best.pt")).to(torch.device(device))
        self.mel2wav = HiFiGANGenerator(path_to_weights=model_num).to(torch.device(device))
        self.phone2mel.eval()
        self.mel2wav.eval()
        self.to(torch.device(device))

    def forward(self,view, path_to_wavfile):
        with torch.no_grad():
            #get spk_id from wavefile and compute speaker embedding
            #speaker_embedding_function_ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",run_opts={"device": str(self.device)},savedir="Models/speechbrain_speaker_embedding_ecapa")
            #speaker_embedding_function_xvector = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",run_opts={"device": str(self.device)},savedir="Models/speechbrain_speaker_embedding_xvector")

            #wav2mel = torch.jit.load("Models/SpeakerEmbedding/wav2mel.pt")
            #dvector = torch.jit.load("Models/SpeakerEmbedding/dvector-step250000.pt").to(self.device).eval()

            #datapoint, sample_rate = torchaudio.load(path_to_wavfile)

            #ecapa_spemb = speaker_embedding_function_ecapa.encode_batch(torch.Tensor(datapoint).to(self.device)).flatten().detach().cpu()
            #xvector_spemb = speaker_embedding_function_xvector.encode_batch(torch.Tensor(datapoint).to(self.device)).flatten().detach().cpu()
            #dvector_spemb = dvector.embed_utterance(wav2mel(torch.Tensor(datapoint), 16000).to(self.device)).flatten().detach().cpu()
            #combined_spemb = torch.cat([ecapa_spemb, xvector_spemb, dvector_spemb], dim=0)

            #torch.save(cached_speaker_embedding, "Models/SpeakerEmbedding/aridialect_embedding.pt")
            #self.speaker_embedding = torch.load(os.path.join("Models", "SpeakerEmbedding", "aridialect_embedding.pt"), map_location='cpu').to(torch.device("cpu")).squeeze(0).squeeze(0)

            #filename = os.path.basename(path_to_wavfile)
            #spkname = filename[0:filename.find("_")]
            #spkembedfile = "Models/SpeakerEmbedding/" + spkname + ".pt"
            #print(filename)
            #print(spkname)
            #print(spkembedfile)
            #combined_spemb = torch.load(spkembedfile)
            #self.speaker_embedding = combined_spemb.to(self.device)

            #self.speaker_embedding = combined_spemb.to(self.device)

            phones = self.text2phone.string_to_tensor(text=path_to_wavfile,view=False,path_to_wavfile=path_to_wavfile).squeeze(0).long().to(torch.device(self.device))
            mel = self.phone2mel(phones, speaker_embedding=self.speaker_embedding).transpose(0, 1)
            wave = self.mel2wav(mel)
        if view:
            fig, ax = plt.subplots(nrows=2, ncols=1)
            ax[0].plot(wave.cpu().numpy())
            lbd.specshow(mel.cpu().numpy(), ax=ax[1], sr=16000, cmap='GnBu', y_axis='mel', x_axis='time', hop_length=256)
            ax[0].set_title(self.text2phone.get_phone_string(text="",view=False, path_to_wavfile=path_to_wavfile))
            ax[0].yaxis.set_visible(False)
            ax[1].yaxis.set_visible(False)
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.9, wspace=0.0, hspace=0.0)
            plt.show()

        return wave


    def read_to_file(self, wav_list, file_location, silent=True):
        """
        :param silent: Whether to be verbose about the process
        :param text_list: A list of strings to be read
        :param file_location: The path and name of the file it should be saved to
        """
        wav = None
        silence = torch.zeros([24000])
        i=0
        for wavname in wav_list:
            if wavname.strip() != "":
                if not silent:
                    print("Now synthesizing: {}".format(wavname))
                if wav is None:
                    wav = self(False, wavname).cpu()
                    wav = torch.cat((wav, silence), 0)
                else:
                    wav = torch.cat((wav, self("",False,wavname).cpu()), 0)
                    wav = torch.cat((wav, silence), 0)
                i=i+1
        soundfile.write(file=file_location, data=wav.cpu().numpy(), samplerate=48000)

    def read_aloud(self, wavname, view=False, blocking=False):
        if wavname.strip() == "":
            return
        wav = self(False,wavname).cpu()
        #wav = self(text, view).cpu()
        wav = torch.cat((wav, torch.zeros([24000])), 0)
        if not blocking:
            sounddevice.play(wav.numpy(), samplerate=48000)
        else:
            sounddevice.play(torch.cat((wav, torch.zeros([12000])), 0).numpy(), samplerate=48000)
            sounddevice.wait()

    def plot_attention(self, wavname):
        sentence_tensor = self.text2phone.string_to_tensor("",False,path_to_wavfile=wavname).squeeze(0).long().to(torch.device(self.device))
        att = self.phone2mel(text=sentence_tensor, speaker_embedding=self.speaker_embedding, return_atts=True)
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.imshow(att.detach().numpy(), interpolation='nearest', aspect='auto', origin="lower")
        axes.set_title("{}".format(sentence))
        axes.xaxis.set_visible(False)
        axes.yaxis.set_visible(False)
        plt.tight_layout()
        plt.show()

    def save_embedding_table(self):
        import json
        phone_to_embedding = dict()
        for phone in self.text2phone.ipa_to_vector:
            if phone in ['?', 'ɚ', 'p', 'u', 'ɹ', 'ɾ', 'ʔ', 'j', 'l', 'ɔ', 'v', 'm', '~', 'ᵻ', 'ɪ', 'ʒ', 'æ', 'n', 'z', 'ŋ', 'i', 'b', 'o', 'ɛ', 'e', 't', '!',
                         'ʊ', 'ð', 'd', 'θ',
                         'ɑ', 'ɡ', 's', 'ɐ', 'k', 'w', 'ə', 'ʌ', 'ʃ', '.', 'a', 'ɜ', 'h', 'f']:
                print(phone)
                phone_to_embedding[phone] = self.phone2mel.enc.embed(torch.LongTensor([self.text2phone.ipa_to_vector[phone]])).detach().numpy().tolist()
        with open("embedding_table_512dim.json", 'w', encoding="utf8") as fp:
            json.dump(phone_to_embedding, fp)
