import os
import random
import time

import soundfile as sf
import torch
from numpy import trim_zeros
from speechbrain.pretrained import EncoderClassifier
from torch.multiprocessing import Manager
from torch.multiprocessing import Process
from torch.utils.data import Dataset
from tqdm import tqdm
from unsilence import Unsilence

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import TextFrontend


class TacotronDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 cache_dir,
                 lang,
                 speaker_embedding=False,
                 speaker_embedding_type="combined",
                 loading_processes=8,
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 cut_silences=True,
                 remove_all_silences=True,
                 rebuild_cache=False,
                 device="cpu"):
        self.speaker_embedding = speaker_embedding
        self.speaker_embedding_type = speaker_embedding_type
        self.taco_train_cache_name = "taco_train_cache_"+speaker_embedding_type+".pt"
        if remove_all_silences:
            os.makedirs(os.path.join(cache_dir, "unsilenced_audios"), exist_ok=True)
            os.makedirs(os.path.join(cache_dir, "normalized_unsilenced_audios"), exist_ok=True)
            os.makedirs(os.path.join(cache_dir, "normalized_audios"), exist_ok=True)
        if not os.path.exists(os.path.join(cache_dir, self.taco_train_cache_name)) or rebuild_cache:
            resource_manager = Manager()
            self.path_to_transcript_dict = resource_manager.dict(path_to_transcript_dict)
            key_list = list(self.path_to_transcript_dict.keys())
            random.shuffle(key_list)
            # build cache
            print("... building dataset cache ...")
            self.datapoints = resource_manager.list()
            # make processes
            key_splits = list()
            process_list = list()
            for i in range(loading_processes):
                key_splits.append(key_list[i * len(key_list) // loading_processes:(i + 1) * len(key_list) // loading_processes])
            for key_split in key_splits:
                process_list.append(
                    Process(target=self.cache_builder_process,
                            args=(key_split,
                                  speaker_embedding,
                                  lang,
                                  min_len_in_seconds,
                                  max_len_in_seconds,
                                  cut_silences,
                                  remove_all_silences,
                                  cache_dir),
                            daemon=True))
                process_list[-1].start()
                time.sleep(5)
            for process in process_list:
                process.join()
            self.datapoints = list(self.datapoints)
            tensored_datapoints = list()
            # we had to turn all of the tensors to numpy arrays to avoid shared memory
            # issues. Now that the multi-processing is over, we can convert them back
            # to tensors to save on conversions in the future.
            print("Converting into convenient format...")
            norm_waves = list()
            if self.speaker_embedding:
                if speaker_embedding:
                    #speaker_embedding_function_ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                    #                                                                  run_opts={"device": str(device)},
                    #                                                                  savedir="Models/speechbrain_speaker_embedding_ecapa")
                    #speaker_embedding_function_xvector = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                    #                                                                    run_opts={"device": str(device)},
                    #                                                                    savedir="Models/speechbrain_speaker_embedding_xvector")
                    #wav2mel = torch.jit.load("Models/SpeakerEmbedding/wav2mel.pt")
                    #dvector = torch.jit.load("Models/SpeakerEmbedding/dvector-step250000.pt").to(device).eval()
                    # everything assumes 16kHz audio as input here
                    for datapoint in tqdm(self.datapoints):
                        #print(datapoint[6])
                        #print(torch.Tensor(datapoint[5]).size())
                        if not torch.Tensor(datapoint[5]).size(dim=0)>0:
                            print(f"Excluding {datapoint[6]} because of its duration of 0 after unsilencing.")
                            continue
                        #ecapa_spemb = speaker_embedding_function_ecapa.encode_batch(torch.Tensor(datapoint[4]).to(device)).flatten().detach().cpu()
                        #xvector_spemb = speaker_embedding_function_xvector.encode_batch(torch.Tensor(datapoint[4]).to(device)).flatten().detach().cpu()
                        #dvector_spemb = dvector.embed_utterance(wav2mel(torch.Tensor(datapoint[4]).unsqueeze(0), 16000).to(device)).flatten().detach().cpu()
                        #combined_spemb = torch.cat([ecapa_spemb, xvector_spemb, dvector_spemb], dim=0)
                        #get speaker name from path in datapoint[6]
                        filename = os.path.basename(datapoint[6])
                        spkname = filename[0:filename.find("_")] 
                        spkembedfile = "Models/SpeakerEmbedding/" + spkname + ".pt"
                        #print(filename)
                        #print(spkname)
                        print(spkembedfile)
                        combined_spemb = torch.load(spkembedfile)

                        use_spkemb = combined_spemb
                        #ecapa embedding 192 entries
                        if self.speaker_embedding_type=="ecapa":
                            use_spkemb = combined_spemb[0:192]
                         #xvector embedding 512 entries
                        elif self.speaker_embedding_type=="xvector":
                            use_spkemb = combined_spemb[192:192+512]
                        #dvector embedding 256 entries
                        elif self.speaker_embedding_type=="dvector":
                            use_spkemb = combined_spemb[192+512:192+512+256]

                        print(self.speaker_embedding_type)
                        print(use_spkemb.size())

                        tensored_datapoints.append([torch.LongTensor(datapoint[0]),
                                                    torch.LongTensor(datapoint[1]),
                                                    torch.Tensor(datapoint[2]),
                                                    torch.LongTensor(datapoint[3]),
                                                    use_spkemb])
                        norm_waves.append(torch.Tensor(datapoint[-2]))
                    #del speaker_embedding_function_ecapa
                    #del speaker_embedding_function_xvector
                    #del wav2mel
                    #del dvector
                # loading the speaker embedding function messes up something in torchaudios resample layer.
                # So you possibly need to restart the program after every dataset creation if you are using
                # multiple datasets together.
            else:
                for datapoint in tqdm(self.datapoints):
                    if not torch.Tensor(datapoint[4]).size(dim=0)>0:
                        print(f"Excluding {datapoint[5]} because of its duration of 0 after unsilencing.")
                        continue
                    tensored_datapoints.append([torch.LongTensor(datapoint[0]),
                                                torch.LongTensor(datapoint[1]),
                                                torch.Tensor(datapoint[2]),
                                                torch.LongTensor(datapoint[3])])
                    norm_waves.append(torch.Tensor(datapoint[-2]))

            self.datapoints = tensored_datapoints
            # save to cache
            torch.save((self.datapoints, norm_waves), os.path.join(cache_dir, self.taco_train_cache_name))
        else:
            # just load the datapoints from cache
            self.datapoints = torch.load(os.path.join(cache_dir, self.taco_train_cache_name), map_location='cpu')

            if isinstance(self.datapoints, tuple):  # check for backwards compatibility
                self.datapoints = self.datapoints[0]
        print("Prepared {} datapoints.".format(len(self.datapoints)))

    def cache_builder_process(self, path_list, speaker_embedding, lang, min_len, max_len, cut_silences, remove_all_silences, cache_dir):
        process_internal_dataset_chunk = list()
        tf = TextFrontend(language=lang)
        _, sr = sf.read(path_list[0])
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=cut_silences)
        if remove_all_silences:
            ap = AudioPreprocessor(input_sr=sr, output_sr=None, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=cut_silences)
            # the unsilence tool unfortunately writes files with a sample rate that we cannot control, so we need special cases
            ap_post = None
        i=0
        for path in tqdm(path_list):
            if remove_all_silences:
                name = path.split("/")[-1].split(".")[:-1]
                if len(name) == 1:
                    name = name[0]
                else:
                    name = ".".join(name)
                suffix = path.split(".")[-1]
                _norm_unsilenced_path = os.path.join(os.path.join(cache_dir, "normalized_unsilenced_audios"), name + "_unsilenced." + suffix)
                _norm_path = os.path.join(os.path.join(cache_dir, "normalized_audios"), name + "." + suffix)
                if not os.path.exists(_norm_unsilenced_path):
                    if not os.path.exists(_norm_path):
                        wave, sr = sf.read(path)
                        print(path)
                        dur_in_seconds = len(wave) / sr
                        if not (min_len <= dur_in_seconds <= max_len):
                            print(f"Excluding {_norm_unsilenced_path} because of its duration of {round(dur_in_seconds, 2)} seconds.")
                            continue
                        try:
                            norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
                        except ValueError:
                            print(f"ValueError {_norm_unsilenced_path} in normalization.")
                            continue
                        dur_in_seconds = len(norm_wave) / 16000
                        if not (min_len <= dur_in_seconds <= max_len):
                            print(f"Excluding {_norm_unsilenced_path} because of its duration of {round(dur_in_seconds, 2)} seconds.")
                            continue
                        print(_norm_path)
                        sf.write(file=_norm_path, data=norm_wave.detach().numpy(), samplerate=sr)
                    try:
                        unsilence = Unsilence(_norm_path)
                        unsilence.detect_silence(silence_time_threshold=0.5, short_interval_threshold=0.03, stretch_time=0.025)
                        unsilence.render_media(_norm_unsilenced_path, silent_speed=12, silent_volume=0, audio_only=True)
                    except BaseException as error:
                        print(f'An exception occurred in unsilencing: {error}')
                        continue
                try:
                    wave, sr = sf.read(_norm_unsilenced_path)
                    if ap_post is None:
                        ap_post = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=cut_silences)
                    if sr != ap_post.sr:
                        print(f"Inconsistent sample rate! {_norm_unsilenced_path}")
                        continue
                    norm_wave = ap_post.resample(torch.Tensor(wave))
                    dur_in_seconds = len(norm_wave) / 16000
                    if not (min_len <= dur_in_seconds <= max_len):
                        print(f"Excluding {_norm_unsilenced_path} because of its duration of {round(dur_in_seconds, 2)} seconds.")
                        continue
                except RuntimeError:
                    # not sure why this sometimes happens, but it is very rare, so it should be fine.
                    print("RuntimeError")
                    continue
            else:
                wave, sr = sf.read(path)
                dur_in_seconds = len(wave) / sr
                if not (min_len <= dur_in_seconds <= max_len):
                    print(f"Excluding {path} because of its duration of {round(dur_in_seconds, 2)} seconds.")
                    continue
                if sr != ap.sr:
                    print(f"Inconsistent sampling rate in the Data! Excluding {path}")
                    continue
                try:
                    norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
                except ValueError:
                    print(f"ValueError {_norm_unsilenced_path} in normalization.")
                    continue
                dur_in_seconds = len(norm_wave) / 16000
                if not (min_len <= dur_in_seconds <= max_len):
                    print(f"Excluding {path} because of its duration of {round(dur_in_seconds, 2)} seconds.")
                    continue
                _norm_unsilenced_path = path

            norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
            # raw audio preprocessing is done
            transcript = self.path_to_transcript_dict[path]
            cached_text = tf.string_to_tensor(transcript,path_to_wavfile=path).squeeze(0).cpu().numpy()
            cached_text_len = torch.LongTensor([len(cached_text)]).numpy()
            cached_speech = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False).transpose(0, 1).cpu().numpy()
            cached_speech_len = torch.LongTensor([len(cached_speech)]).numpy()
            if speaker_embedding:
                process_internal_dataset_chunk.append([cached_text,
                                                       cached_text_len,
                                                       cached_speech,
                                                       cached_speech_len,
                                                       norm_wave.numpy(),
                                                       norm_wave.cpu().detach().numpy(),
                                                       path])
            else:
                process_internal_dataset_chunk.append([cached_text,
                                                       cached_text_len,
                                                       cached_speech,
                                                       cached_speech_len,
                                                       norm_wave.cpu().detach().numpy(),
                                                       path])
        
        #print(len(self.datapoints))
        #print(len(process_internal_dataset_chunk))
        self.datapoints += process_internal_dataset_chunk
        i=i+1

    def __getitem__(self, index):
        if not self.speaker_embedding:
            return self.datapoints[index][0], \
                   self.datapoints[index][1], \
                   self.datapoints[index][2], \
                   self.datapoints[index][3]
        else:
            return self.datapoints[index][0], \
                   self.datapoints[index][1], \
                   self.datapoints[index][2], \
                   self.datapoints[index][3], \
                   self.datapoints[index][4]

    def __len__(self):
        return len(self.datapoints)
