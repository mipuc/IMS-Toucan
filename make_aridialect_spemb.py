import torch
import torchaudio
from librosa.util import find_files
from pathlib import Path
import os
from Preprocessing.AudioPreprocessor import AudioPreprocessor
import soundfile as sf
from speechbrain.pretrained import EncoderClassifier
from numpy import trim_zeros

wav_dir="../data/aridialect/aridialect_wav16000"
wav_files="../data/aridialect/train-text-pac.txt"
device="cpu"
cut_silences=True
speaker_embedding_function_ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",run_opts={"device": str(device)},savedir="Models/speechbrain_speaker_embedding_ecapa")
speaker_embedding_function_xvector = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",run_opts={"device": str(device)},savedir="Models/speechbrain_speaker_embedding_xvector")
wav2mel = torch.jit.load("Models/SpeakerEmbedding/wav2mel.pt")
dvector = torch.jit.load("Models/SpeakerEmbedding/dvector-step250000.pt").to(device).eval()
ap = AudioPreprocessor(input_sr=16000, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=cut_silences)

wav_list=[]
with open(wav_files, encoding="utf8") as f:
    transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            wav_list.append(line.split("|")[0] + ".wav")

print(wav_list)

#spkrs_path = list(os.path.basename(x) for x in Path(wav_dir).iterdir())
#spkrs = list(set(str(x)[0:str(x).find("_")] for x in spkrs_path))
spkrs = list(set(str(x)[0:str(x).find("_")] for x in wav_list))
#spkrs = list(os.path.basename(x) for x in Path(wav_dir).iterdir())
print(spkrs)

for spk in spkrs:
#for spk in ['spo']:
    print(spk)
    audio_paths = [os.path.join(wav_dir,filename) for filename in wav_list if filename.startswith(spk)]
    print(audio_paths)
    combined_spemb = torch.zeros([960])
    print(len(audio_paths))
    for audio_path in audio_paths:
        #read audio to datapoint
        #continue
        print(audio_path)
        try:
            wave, sr = sf.read(audio_path)
            norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
        except ValueError:
            print("ValueError "+audio_path)
            continue
        norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
        norm_wave.cpu().detach().numpy()

        ecapa_spemb = speaker_embedding_function_ecapa.encode_batch(torch.Tensor(norm_wave).to(device)).flatten().detach().cpu()
        xvector_spemb = speaker_embedding_function_xvector.encode_batch(torch.Tensor(norm_wave).to(device)).flatten().detach().cpu()
        dvector_spemb = dvector.embed_utterance(wav2mel(torch.Tensor(norm_wave).unsqueeze(0), 16000).to(device)).flatten().detach().cpu()
        print("Embedding dimensions")
        print(ecapa_spemb.size())
        print(xvector_spemb.size())
        print(dvector_spemb.size())
        combi = torch.cat([ecapa_spemb, xvector_spemb, dvector_spemb], dim=0)
        #print(combi)
        #print(combi.size())
        combined_spemb = combined_spemb + combi
        #print(combined_spemb)

    combined_spemb_mean = torch.div(combined_spemb,len(audio_paths))
    print(combined_spemb_mean)
    torch.save(combined_spemb_mean, "Models/SpeakerEmbedding/"+spk+".pt")
    #ecapa embedding 192 entries
    torch.save(combined_spemb_mean[0:192-1], "Models/SpeakerEmbedding/"+spk+"_ecapa.pt")
    #xvector embedding 512 entries
    torch.save(combined_spemb_mean[192:192+512-1], "Models/SpeakerEmbedding/"+spk+"_xvector.pt")
    #dvector embedding 256 entries
    torch.save(combined_spemb_mean[192+512:192+512+256-1], "Models/SpeakerEmbedding/"+spk+"_dvector.pt")








