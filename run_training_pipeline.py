import argparse
import sys
import configparser
import os

from TrainingInterfaces.TrainingPipelines.FastSpeech2_LJSpeech import run as fast_LJSpeech
from TrainingInterfaces.TrainingPipelines.FastSpeech2_LibriTTS import run as fast_LibriTTS
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Nancy import run as fast_Nancy
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Thorsten import run as fast_Thorsten
from TrainingInterfaces.TrainingPipelines.HiFiGAN_combined import run as hifigan_combined
from TrainingInterfaces.TrainingPipelines.HiFiGAN_aridialect import run as hifigan_aridialect
from TrainingInterfaces.TrainingPipelines.Tacotron2_Cycle import run as taco_cycle
from TrainingInterfaces.TrainingPipelines.Tacotron2_LJSpeech import run as taco_LJSpeech
from TrainingInterfaces.TrainingPipelines.Tacotron2_LibriTTS import run as taco_LibriTTS
from TrainingInterfaces.TrainingPipelines.Tacotron2_MultiEnglish import run as taco_multi
from TrainingInterfaces.TrainingPipelines.Tacotron2_Nancy import run as taco_Nancy
from TrainingInterfaces.TrainingPipelines.Tacotron2_Thorsten import run as taco_Thorsten
from TrainingInterfaces.TrainingPipelines.Tacotron2_aridialect import run as taco_aridialect


pipeline_dict = {
    "fast_thorsten": fast_Thorsten,
    "taco_thorsten": taco_Thorsten,

    "fast_libri"   : fast_LibriTTS,
    "taco_libri"   : taco_LibriTTS,
    "taco_aridialect"   : taco_aridialect,

    "fast_lj"      : fast_LJSpeech,
    "taco_lj"      : taco_LJSpeech,

    "fast_nancy"   : fast_Nancy,
    "taco_nancy"   : taco_Nancy,

    "hifi_combined": hifigan_combined,
    "hifi_aridialect": hifigan_aridialect,

    "taco_multi"   : taco_multi,
    "taco_cycle"   : taco_cycle
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IMS Speech Synthesis Toolkit - Call to Train')

    parser.add_argument('--config',
                        type=str,
                        help="python config file")

    args = parser.parse_args()

    os.environ['TOUCAN_CONFIG_FILE'] = args.config
    configparams =  configparser.ConfigParser(allow_no_value=True)
    configparams.read( os.environ.get('TOUCAN_CONFIG_FILE'))
    print(configparams["TRAIN"]["labelfile"])
    print(configparams["TRAIN"]["wavdir"])

    if configparams["TRAIN"].getboolean("finetune") and configparams["TRAIN"]["resume_checkpoint"] == "":
        print("Need to provide path to checkpoint to fine-tune from!")
        sys.exit()

    if configparams["TRAIN"].getboolean("finetune") and "hifigan" in configparams["TRAIN"]["pipeline"]:
        print("Fine-tuning for HiFiGAN is not implemented as it didn't seem necessary. Should generalize across speakers without fine-tuning.")
        sys.exit()

    if configparams["TRAIN"]["pipeline"]=="hifi_combined":
        pipeline_dict[configparams["TRAIN"]["pipeline"]](gpu_id=configparams["TRAIN"]["gpu_id"],
                                 resume_checkpoint=configparams["TRAIN"]["resume_checkpoint"],
                                 finetune=configparams["TRAIN"].getboolean("finetune"),
                                 model_dir=configparams["TRAIN"]["model_save_dir"])
    else:
        pipeline_dict[configparams["TRAIN"]["pipeline"]](gpu_id=configparams["TRAIN"]["gpu_id"],
                                 resume_checkpoint=configparams["TRAIN"]["resume_checkpoint"],
                                 resume=configparams["TRAIN"].getboolean("resume"),
                                 finetune=configparams["TRAIN"].getboolean("finetune"),
                                 model_dir=configparams["TRAIN"]["model_save_dir"],
                                 speaker_embedding_type=configparams["TRAIN"]["speaker_embedding_type"])


