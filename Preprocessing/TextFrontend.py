# -*- coding: utf-8 -*-

import re
import sys
import os
from collections import defaultdict

import phonemizer
import torch
from cleantext import clean


class TextFrontend:

    def __init__(self,
                 language,
                 use_word_boundaries=False,
                 use_explicit_eos=False,
                 use_prosody=False,  # unfortunately the non-segmental
                 # nature of prosodic markers mixed with the sequential
                 # phonemes hurts the performance of end-to-end models a
                 # lot, even though one might think enriching the input
                 # with such information would help such systems.
                 use_lexical_stress=False,
                 path_to_phoneme_list="Preprocessing/ipa_list.txt",
                 silent=True,
                 allow_unknown=False,
                 inference=False,
                 strip_silence=True,
                 path_to_sampa_mapping_list="Preprocessing/sampa_to_ipa.txt"):
        """
        Mostly preparing ID lookups
        """
        self.strip_silence = strip_silence
        self.use_word_boundaries = use_word_boundaries
        self.allow_unknown = allow_unknown
        self.use_explicit_eos = use_explicit_eos
        self.use_prosody = use_prosody
        self.use_stress = use_lexical_stress
        self.inference = inference
        self.sampa_to_ipa_dict = dict()
        if allow_unknown:
            self.ipa_to_vector = defaultdict()
            self.default_vector = 165
        else:
            self.ipa_to_vector = dict()

        with open(path_to_sampa_mapping_list, "r", encoding='utf8') as f:
            sampa_to_ipa = f.read()
        sampa_to_ipa_list = sampa_to_ipa.split("\n")
        for pair in sampa_to_ipa_list:
            if pair.strip() != "":
                #print(pair)
                self.sampa_to_ipa_dict[pair.split(" ")[0]] = pair.split(" ")[1]

        with open(path_to_phoneme_list, "r", encoding='utf8') as f:
            phonemes = f.read()
            # using https://github.com/espeak-ng/espeak-ng/blob/master/docs/phonemes.md
        phoneme_list = phonemes.split("\n")
        for index in range(1, len(phoneme_list)):
            self.ipa_to_vector[phoneme_list[index]] = index
            # note: Index 0 is unused, so it can be used for padding as is convention.
            #       Index 1 is reserved for end_of_utterance
            #       Index 2 is reserved for begin of sentence token
            #       Index 13 is used for pauses (heuristically)

        # The point of having the phonemes in a separate file is to ensure reproducibility.
        # The line of the phoneme is the ID of the phoneme, so you can have multiple such
        # files and always just supply the one during inference which you used during training.

        if language == "en":
            self.clean_lang = "en"
            self.g2p_lang = "en-us"
            self.expand_abbreviations = english_text_expansion
            if not silent:
                print("Created an English Text-Frontend")

        elif language == "de":
            self.clean_lang = "de"
            self.g2p_lang = "de"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a German Text-Frontend")

        elif language == "el":
            self.clean_lang = None
            self.g2p_lang = "el"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Greek Text-Frontend")

        elif language == "es":
            self.clean_lang = None
            self.g2p_lang = "es"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Spanish Text-Frontend")

        elif language == "fi":
            self.clean_lang = None
            self.g2p_lang = "fi"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Finnish Text-Frontend")

        elif language == "ru":
            self.clean_lang = None
            self.g2p_lang = "ru"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Russian Text-Frontend")

        elif language == "hu":
            self.clean_lang = None
            self.g2p_lang = "hu"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Hungarian Text-Frontend")

        elif language == "nl":
            self.clean_lang = None
            self.g2p_lang = "nl"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Dutch Text-Frontend")

        elif language == "fr":
            self.clean_lang = None
            self.g2p_lang = "fr-fr"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a French Text-Frontend")

        elif language == "at-lab":
            self.clean_lang = None
            self.g2p_lang = "at-lab"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created an Austrian German Text-Frontend")

        else:
            print("Language not supported yet")
            sys.exit()

    def string_to_tensor(self, text, view=False, path_to_wavfile=""):
        """
        Fixes unicode errors, expands some abbreviations,
        turns graphemes into phonemes and then vectorizes
        the sequence as IDs to be fed into an embedding
        layer
        """
        phones = self.get_phone_string(text=text, include_eos_symbol=False, path_to_wavfile=path_to_wavfile)

        if view:
            print("Phonemes: \n{}\n".format(phones))
        phones_vector = list()
        # turn into numeric vectors
        for char in phones:
            if self.allow_unknown:
                phones_vector.append(self.ipa_to_vector.get(char, self.default_vector))
            else:
                try:
                    phones_vector.append(self.ipa_to_vector[char])
                except KeyError:
                    print("unknown phoneme: {}".format(char))
        if self.use_explicit_eos:
            phones_vector.append(self.ipa_to_vector["end_of_input"])
        return torch.LongTensor(phones_vector).unsqueeze(0)


    def get_phone_string(self, text, include_eos_symbol=True, path_to_wavfile=""):
        # clean unicode errors, expand abbreviations, handle emojis etc.
        utt = clean(text, fix_unicode=True, to_ascii=False, lower=False, lang=self.clean_lang)
        self.expand_abbreviations(utt)
        # if an aligner has produced silence tokens before, turn
        # them into silence markers now so that they survive the
        # phonemizer:
        utt = utt.replace("_SIL_", "~")
        # phonemize
        if self.g2p_lang=="at-lab":
            phones = self.phonemize_from_labelfile(text=utt, path_to_wavfile=path_to_wavfile, include_eos_symbol=False)
        else:
            phones = phonemizer.phonemize(utt,
                                          language_switch='remove-flags',
                                          backend="espeak",
                                          language=self.g2p_lang,
                                          preserve_punctuation=True,
                                          strip=True,
                                          punctuation_marks=';:,.!?¡¿—…"«»“”~/',
                                          with_stress=self.use_stress).replace(";", ",").replace("/", " ") \
                .replace(":", ",").replace('"', ",").replace("-", ",").replace("-", ",").replace("\n", " ") \
                .replace("\t", " ").replace("¡", "").replace("¿", "").replace(",", "~")
        phones = re.sub("~+", "~", phones)
        if not self.use_prosody:
            # retain ~ as heuristic pause marker, even though all other symbols are removed with this option.
            # also retain . ? and ! since they can be indicators for the stop token
            phones = phones.replace("ˌ", "").replace("ː", "").replace("ˑ", "") \
                .replace("˘", "").replace("|", "").replace("‖", "")
        if not self.use_word_boundaries:
            phones = phones.replace(" ", "")
        else:
            phones = re.sub(r"\s+", " ", phones)
        if self.strip_silence:
            phones = phones.lstrip("~").rstrip("~")
        if self.inference:
            phones += "~"  # adding a silence in the end during inference produces more natural sounding prosody
        if include_eos_symbol:
            phones += "#"
        return phones

    def phonemize_from_labelfile(self, text, path_to_wavfile, include_eos_symbol=True):
        head, tail = os.path.split(path_to_wavfile)
        labelfile=tail.replace(".wav",".lab")
        print(labelfile)
        sampa_phones=[]
        phones=""
        with open(os.path.join(head.replace("aridialect_wav16000","aridialect_labels"),labelfile), encoding="utf8") as f:
            labels = f.read()
        label_lines = labels.split("\n")
        for line in label_lines:
            if line.strip() != "":
                sampa_phones.append(line[line.find("-")+1:line.find("+")])
        #print(sampa_phones)
        phones = self.sampa_to_ipa(sampa_phones)
        if self.strip_silence:
            phones = phones.lstrip("~").rstrip("~")
        #preserve final punctuation
        #print(text[len(text)-1])
        #if ';:,.!?¡¿—…"«»“”~/'.find(text[len(text)-1].strip())!=-1:
        #    phones = phones + text[len(text)-1].strip()
        #print(phones)
        return phones


    def sampa_to_ipa(self, sampa_phones):
        ipa_phones = ""
        for p in sampa_phones:
          ipa_phones = ipa_phones+self.sampa_to_ipa_dict[p]

        return ipa_phones.replace(";", ",").replace("/", " ") \
                .replace(":", ",").replace('"', ",").replace("-", ",").replace("-", ",").replace("\n", " ") \
                .replace("\t", " ").replace("¡", "").replace("¿", "").replace(",", "~")



def english_text_expansion(text):
    """
    Apply as small part of the tacotron style text cleaning pipeline, suitable for e.g. LJSpeech.
    See https://github.com/keithito/tacotron/
    Careful: Only apply to english datasets. Different languages need different cleaners.
    """
    _abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in
                      [('Mrs.', 'misess'), ('Mr.', 'mister'), ('Dr.', 'doctor'), ('St.', 'saint'), ('Co.', 'company'), ('Jr.', 'junior'), ('Maj.', 'major'),
                       ('Gen.', 'general'), ('Drs.', 'doctors'), ('Rev.', 'reverend'), ('Lt.', 'lieutenant'), ('Hon.', 'honorable'), ('Sgt.', 'sergeant'),
                       ('Capt.', 'captain'), ('Esq.', 'esquire'), ('Ltd.', 'limited'), ('Col.', 'colonel'), ('Ft.', 'fort')]]
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


if __name__ == '__main__':
    # test an English utterance
    tfr_en = TextFrontend(language="en", use_word_boundaries=False, use_explicit_eos=False,
                          path_to_phoneme_list="Preprocessing/ipa_list.txt")
    print(tfr_en.string_to_tensor("Hello world, this is a test!", view=True))

    # test a German utterance
    tfr_de = TextFrontend(language="de", use_word_boundaries=False, use_explicit_eos=False,
                          path_to_phoneme_list="Preprocessing/ipa_list.txt")
    print(tfr_de.string_to_tensor("Hallo Welt, dies ist ein Test!", view=True))


    tfr_at_lab = TextFrontend(language="at-lab", use_word_boundaries=False, use_explicit_eos=False,
                          path_to_phoneme_list="Preprocessing/ipa_list.txt")
    #uses the corresponding label file, which matches the *.wav file
    print(tfr_at_lab.string_to_tensor("Hello world, this is a test!",  view=True, path_to_wavfile="/home/mpucher/data/aridialect/aridialect_wav16000/spo_at_falter060401bis060630_001683.wav"))

