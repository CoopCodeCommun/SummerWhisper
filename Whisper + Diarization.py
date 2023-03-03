#!/usr/bin/env python
# coding: utf-8

# In[37]:


import whisper, json
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=True)
# Check si CUDA ok
import torch
torch.cuda.is_available()


# In[38]:


pathfile="./AlainWOrk/wav/e.wav"
diarization_result = pipeline(pathfile)
print('diarization_result ok')

model = whisper.load_model("large-v2")
print("Fait chauffer le gpu Marcel ! ")

whisper_res = model.transcribe(pathfile, verbose=True)

print("whisper_res OK")

# with open("jmarc.wav.json", "r") as f:
#     whisper_res = json.load(f)


# In[53]:


from pyannote.core import Segment, Annotation, Timeline


def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res['segments']:
        start = item['start']
        end = item['end']
        text = item['text']
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text


def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence


PUNC_SENT_END = []
# PUNC_SENT_END = ['.', '?', '!']


def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk

        elif text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text


def diarize_text(transcribe_res, diarization_result):
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed


def write_to_txt(spk_sent, file):
    with open(file, 'w') as fp:
        for seg, spk, sentence in spk_sent:
            line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sentence}\n'
            fp.write(line)
        
        
def add_space_before_punctuation(text):
    return re.sub(r'(\w)([?!])', r'\1 \2', text)


def replace_words(text):

    word_dict = {
        "on a": "nous avons",
        "on est": "nous sommes",
        "on vient": "nous venons",
        "on va": "nous allons",
        "on obtient": "nous obtenons",
        "on fait": "nous faisons",
        "on coordonne": "nous coordonnons",
        "on se frotte": "nous nous frottons",
        "on se donne": "nous nous donnons",
        "on mange": "nous mangeons",
        "on donne": "nous donnons",
        "on apprend": "nous apprenons",
        "on s'est dit": "nous nous sommes dit",
        "on sait": "nous savons",
        "on prend": "nous prenons",
        "ça": "cela",
        "est-ce que vous avez": "avez-vous",
        "est-ce qu'il y a": "y a-t-il ",
        "puisqu'on voit": "puisque nous voyons",
        "qu'on voit": "que nous voyons",
        "comment on peut": "comment pouvons-nous",
        "on participe": "nous participons",
        "-là": "",
        "on ne s'est pas": "nous ne nous sommes pas",
        "qu'on n'était pas": "que nous n'étions pas",
        "qu'on était pas": "que nous n'étions pas",
        "qu'on ne mettait pas": "que nous ne mettions pas",
        "qu'on mettait pas": "que nous ne mettions pas",
        "on peut se": "nous pouvons nous",
        "qu'on en a": "que nous en avons",
        "est-ce qu'on peut": "pouvons-nous",
    }
    

    for old_word, new_word in word_dict.items():
        pattern = r'\b(' + old_word + r')\b'                   
        pattern_case_insensitive = re.compile(pattern, re.IGNORECASE)
        text = re.sub(pattern_case_insensitive, lambda m: new_word.capitalize() if m.group(1)[0].isupper() else new_word.lower(), text)
        text = add_space_before_punctuation(text)
    return text                                                             
                        


# In[55]:


final_result = diarize_text(whisper_res, diarization_result)

import re
ex_speaker = None
for segment, speaker, text in final_result:
#     if speaker == None:
#         speaker = ex_speaker
        
    if speaker != ex_speaker:
        print("")
#         print(f'{segment}')
        print(f'{speaker}')
        print("")
        ex_speaker = speaker
    print(replace_words(re.sub(r'^\ ', '',text)))


# In[ ]:




