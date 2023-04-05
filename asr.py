#!/usr/bin/env python3
import whisper
import argparse
import json
import numpy as np
import time
import os
from tqdm import tqdm

def custom_transcribe(path,model,options):
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    return  whisper.decode(model, mel, options)

# Parsing arguments
parser = argparse.ArgumentParser(
    description='ASR inference.'
)
parser.add_argument("--max_inference_time", required=False, default=60, type=int, help="Max time in minutes")
parser.add_argument("--file", required=False, default="Lucas/dev_manifest_human_conformer_ctc_large_normalized.json", type=str, help="The json.")
parser.add_argument("--outfile", required=False, default="results.json", type=str, help="The json.")
parser.add_argument("--model", required=False, default="small.en", type=str, help="The model path or type (medium, large).")
parser.add_argument("--beam_size", required=False, default=None,type=int, help="The beam search size.")
parser.add_argument("--best_of", required=False, default=None, type=int, help="N best.")
parser.add_argument("--temperature", required=False, default=0.0, type=float, help="Temperature.")
parser.add_argument("--multiple_samples", required=False, action='store_true', help="Keep multiple samples.")

#parser.add_argument("--patience", required=False, default=0.0, type=float, help="Patience.")
#parser.add_argument("--temperature_increment_on_fallback", required=False, default=0.2, type=float, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below ")
#parser.add_argument("--compression_ratio_threshold", required=False, default=2.4, type=float, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
#parser.add_argument("--logprob_threshold", required=False, default=2.4, type=float, help="if the average log probability is lower than this value, treat the decoding as failed")
#parser.add_argument("--no_speech_threshold", required=False, default=0.6, type=float, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

args = parser.parse_args().__dict__
max_inference_time = args["max_inference_time"]
model = whisper.load_model(args["model"])

start = time.time()
count = 0
options = whisper.DecodingOptions(
        language="en", 
        fp16=False, # for CPU usage
        without_timestamps=True,
        beam_size=args["beam_size"], 
        best_of=args['best_of'], 
        temperature=args['temperature'],
        multiple_samples=args['multiple_samples']
        )

#wav_file = 'audio_samples/pmul4029/Turn-7.wav' # sacajawea hotel
#wav_file = 'audio_samples/mul2418/Turn-3.wav' # bouchon
#wav_file = 'audio_samples/pmul4583/Turn-3.wav' # neylandville
#wav_file = 'audio_samples/mul0384/Turn-1.wav' # rolfe's chop house
#results = custom_transcribe(wav_file, model, options)
#for result in results:
#    pred_text = result.text
#    bpe_tokens = result.bpe_tokens
#    bpe_logprobs = [round(p, 5) for p in result.bpe_logprobs]
#    sum_logprobs = result.sum_logprobs
#    print(pred_text)
#    print(bpe_tokens)
#    print(bpe_logprobs)
#    print(sum_logprobs)

#print(result.audio_features.shape)

count_line = 0
if os.path.exists(args["outfile"]):
    with open(args["outfile"],'r') as g:
        for line in g.readlines():
            count_line += 1

with open(args["file"],"r") as f:
    with open(args["outfile"],'a') as g:
        for line in f.readlines():
        #for line in tqdm(f.readlines()):
            if count >= count_line:
                dico = json.loads(line)
                #print(dico)
                results = custom_transcribe(dico["audio_filepath"], model, options)
                dico['samples'] = [
                        {
                            'text': result.text,
                            'sum_probs': round(result.sum_logprobs, 4),
                            'toks': result.bpe_tokens,
                            'probs': [round(p, 4) for p in result.bpe_logprobs],
                            }
                        for result in results
                        ]
                json.dump(dico, g)
                g.write("\n")
            else:
                pass
            count += 1
            if time.time() - start > max_inference_time*60:
                print("Exiting for requeue")
                exit(42)

