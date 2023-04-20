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
parser.add_argument("--file", required=False, default="", type=str, help="The json.")
parser.add_argument("--outfile", required=False, default="results.json", type=str, help="The json.")
parser.add_argument("--model", required=False, default="tiny.en", type=str, help="The model path or type (medium, large).")
parser.add_argument("--beam_size", required=False, default=None,type=int, help="The beam search size.")
parser.add_argument("--best_of", required=False, default=None, type=int, help="N best.")
parser.add_argument("--temperature", required=False, default=0.0, type=float, help="Temperature.")
parser.add_argument("--multiple_samples", required=False, action='store_true', help="Keep multiple samples.")

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

count_line = 0
if os.path.exists(args["outfile"]):
    with open(args["outfile"],'r') as g:
        for line in g.readlines():
            count_line += 1

with open(args["file"],"r") as f:
    with open(args["outfile"],'a') as g:
        for line in f.readlines():
            if count >= count_line:
                dico = json.loads(line)
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
