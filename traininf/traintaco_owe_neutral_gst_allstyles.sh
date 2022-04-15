#!/bin/bash

for f in styles/*.wav
do
  echo $f;
  style=`basename $f | cut -f1 -d"_"`;
  echo $style;
  python run_utts_to_file_reader.py --config traininf/taco_owe_neutral_gst.ini --style_embed_wav $f;
done
