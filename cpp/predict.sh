#!/bin/bash
/usr/bin/time -f "rss=%M elapsed=%E" ./main litemedsam-encoder.xml litemedsam-decoder.xml efficientvit-encoder.xml efficientvit-decoder.xml /workspace/outputs/ /workspace/inputs/ /workspace/outputs/

