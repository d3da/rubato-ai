#!/bin/bash
#
# Download v3.0.0 of the maestro dataset from magenta.
# requires wget and unzip installed
set -e

cd $( dirname $0 )

echo ">>> Downloading maestro-v3.0.0-midi.zip"
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -u maestro-v3.0.0-midi.zip
echo "<<< Done"
