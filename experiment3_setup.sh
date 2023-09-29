#!/bin/bash

echo "Downloading pre-trained filtering phase models ..."
curl -o models.tar.gz -L "https://zenodo.org/record/8366378/files/models.tar.gz?download=1"
tar -xf models.tar.gz
