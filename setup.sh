#!/bin/bash
git-lfs pull
pip install setuptools==70.*
pip install -r requirements.txt
pip install -e git+https://github.com/ahans30/Binoculars.git@c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8#egg=Binoculars --no-dependencies