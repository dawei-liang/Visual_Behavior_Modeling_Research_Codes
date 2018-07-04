# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 16:15:35 2018

@author: david
"""
import os

def check_dir(rootdir):
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)