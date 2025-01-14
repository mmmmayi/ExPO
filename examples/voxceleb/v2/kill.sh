#!/bin/bash
kill $(ps aux | grep wespeaker/bin/train.py | grep -v grep | awk '{print $2}')

