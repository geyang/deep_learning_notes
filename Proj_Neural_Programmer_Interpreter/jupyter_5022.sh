#!/usr/bin/env bash
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- jupyter notebook --ip=* --port=5022 --no-browser
