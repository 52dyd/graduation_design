#!/bin/zsh
rm -rf ./tmp
mkdir tmp
cd ./tmp
sudo perf record -F 99 -a -g -- sleep 30 
sudo perf script > out.perf
/MSF/space_lzj/tmp/FlameGraph-1.0/stackcollapse-perf.pl out.perf > out.folded
/MSF/space_lzj/tmp/FlameGraph-1.0/flamegraph.pl out.folded > kernel.svg
ifconfig | grep inet.222
python -m http.server 8888
