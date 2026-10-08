#!/bin/bash
# Compile figures/src/fig_*.tex (standalone TikZ) and render each to a 300-dpi PNG in figures/.
cd "$(dirname "$0")/src" || exit 1
mkdir -p ../build
status=0
for f in ${@:-fig_*.tex}; do
  n=$(basename "$f" .tex)
  if pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build "$n.tex" > ../build/$n.out 2>&1; then
    pdftoppm -r 300 -png -singlefile ../build/$n.pdf ../$n
    python3 -c "from PIL import Image; im=Image.open('../$n.png'); print('ok  $n', im.size, '%.1f x %.1f cm' % (im.size[0]/300*2.54, im.size[1]/300*2.54))"
  else
    echo "FAIL $n"; grep -A6 '^!' ../build/$n.log | head -24; status=1
  fi
done
exit $status
