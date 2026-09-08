# WWW 2027 anonymous manuscript source

This directory uses the official ACM `acmart` class with the option required by
the WWW 2027 Research Track:

```tex
\documentclass[sigconf,anonymous,review]{acmart}
```

The official call requires an English double-column manuscript with eight main
pages; references and an optional appendix may bring the PDF to twelve pages.
The source deliberately uses the class supplied by Overleaf/TeX Live rather
than committing a third-party class file. Compile with `latexmk -pdf main.tex`.

Rows marked `TBD` are evidence gates, not claims. Replace them only with the
artifacts written by `experiments/cross_dataset/run_directional_calibration.sh`
and the matched-seed training grid.
