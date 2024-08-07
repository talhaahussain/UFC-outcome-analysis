# UFC-outcome-analysis
Continuous Assessment for ECM3420 - Learning From Data, set by Dr. Chico Camargo, Dr. Diogo Pacheco and Dr. Marcos Oliveira (Year 3, Semester 1). Involves the use of machine learning methods, specifically a multi-layer perceptron (MLP), to explore which is the best predictor of the outcome of a UFC fight - the fighters' physical metrics or their historical data.

This work received a final mark of 75/100.

Please see `specification.md` for specification. (Unfortunately, original specification does not exist; this is a replica.)

### Prerequisites

`pandas`, `numpy` and `sklearn` are required to run `src/physical-fp.py` and `src/historical-fp.py`. These can be installed with:

```bash
pip install -r requirements.txt
```

### Usage

Please run Python source files with

```
python physical-fp.py
```

and

```
python historical-fp.py
```

Results are printed to `stdout`, and can be redirected to a file if you wish.

### Results

Please see `doc/report.pdf` and `doc/slides.pdf` for results. A YouTube video discussing the results is also available; please click <a href="https://youtu.be/tM9piZdOQkc">here</a> for the link. 

### Footnote

This research makes use of a dataset that has not been included due to size limitations. The dataset can be accessed <a href="https://www.kaggle.com/datasets/rajeevw/ufcdata">here</a>. All credits go to their respective owners.
