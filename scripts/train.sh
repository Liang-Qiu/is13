#! /bin/bash

source venv/bin/activate
for counter in $(seq 1 39)
do
    python examples/bilstm-lm.py --with_lm False  --with_glove False --bi_lstm False --nsentences $((counter * 100))
    python examples/bilstm-lm.py --with_lm True  --with_glove False --bi_lstm False --nsentences $((counter * 100))
    python examples/bilstm-lm.py --with_lm True  --with_glove False --bi_lstm True --nsentences $((counter * 100))
    python examples/bilstm-lm.py --with_lm True  --with_glove True --bi_lstm True --nsentences $((counter * 100))
done

echo All done
