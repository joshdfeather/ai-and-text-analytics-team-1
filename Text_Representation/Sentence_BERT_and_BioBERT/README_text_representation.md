Text Representation Experiments

This folder contains my individual contribution to the PubMedQA medical question-answering coursework.

Models explored
- TF-IDF baseline
- TF-IDF with n-grams
- Sentence-BERT
- BioBERT

Main files
- `notebooks/text_representation_sbert_biobert.ipynb`
- `outputs/sentence_bert/`
- `outputs/biobert/`
- `analysis/text_representation_error_analysis.md`

Main finding
Dense representations such as Sentence-BERT and BioBERT capture more semantic information than sparse TF-IDF features, but both models still struggled with the minority "maybe" class. This supports the group report’s argument that PubMedQA requires deeper question-context interaction and careful error analysis.
