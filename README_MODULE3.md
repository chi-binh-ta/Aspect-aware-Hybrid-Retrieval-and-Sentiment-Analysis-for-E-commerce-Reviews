# README_MODULE3: Sentiment Inference and Review Analytics

Module 3 summarizes sentiment-labeled Vietnamese e-commerce reviews and prepares analytics outputs for the final report.

## Inputs
- data/processed/sentiment_reviews.csv
- data/processed/sentiment_reviews_5class.csv
- data/reports/sentiment_label_distribution.csv

## Outputs
- results/module3/overall_sentiment_distribution.csv
- results/module3/rating_sentiment_crosstab.csv
- results/module3/top_negative_reviews.csv
- results/module3/rag_sentiment_summary.json
- figures/module3/

## Notes
Some Module 3 outputs may be regenerated from existing processed sentiment review files. The regenerated outputs are used for report assembly and descriptive analytics, not for retraining any model.

## Historical Colab Packaging Note

Earlier Colab handoff versions used a separate `ecommerce_sentiment_rag_module_3_ready.zip` package and the following unzip pattern:

```python
!unzip -q /content/ecommerce_sentiment_rag_module_3_ready.zip -d /content
```

That packaging step is historical. The current GitHub-ready project already keeps Module 3 code, outputs, and figures in the repository structure, so a separate `module_3_ready` zip is no longer required.
