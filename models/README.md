# Model Artifacts

## Files
- `pipeline.joblib`: Serialized scikit-learn Pipeline containing `SurveyFeatureTransformer` and `VotingClassifier` ensemble of CatBoost, LightGBM, and XGBoost.
- `metadata.json`: Model version, training parameters, ensemble weights (`[1.5, 1.0, 1.0]`), optimal decision threshold calibrated on the validation split, and aggregated feature importances.

## Inference Usage
```python
from ml.predict import CatGBMPredictor

predictor = CatGBMPredictor()
result = predictor.predict_sample({
    'GENDER': 'M',
    'AGE': 65,
    'SMOKING': 2,
    # ... remaining 12 symptoms
})
print(result['predicted_class'], result['probability_percentage'])
```
