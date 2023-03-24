from django import forms
from .models import Dataset

ALGORITHM_CHOICES = [
    ('', 'Choose an algorithm'),
    ('dt', 'Decision Tree'),
    ('rf', 'Random Forest'),
    ('lr', 'Linear Regression'),
    ('ridge', 'Ridge Regression'),
]

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ('file',)

class MLForm(forms.Form):
    target_column = forms.CharField(label='Target column', max_length=255)
    feature_columns = forms.CharField(label='Feature columns (comma-separated)', max_length=255)
    task = forms.ChoiceField(label='Task', choices=[('classification', 'Classification'), ('regression', 'Regression')])
    algorithm = forms.ChoiceField(label='Algorithm', choices=ALGORITHM_CHOICES)
    train_test_split = forms.FloatField(label='Train-test split percentage', min_value=0.1, max_value=0.9, initial=0.7)

    def clean_feature_columns(self):
        feature_columns = self.cleaned_data['feature_columns']
        return [col.strip() for col in feature_columns.split(',')]

