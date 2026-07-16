from django import forms


class PredictionForm(forms.Form):
    cylinders = forms.FloatField(label="Cylindres", min_value=1, max_value=12)
    displacement = forms.FloatField(label="Cylindrée", min_value=1)
    horsepower = forms.FloatField(label="Puissance", min_value=1)
    weight = forms.FloatField(label="Poids", min_value=1)
    acceleration = forms.FloatField(label="Accélération", min_value=1)
    model_year = forms.DateField(
        label="Année du modèle",
        help_text="Choisis une date dans l'année souhaitée. Seule l'année sera utilisée.",
        widget=forms.DateInput(attrs={"type": "date"}),
    )
    origin = forms.ChoiceField(
        label="Origine",
        choices=[
            ("USA", "USA"),
            ("Europe", "Europe"),
            ("Japan", "Japan"),
        ],
        initial="USA",
    )

    def clean_model_year(self):
        year = self.cleaned_data["model_year"].year
        return year % 100
