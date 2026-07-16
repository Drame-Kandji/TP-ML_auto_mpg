from django.shortcuts import render

from .forms import PredictionForm
from .model_service import predict_mpg


def predict_view(request):
    prediction = None
    form = PredictionForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        prediction = predict_mpg(form.cleaned_data)

    return render(
        request,
        "predictor/index.html",
        {
            "form": form,
            "prediction": prediction,
        },
    )
