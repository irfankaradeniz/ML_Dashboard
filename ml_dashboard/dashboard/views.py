from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from .models import Dataset
from .forms import DatasetUploadForm, MLForm
from .ml_functions import train_model, test_model
import pickle
import pandas as pd
import base64

def index(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('dashboard:index')
    else:
        form = DatasetUploadForm()

    datasets = Dataset.objects.all()
    return render(request, 'index.html', {'form': form, 'datasets': datasets})

def ml_process(request, dataset_id):
    dataset = Dataset.objects.get(pk=dataset_id)

    if request.method == 'POST':
        form = MLForm(request.POST)
        if form.is_valid():
            results = train_model(dataset, form.cleaned_data)
            return redirect('dashboard:train', dataset_id=dataset_id)  # Change to use the 'train' view function
    else:
        form = MLForm()

    return render(request, 'train.html', {'form': form, 'dataset': dataset})  # Change the template to 'train.html'

def train(request, dataset_id):
    if request.method == 'POST':
        form = MLForm(request.POST)
        if form.is_valid():
            dataset = Dataset.objects.get(pk=dataset_id)
            results = train_model(dataset, form.cleaned_data)
            
            request.session['model'] = base64.b64encode(pickle.dumps(results['model'])).decode()
            request.session['X_test'] = results['X_test'].to_json()
            request.session['y_test'] = results['y_test'].to_json()
            
            return redirect('dashboard:test')
    else:
        form = MLForm()
    return render(request, 'train.html', {'form': form, 'dataset_id': dataset_id})

def test(request):
    if 'model' not in request.session:
        return redirect('dashboard:index')
        
    model = pickle.loads(base64.b64decode(request.session['model'].encode()))
    X_test = pd.read_json(request.session['X_test'])
    y_test = pd.read_json(request.session['y_test'], typ='series')
    
    results = test_model(None, {'model': model, 'X_test': X_test, 'y_test': y_test})
    
    return render(request, 'results.html', {'results': results})

def upload(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            dataset = Dataset.objects.last()
            return redirect('dashboard:train', dataset_id=dataset.pk)
    else:
        form = DatasetUploadForm()

    return render(request, 'upload.html', {'form': form})

def delete_dataset(request, dataset_id):
    dataset = Dataset.objects.get(pk=dataset_id)
    dataset.file.delete()  # This deletes the file from the file system
    dataset.delete()  # This deletes the dataset entry from the database
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))