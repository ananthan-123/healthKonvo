from django.http import response

from .predict import predictOut

from django.http.response import JsonResponse, HttpResponseRedirect
from django.shortcuts import redirect
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
# Create your views here.

# renders home page


@ csrf_exempt
def home(request):
    return render(request, 'index.html')


@ csrf_exempt
def get_bot_response(request):
    userText = request.GET.get('msg')
    out = predictOut(userText)
    return JsonResponse({'res': out})
