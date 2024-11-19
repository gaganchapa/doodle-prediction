from django.urls import path
from .views import landing_page, canvas_page,contact,ai,classify_doodle

urlpatterns = [
    path('', landing_page, name='landing_page'),
    path('canvas/', canvas_page, name='canvas_page'),
    path('classify_doodle/', classify_doodle, name='classify_doodle'),
    path('contact/', contact, name='contact'),
    path('ai/', ai, name='ai'),

]
