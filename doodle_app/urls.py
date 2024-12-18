from django.urls import path
from .views import landing_page, canvas_page,contact,ai,classify_doodle,get_random_object


urlpatterns = [
    path('', landing_page, name='landing_page'),
    path('canvas/', canvas_page, name='canvas_page'),
    path('classify_doodle/', classify_doodle, name='classify_doodle'),
    path('get_random_object/', get_random_object, name='get_random_object'),
    path('contact/', contact, name='contact'),
    path('ai/', ai, name='ai'),
]
