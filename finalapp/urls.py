from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='land'),
    path('link/',views.link, name='link'),
    path('load/',views.load, name='load'),
    path('history/',views.show_db, name='show_db'),
    path('play/',views.play,name="play"),
    path('playing/', views.process_youtube_video, name='process_youtube_video'),
]