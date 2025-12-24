from django.urls import path
from . import views

urlpatterns = [
    path('posture/', views.detect_posture_api, name='detect_posture'),
    path('faces/', views.detect_faces_api, name='detect_faces'),
    path('both/', views.detect_both_api, name='detect_both'),
    path('session/<str:session_id>/', views.get_session_detections, name='get_session_detections'),
]

