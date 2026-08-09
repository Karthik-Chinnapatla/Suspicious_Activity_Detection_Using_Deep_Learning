from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_view, name='landing'),
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('logout/', views.custom_logout_view, name='logout'),
    
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('detect/', views.detect_view, name='detect'),
    path('result/<int:pk>/', views.result_view, name='result'),
    path('history/', views.history_view, name='history'),
    path('history/delete/<int:pk>/', views.delete_detection_view, name='delete_detection'),
    
    path('demo/generate/', views.generate_demo_video_view, name='generate_demo_video'),
]
