from django.contrib import admin
from django.urls import path
from camera_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home_view, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Rotas da câmera
    path('camera/', views.camera_view, name='camera'),
    path("process_frame/", views.process_frame, name="process_frame"),
    path('configuracao/', views.editar_configuracao, name='configuracao'),
    path("salvar_intervalo/", views.salvar_intervalo, name="salvar_intervalo"),
]
