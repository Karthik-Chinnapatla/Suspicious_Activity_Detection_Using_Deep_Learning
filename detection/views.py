import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib import messages
from django.urls import reverse_lazy
from django.views.decorators.http import require_POST
from django.conf import settings

from .models import Detection
from .forms import VideoUploadForm
from .video_processor import process_surveillance_video
from .cnn_model import cnn_model_instance
from .utils import get_dashboard_statistics, create_demo_surveillance_video

def landing_view(request):
    """Public landing page view for Security Monitoring System."""
    if request.user.is_authenticated:
        return redirect('dashboard')
    context = {
        'active_page': 'landing',
    }
    return render(request, 'detection/landing.html', context)


class CustomLoginView(LoginView):
    template_name = 'registration/login.html'
    redirect_authenticated_user = True

    def form_invalid(self, form):
        messages.error(self.request, "Invalid username or password. Please try again.")
        return super().form_invalid(form)

from django.contrib.auth import logout

def custom_logout_view(request):
    """Secure logout view supporting both GET and POST requests."""
    logout(request)
    messages.info(request, "You have been logged out securely.")
    return redirect('login')

@login_required
def dashboard_view(request):
    """CCTV AI Control Center Dashboard."""
    stats = get_dashboard_statistics()
    recent_detections = Detection.objects.all()[:5]
    model_info = cnn_model_instance.get_summary_info()

    context = {
        'stats': stats,
        'recent_detections': recent_detections,
        'model_info': model_info,
        'active_page': 'dashboard',
    }
    return render(request, 'detection/dashboard.html', context)


@login_required
def detect_view(request):
    """Video Upload and Activity Detection processing view."""
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                sample_rate = int(form.cleaned_data.get('sample_rate') or 5)
                
                # Create draft model instance to save uploaded video file
                detection_obj = form.save(commit=False)
                detection_obj.user = request.user
                detection_obj.title = os.path.basename(detection_obj.uploaded_video.name)
                detection_obj.save()

                input_video_path = detection_obj.uploaded_video.path
                
                # Determine path for processed output video
                filename = os.path.basename(input_video_path)
                processed_relative = os.path.join('processed', filename)
                output_video_path = os.path.join(settings.MEDIA_ROOT, processed_relative)

                # Execute OpenCV + CNN prediction pipeline
                result = process_surveillance_video(
                    input_video_path=input_video_path,
                    output_video_path=output_video_path,
                    sample_rate=sample_rate
                )

                # Update Detection model with real predictions & metadata
                detection_obj.processed_video = processed_relative
                detection_obj.activity = result['activity']
                detection_obj.confidence = result['confidence']
                detection_obj.is_suspicious = result['is_suspicious']
                detection_obj.total_frames = result['total_frames']
                detection_obj.sampled_frames = result['sampled_frames']
                detection_obj.fps = result['fps']
                detection_obj.resolution = result['resolution']
                detection_obj.processing_time_sec = result['processing_time_sec']
                detection_obj.save()

                messages.success(request, f"Detection complete! Activity: {result['activity']} ({result['confidence']}%)")
                return redirect('result', pk=detection_obj.pk)

            except Exception as e:
                messages.error(request, f"Error processing video: {str(e)}")
        else:
            for field, errors in form.errors.items():
                for err in errors:
                    messages.error(request, f"{field.title()}: {err}")
    else:
        form = VideoUploadForm()

    context = {
        'form': form,
        'model_info': cnn_model_instance.get_summary_info(),
        'active_page': 'detect',
    }
    return render(request, 'detection/detect.html', context)


@login_required
def result_view(request, pk):
    """Displays detailed activity detection results for a processed video."""
    detection = get_object_or_404(Detection, pk=pk)
    
    context = {
        'detection': detection,
        'active_page': 'detect',
    }
    return render(request, 'detection/result.html', context)


@login_required
def history_view(request):
    """Detection log history table with search and filtering."""
    query = request.GET.get('q', '').strip()
    activity_filter = request.GET.get('activity', '').strip()

    detections = Detection.objects.all()

    if query:
        detections = detections.filter(title__icontains=query)
    
    if activity_filter:
        detections = detections.filter(activity__iexact=activity_filter)

    available_activities = cnn_model_instance.output_classes

    context = {
        'detections': detections,
        'query': query,
        'activity_filter': activity_filter,
        'available_activities': available_activities,
        'active_page': 'history',
    }
    return render(request, 'detection/history.html', context)


@login_required
@require_POST
def delete_detection_view(request, pk):
    """Deletes detection record and associated video media files."""
    detection = get_object_or_404(Detection, pk=pk)
    title = detection.title
    detection.delete()
    messages.success(request, f"Detection record '{title}' successfully deleted.")
    return redirect('history')


@login_required
def generate_demo_video_view(request):
    """Helper view to auto-generate a sample test surveillance video for evaluation."""
    activity_type = request.GET.get('type', 'Fighting')
    filename = f"demo_surveillance_{activity_type.lower()}.mp4"
    relative_path = os.path.join('uploads', 'demo', filename)
    full_path = os.path.join(settings.MEDIA_ROOT, relative_path)

    create_demo_surveillance_video(full_path, activity_type=activity_type)

    messages.info(request, f"Generated sample {activity_type} surveillance video: {filename}")
    return redirect('detect')
