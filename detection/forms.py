import os
from django import forms
from .models import Detection

ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
MAX_FILE_SIZE_MB = 100

class VideoUploadForm(forms.ModelForm):
    sample_rate = forms.ChoiceField(
        choices=[
            (3, 'High Accuracy (Sample every 3rd frame)'),
            (5, 'Balanced - Recommended (Sample every 5th frame)'),
            (10, 'Fast Processing (Sample every 10th frame)'),
        ],
        initial=5,
        required=False,
        widget=forms.Select(attrs={'class': 'form-select security-input', 'id': 'sample_rate_select'}),
        help_text="Select frame extraction sampling rate for performance vs accuracy tradeoff."
    )

    class Meta:
        model = Detection
        fields = ['uploaded_video']
        widgets = {
            'uploaded_video': forms.FileInput(attrs={
                'class': 'form-control security-input',
                'accept': 'video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm',
                'id': 'video_file_input',
            })
        }

    def clean_uploaded_video(self):
        video = self.cleaned_data.get('uploaded_video')
        if not video:
            raise forms.ValidationError("Please select a valid surveillance video file.")

        ext = os.path.splitext(video.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise forms.ValidationError(
                f"Unsupported video format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        if video.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise forms.ValidationError(
                f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB} MB."
            )

        return video
