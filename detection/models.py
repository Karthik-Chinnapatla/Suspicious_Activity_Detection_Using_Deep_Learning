import os
from django.db import models
from django.contrib.auth.models import User
from django.core.files.storage import default_storage

class Detection(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='detections', null=True, blank=True)
    title = models.CharField(max_length=255, help_text="Filename or custom title")
    uploaded_video = models.FileField(upload_to='uploads/%Y/%m/%d/')
    processed_video = models.FileField(upload_to='processed/%Y/%m/%d/', null=True, blank=True)
    
    activity = models.CharField(max_length=100, default='Normal')
    confidence = models.FloatField(default=0.0, help_text="Percentage confidence score")
    is_suspicious = models.BooleanField(default=False)
    
    total_frames = models.IntegerField(default=0)
    sampled_frames = models.IntegerField(default=0)
    fps = models.FloatField(default=30.0)
    resolution = models.CharField(max_length=50, blank=True, default='')
    processing_time_sec = models.FloatField(default=0.0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Activity Detection'
        verbose_name_plural = 'Activity Detections'

    def __str__(self):
        return f"{self.title} - {self.activity} ({self.confidence:.1f}%)"

    @property
    def badge_color(self):
        """Returns CSS class for activity badges."""
        if self.activity.upper() in ['FIGHTING', 'FIRE', 'BURGLARY', 'ROBBERY', 'SHOOTING', 'WEAPON']:
            return 'danger'
        elif self.activity.upper() in ['NORMAL']:
            return 'success'
        return 'warning'

    @property
    def status_label(self):
        """Returns readable status label."""
        return 'SUSPICIOUS' if self.is_suspicious else 'NORMAL'

    def delete(self, *args, **kwargs):
        """Clean up video files from storage when database entry is deleted."""
        if self.uploaded_video and os.path.isfile(self.uploaded_video.path):
            os.remove(self.uploaded_video.path)
        if self.processed_video and os.path.isfile(self.processed_video.path):
            os.remove(self.processed_video.path)
        super().delete(*args, **kwargs)
