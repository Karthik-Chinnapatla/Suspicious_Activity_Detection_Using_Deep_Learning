from django.contrib import admin
from .models import Detection

@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('title', 'activity', 'confidence', 'is_suspicious', 'user', 'total_frames', 'created_at')
    list_filter = ('activity', 'is_suspicious', 'created_at')
    search_fields = ('title', 'activity', 'user__username')
    readonly_fields = ('total_frames', 'sampled_frames', 'fps', 'resolution', 'processing_time_sec', 'created_at', 'updated_at')
    ordering = ('-created_at',)
