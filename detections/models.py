from django.db import models
from django.contrib.auth.models import User


class PostureDetection(models.Model):
    """Model to store posture detection results"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    posture_label = models.CharField(max_length=50)  # appropriate, cheating, defensive
    confidence = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(max_length=100, null=True, blank=True)  # To group detections by session
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['session_id']),
        ]
    
    def __str__(self):
        return f"{self.posture_label} - {self.timestamp}"


class FaceDetection(models.Model):
    """Model to store face detection results"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    face_count = models.IntegerField(default=0)
    detections = models.JSONField(default=list)  # Store detection details (bbox, confidence, etc.)
    timestamp = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(max_length=100, null=True, blank=True)  # To group detections by session
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['session_id']),
        ]
    
    def __str__(self):
        return f"{self.face_count} faces - {self.timestamp}"


class DetectionSession(models.Model):
    """Model to track interview/detection sessions"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=100, unique=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"Session {self.session_id} - {self.started_at}"

