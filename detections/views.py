from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.models import User
import uuid
import json

from .models import PostureDetection, FaceDetection, DetectionSession
from .ml_utils import detect_posture, detect_faces, base64_to_image


@api_view(['POST'])
@permission_classes([AllowAny])  # Change to IsAuthenticated if you want authentication
def detect_posture_api(request):
    """
    API endpoint to detect posture from image
    
    Expected payload:
    {
        "image": "base64_encoded_image_string",
        "session_id": "optional_session_id"
    }
    """
    try:
        image_base64 = request.data.get('image')
        if not image_base64:
            return Response(
                {'error': 'Image data is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Convert base64 to image array
        image_array = base64_to_image(image_base64)
        
        # Detect posture
        result = detect_posture(image_array)
        
        if result is None:
            return Response(
                {'error': 'No landmarks detected in image'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get session_id or create new one
        session_id = request.data.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get user if authenticated
        user = request.user if request.user.is_authenticated else None
        
        # Save to database
        posture_detection = PostureDetection.objects.create(
            user=user,
            posture_label=result['posture_label'],
            confidence=result.get('confidence'),
            session_id=session_id
        )
        
        return Response({
            'success': True,
            'posture_label': result['posture_label'],
            'confidence': result.get('confidence'),
            'session_id': session_id,
            'detection_id': posture_detection.id,
            'timestamp': posture_detection.timestamp
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])  # Change to IsAuthenticated if you want authentication
def detect_faces_api(request):
    """
    API endpoint to detect faces from image
    
    Expected payload:
    {
        "image": "base64_encoded_image_string",
        "session_id": "optional_session_id"
    }
    """
    try:
        image_base64 = request.data.get('image')
        if not image_base64:
            return Response(
                {'error': 'Image data is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Convert base64 to image array
        image_array = base64_to_image(image_base64)
        
        # Detect faces
        result = detect_faces(image_array)
        
        # Get session_id or create new one
        session_id = request.data.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get user if authenticated
        user = request.user if request.user.is_authenticated else None
        
        # Save to database
        face_detection = FaceDetection.objects.create(
            user=user,
            face_count=result['face_count'],
            detections=result['detections'],
            session_id=session_id
        )
        
        return Response({
            'success': True,
            'face_count': result['face_count'],
            'detections': result['detections'],
            'session_id': session_id,
            'detection_id': face_detection.id,
            'timestamp': face_detection.timestamp
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def detect_both_api(request):
    """
    API endpoint to detect both posture and faces from image in one call
    
    Expected payload:
    {
        "image": "base64_encoded_image_string",
        "session_id": "optional_session_id"
    }
    """
    try:
        image_base64 = request.data.get('image')
        if not image_base64:
            return Response(
                {'error': 'Image data is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Convert base64 to image array (only once)
        image_array = base64_to_image(image_base64)
        
        # Detect both posture and faces
        posture_result = detect_posture(image_array)
        face_result = detect_faces(image_array)
        
        # Get session_id or create new one
        session_id = request.data.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get user if authenticated
        user = request.user if request.user.is_authenticated else None
        
        # Save posture to database
        posture_detection = None
        if posture_result:
            posture_detection = PostureDetection.objects.create(
                user=user,
                posture_label=posture_result['posture_label'],
                confidence=posture_result.get('confidence'),
                session_id=session_id
            )
        
        # Save face detection to database
        face_detection = FaceDetection.objects.create(
            user=user,
            face_count=face_result['face_count'],
            detections=face_result['detections'],
            session_id=session_id
        )
        
        return Response({
            'success': True,
            'posture': {
                'posture_label': posture_result['posture_label'] if posture_result else None,
                'confidence': posture_result.get('confidence') if posture_result else None,
            },
            'faces': {
                'face_count': face_result['face_count'],
                'detections': face_result['detections'],
            },
            'session_id': session_id,
            'timestamps': {
                'posture': posture_detection.timestamp if posture_detection else None,
                'faces': face_detection.timestamp
            }
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])  # You might want to restrict this to authenticated users
def get_session_detections(request, session_id):
    """
    Get all detections for a specific session
    """
    try:
        posture_detections = PostureDetection.objects.filter(session_id=session_id)
        face_detections = FaceDetection.objects.filter(session_id=session_id)
        
        return Response({
            'session_id': session_id,
            'posture_detections': [
                {
                    'id': d.id,
                    'posture_label': d.posture_label,
                    'confidence': d.confidence,
                    'timestamp': d.timestamp
                }
                for d in posture_detections
            ],
            'face_detections': [
                {
                    'id': d.id,
                    'face_count': d.face_count,
                    'detections': d.detections,
                    'timestamp': d.timestamp
                }
                for d in face_detections
            ]
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

