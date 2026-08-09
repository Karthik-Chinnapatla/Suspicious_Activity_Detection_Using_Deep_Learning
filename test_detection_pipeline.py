import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.conf import settings
from detection.utils import create_demo_surveillance_video
from detection.video_processor import process_surveillance_video
from detection.models import Detection
from django.contrib.auth.models import User

def test_pipeline():
    admin_user = User.objects.get(username='admin')

    tests = [
        ("Fire", "surveillance_fire_hazard.mp4", "Fire"),
        ("Fighting", "cctv_fighting_brawl.mp4", "Fighting"),
        ("Normal", "normal_hallway_walk.mp4", "Normal"),
    ]

    print("\n=======================================================")
    print("--- RUNNING MULTI-CLASS ACTIVITY DETECTION TEST SUITE ---")
    print("=======================================================")

    for target_activity, filename, expected in tests:
        input_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
        create_demo_surveillance_video(input_path, activity_type=target_activity, duration_sec=3)
        
        output_path = os.path.join(settings.MEDIA_ROOT, 'processed', filename)
        res = process_surveillance_video(input_path, output_path, sample_rate=3)

        print(f"\n[Video: {filename}]")
        print(f"Target Activity   : {target_activity}")
        print(f"Detected Activity : {res['activity']}")
        print(f"Confidence Score  : {res['confidence']}%")
        print(f"Is Suspicious     : {res['is_suspicious']}")

        assert res['activity'] == expected, f"Failed for {filename}: Expected {expected}, got {res['activity']}"

    print("\n[OK] ALL MULTI-CLASS DETECTION TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_pipeline()
