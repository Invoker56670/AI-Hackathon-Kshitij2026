import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n" + "="*50)
print(f"TensorFlow Version: {tf.__version__}")
print("="*50)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✅ SUCCESS: {len(gpus)} GPU(s) DETECTED!")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.name} ({gpu.device_type})")
        
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        print(f"  Details: {details.get('device_name', 'Unknown')}")
    except:
        pass
    print("\nUse this environment for training to accelerate performance.")
else:
    print("\n⚠️ WARNING: NO GPU DETECTED.")
    print("Check your drivers or CUDA installation.")
print("="*50 + "\n")
