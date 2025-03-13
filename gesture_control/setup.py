from setuptools import setup, find_packages

package_name = 'gesture_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),  # 🔥 Utilisation correcte pour détecter les scripts
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='samar',
    maintainer_email='samar@example.com',
    description='Gesture-based control for TurtleSim',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_control = gesture_control.gesture_control:main',
            'robot_controller = gesture_control.robot_controller:main',
        ],
    },
)

