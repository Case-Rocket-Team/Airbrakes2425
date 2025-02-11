import math

def convert_motor_angle_to_encoder_counts(initial_angle):
    distance = 2*math.cos(initial_angle) + math.sqrt(3.0625 - (2*math.sin(initial_angle) - 0.35)**2)
    motor_revolutions = distance * 8
    encoder_counts = motor_revolutions * 921.744

    return encoder_counts
