import math


def bat_dynamic(motor_eff_2d, r_w, soc, force, x_dot, bat_eff_dis, bat_eff_cha, dt, bat_q):
    """
    Function to calculate the battery dynamics
    Args:
        motor_eff_2d: motor efficiency function
        soc: state of charge current time step
        motor_torque: motor torque current time step
        rpm: motor rpm current time step
        bat_eff_dis: battery discharge efficiency function
        bat_eff_cha: battery charge efficiency function
        dt: time step
        bat_q: battery capacity
    Returns:
        next_soc: state of charge next time step
    """

    if force > 0:
        total_eff = bat_eff_dis([soc]) * motor_eff_2d((force * r_w, abs(x_dot) * 60 / (2 * math.pi * r_w)))
        bat_power = -1 / total_eff * force * abs(x_dot)
    else:
        total_eff = bat_eff_cha([soc]) * motor_eff_2d((force * r_w, abs(x_dot) * 60 / (2 * math.pi * r_w)))
        bat_power = -total_eff * force * abs(x_dot)
    next_soc = soc + bat_power * dt / bat_q
    return next_soc


def pb_cal(motor_eff_2d, force, x_dot, soc, r_w, bat_eff_dis, bat_eff_cha):
    """
    Function to calculate the power of battery
    Args:
        motor_eff_2d: motor efficiency function
        force: force current time step
        x_dot: speed current time step
        soc: state of charge current time step
        r_w: wheel radius
        bat_eff_dis: battery discharge efficiency function
        bat_eff_cha: battery charge efficiency function
    Returns:
        pb: power of battery
    """
    if force > 0:
        total_eff = motor_eff_2d((force * r_w, abs(x_dot) * 60 / (2 * math.pi * r_w))) * bat_eff_dis([soc])
        pb = force * x_dot / total_eff
    else:
        total_eff = motor_eff_2d((force * r_w, abs(x_dot) * 60 / (2 * math.pi * r_w))) * bat_eff_cha([soc])
        pb = force * x_dot * total_eff
    return pb
